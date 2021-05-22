import librosa
import numpy as np
import torch
import torchaudio
from torch import distributions
from torchvision import transforms

from config import Params, MelSpectrogramConfig


class MelSpectrogram(torch.nn.Module):
    def __init__(self, config):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()

        return mel


class AddNormalNoise(object):
    def __init__(self):
        self.var = Params.noise_variance

    def __call__(self, wav):
        noiser = distributions.Normal(0, self.var)
        if np.random.uniform() < 0.5:
            wav += noiser.sample(wav.size())
        return wav.clamp(-1, 1)


class TimeStretch(object):
    def __init__(self):
        self.min_scale = Params.min_time_stretch
        self.max_scale = Params.max_time_stretch

    def __call__(self, wav):
        random_stretch = np.random.uniform(self.min_scale, self.max_scale, 1)[0]
        if np.random.uniform() < 0.5:
            wav_stretched = librosa.effects.time_stretch(wav.numpy(), random_stretch)
        else:
            wav_stretched = wav.numpy()
        return torch.from_numpy(wav_stretched)


class PitchShifting(object):
    def __init__(self):
        self.sample_rate = Params.sample_rate
        self.min_shift = Params.min_shift
        self.max_shift = Params.max_shift

    def __call__(self, wav):
        random_shift = np.random.uniform(self.min_shift, self.max_shift, 1)[0]
        if np.random.uniform() < 0.5:
            wav_shifted = librosa.effects.pitch_shift(
                wav.numpy(), self.sample_rate, random_shift
            )
        else:
            wav_shifted = wav.numpy()
        return torch.from_numpy(wav_shifted)


class NormalizePerFeature(object):
    """
    Normalize the spectrogram to mean=0, std=1 per channel
    """

    def __call__(self, spec):
        log_mel = torch.log(torch.clamp(spec, min=1e-18))
        mean = torch.mean(log_mel, dim=1, keepdim=True)
        std = torch.std(log_mel, dim=1, keepdim=True) + 1e-5
        log_mel = (log_mel - mean) / std
        return log_mel


transforms = {
    "train": transforms.Compose(
        [
            AddNormalNoise(),
            PitchShifting(),
            TimeStretch(),
            MelSpectrogram(MelSpectrogramConfig),
            NormalizePerFeature(),
            torchaudio.transforms.TimeMasking(Params.time_masking, True),
        ]
    ),
    "test": transforms.Compose(
        [
            MelSpectrogram(MelSpectrogramConfig),
            NormalizePerFeature(),
        ]
    ),
}


def collate_fn(batch):
    """
    Stacking sequences of variable lengths in batches with zero-padding in the end of sequences
    :param batch: list of tuples with (inputs with shape (time, channels), inputs_length, targets, targets_length)
    :return: tensor (batch, channels, max_length of inputs) with zero-padded inputs,
             tensor (batch, ) with inputs_lengths,
             tensor (batch, max_length of targets) with zero-padded targets,
             tensor (batch, ) with targets_lengths
    """

    inputs, inputs_length, targets, targets_length = list(zip(*batch))
    input_aligned = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)

    start_with_eos = [torch.cat([torch.Tensor([2]), target]) for target in targets]
    end_with_eos = [torch.cat([target, torch.Tensor([2])]) for target in targets]

    prev_output_tokens_aligned = torch.nn.utils.rnn.pad_sequence(
        start_with_eos, batch_first=True, padding_value=1.0
    )
    target_aligned = torch.nn.utils.rnn.pad_sequence(
        end_with_eos, batch_first=True, padding_value=1.0
    )

    sample = {}
    net_input = {}

    net_input["src_tokens"] = input_aligned
    net_input["src_lengths"] = torch.Tensor(inputs_length).long()
    net_input["prev_output_tokens"] = prev_output_tokens_aligned.long()
    sample["net_input"] = net_input

    sample["targets"] = target_aligned.long()
    sample["target_lengths"] = torch.Tensor(targets_length).long()
    sample["ntokens"] = sum(targets_length)
    sample["nsentences"] = len(inputs)

    return sample
