import os
import string

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torch.utils.data import Dataset

from config import Params

PUNCTUATION = string.punctuation + '—–«»−…‑'


def prepare_bpe(data_path):
    """
    Loads bpe model if exists, if not - traines and saves it
    :return:
    """
    # if not os.path.exists(bpe_path):
    #     df = pd.read_csv("LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
    #     train_data_path = 'bpe_texts.txt'
    #     with open(train_data_path, "w") as f:
    #         for i, row in df.iterrows():
    #             text = row[2].lower().strip().translate(str.maketrans('', '', PUNCTUATION))
    #             f.write(f"{text}\n")
    #     yttm.BPE.train(data=train_data_path, vocab_size=Params.vocab_size, model=bpe_path)
    #     os.system(f'rm {train_data_path}')
    bpe = spm.SentencePieceProcessor()
    bpe.load(os.path.join(data_path, Params.bpe_model))
    return bpe


class LJDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dir = "data/LJSpeech-1.1/wavs"
        self.filenames = df.index.values
        self.labels = df[2].values
        self.transform = transform
        self.bpe = prepare_bpe("data")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        text = self.labels[idx].lower().strip().translate(str.maketrans('', '', PUNCTUATION))
        text = np.array(self.bpe.encode_as_ids(text))
        wav, sr = torchaudio.load(os.path.join(self.dir, f'{filename}.wav'))
        wav = wav.squeeze()
        input = self.transform(wav)
        len_input = input.shape[1]
        return input.T, len_input, torch.Tensor(text), len(text)

    def __len__(self):
        return len(self.filenames)


class LibriSpeechDataset(Dataset):
    def __init__(self, mode="train", transform=None):
        if mode == "train":
            self.dir = "/data/aotabisheva/data/libri/train-wav"
        else:
            self.dir = "/data/aotabisheva/data/libri/test-wav"
        self.filenames_dir = os.path.join(self.dir, "metadata.txt")
        self.labels = []
        self.filenames = []
        with open(self.filenames_dir, "r") as f:
            for line in f.readlines():
                name, text = line.split(' ', 1)
                self.labels.append(text.strip().lower())
                self.filenames.append(name.strip())
        self.transform = transform
        self.bpe = prepare_bpe("/data/aotabisheva/data/libri/train-wav")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        text = self.labels[idx].translate(str.maketrans('', '', PUNCTUATION))
        text = np.array(self.bpe.encode_as_ids(text))
        wav, sr = torchaudio.load(os.path.join(self.dir, f'{filename}.wav'))
        wav = wav.squeeze()
        input = self.transform(wav)
        len_input = input.shape[1]
        return input.T, len_input, torch.Tensor(text), len(text)

    def __len__(self):
        return len(self.filenames)


