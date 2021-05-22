import os
import string

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torch.utils.data import Dataset

from config import Params

PUNCTUATION = string.punctuation + "—–«»−…‑"


def prepare_bpe():
    """
    Loads bpe model if exists, if not - traines and saves it
    :return:
    """
    bpe = spm.SentencePieceProcessor()
    bpe.load(os.path.join(Params.dataset_path, Params.bpe_model))
    return bpe


class LJDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dir = os.path.join(Params.dataset_path, "wavs")
        self.filenames = df.index.values
        self.labels = df[2].values
        self.transform = transform
        self.bpe = prepare_bpe()

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        text = (
            self.labels[idx]
            .lower()
            .strip()
            .translate(str.maketrans("", "", PUNCTUATION))
        )
        text = np.array(self.bpe.encode_as_ids(text))
        wav, sr = torchaudio.load(os.path.join(self.dir, f"{filename}.wav"))
        wav = wav.squeeze()
        input = self.transform(wav)
        len_input = input.shape[1]
        return input.T, len_input, torch.Tensor(text), len(text)

    def __len__(self):
        return len(self.filenames)
