import os
import string

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torch.utils.data import Dataset

from config import Params

PUNCTUATION = string.punctuation + '—–«»−…‑'


def prepare_bpe():
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
    bpe.load(os.path.join("data", Params.bpe_model))
    return bpe


class LJDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dir = "data/LJSpeech-1.1/wavs"
        self.filenames = df.index.values
        self.labels = df[2].values
        self.transform = transform
        self.bpe = prepare_bpe()

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        text = self.labels[idx].lower().strip().translate(str.maketrans('', '', PUNCTUATION))
        text = np.array(self.bpe.encode_as_ids(text))
        wav, sr = torchaudio.load(os.path.join(self.dir, f'{filename}.wav'))
        wav = wav.squeeze()
        try:
            input = self.transform(wav)
        except:
            print(filename, self.labels[idx], self.transform, type(self.transform))
            raise Exception("pipa")
        len_input = input.shape[1]
        return input.T, len_input, torch.Tensor(text), len(text)

    def __len__(self):
        return len(self.filenames)