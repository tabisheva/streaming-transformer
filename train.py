from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.dataset import LJDataset, LibriSpeechDataset
from config import Params
from src.decoder import CerWer
from src.data_transforms import transforms, collate_fn
import wandb
import numpy as np
from all import Dictionary

from model import AugmentedMemoryConvTransformerModel

class ScheduledOpt:

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            self._rate = rate
            self.optimizer.step()

    def rate(self, step=None):
        # https://arxiv.org/pdf/1706.03762.pdf
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


torch.manual_seed(24)
if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24)
if Params.dataset != "LS":
    df = pd.read_csv("data/LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
    train, test = train_test_split(df, test_size=0.1, random_state=10)

    train_dataset = LJDataset(train, transform=transforms['train'])
    test_dataset = LJDataset(test, transform=transforms['test'])
else:
    train_dataset = LibriSpeechDataset(mode="train", transform=transforms['train'])
    test_dataset = LibriSpeechDataset(mode="test", transform=transforms['test'])

train_dataloader = DataLoader(train_dataset,
                              batch_size=Params.batch_size,
                              num_workers=Params.num_workers,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=collate_fn)

test_dataloader = DataLoader(test_dataset,
                             batch_size=Params.batch_size,
                             num_workers=Params.num_workers,
                             pin_memory=True,
                             collate_fn=collate_fn)

device = torch.device(Params.device if torch.cuda.is_available() else "cpu")

@dataclass
class TargetDictHolder:
    target_dictionary: Any
    tgt_dict: Any
if Params.dataset == "LS":
    DICT_PATH = f"/data/aotabisheva/data/libri/train-wav/vocabulary_LS_{Params.vocab_size}.txt"
else:
    DICT_PATH = f"/home/aotabisheva/streaming_transformer/data/vocabulary_LJ_{Params.vocab_size}.txt"
tgt_dict = Dictionary.load(DICT_PATH)
target_dict = TargetDictHolder(target_dictionary=tgt_dict, tgt_dict=tgt_dict)

model = AugmentedMemoryConvTransformerModel.build_model(Params, target_dict)
if Params.from_pretrained:
    model.load_state_dict(torch.load(Params.model_path, map_location=device))
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index = 1)
#optimizer = torch.optim.AdamW(model.parameters(), lr=Params.lr)
optimizer = ScheduledOpt(Params.encoder_ffn_embed_dim, 8000, 
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

num_steps = len(train_dataloader) * Params.num_epochs
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00001)
if Params.dataset == "LS":
    cerwer = CerWer("/data/aotabisheva/data/libri/train-wav")
else:
    cerwer = CerWer("/home/aotabisheva/streaming_transformer/data")

if Params.wandb_log:
    wandb.init(project=Params.wandb_name)
#    wandb.watch(model, log="all", log_freq=1000)

def to_gpu(sample, device):
    new_sample = dict()
    for k, v in sample.items():
        if isinstance(v, int):
            new_sample[k] = v
        elif k == "net_input":
            new_sample[k] = dict()
            for k1, v1 in sample[k].items():
                new_sample[k][k1] = v1.to(device)
        else:
            new_sample[k] = v.to(device)
    return new_sample

start_epoch = Params.start_epoch + 1 if Params.from_pretrained else 1
for epoch in range(start_epoch, Params.num_epochs + 1):
    train_cer, train_wer, val_wer, val_cer = 0.0, 0.0, 0.0, 0.0
    train_losses = []
    model.train()
    for idx, sample in enumerate(train_dataloader):
        sample = to_gpu(sample, device)
        outputs, _ = model(**sample["net_input"])
        outputs = outputs.permute(0, 2, 1)
        #optimizer.zero_grad()
        optimizer.optimizer.zero_grad()
        loss = criterion(outputs, sample["targets"]).cpu()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Params.clip_grad_norm)
        optimizer.step()
#        lr_scheduler.step()
        train_losses.append(loss.item())
        _, max_probs = torch.max(outputs, 1)
        train_epoch_cer, train_epoch_wer, train_decoded_words, train_target_words = cerwer(max_probs.long().cpu().numpy(),
                                                                                           sample["targets"].cpu().numpy(),
                                                                                           sample["net_input"]["src_lengths"],
                                                                                           sample["target_lengths"])
        train_wer += train_epoch_wer
        train_cer += train_epoch_cer
        if (idx + 1) % 100 == 0:
            if Params.wandb_log:
                wandb.log({"train_loss": loss.item()})

    model.eval()
    with torch.no_grad():
        val_losses = []
        for sample in test_dataloader:
            sample = to_gpu(sample, device)
            outputs, _ = model(**sample["net_input"])
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, sample["targets"]).cpu()
            val_losses.append(loss.item())
            _, max_probs = torch.max(outputs, 1)
            val_epoch_cer, val_epoch_wer, val_decoded_words, val_target_words = cerwer(max_probs.long().cpu().numpy(),
                                                                                           sample["targets"].cpu().numpy(),
                                                                                           sample["net_input"]["src_lengths"],
                                                                                           sample["target_lengths"])
            val_wer += val_epoch_wer
            val_cer += val_epoch_cer

    if Params.wandb_log:
        wandb.log({"val_wer": val_wer / len(test_dataset),
                   "train_cer": train_cer / len(train_dataset),
                   "val_loss": np.mean(val_losses),
                   "train_wer": train_wer / len(train_dataset),
                   "val_cer": val_cer / len(test_dataset),
                   "train_samples": wandb.Table(columns=["Target text", "Predicted text"],
                                                data=[[train_target_words, train_decoded_words]]),
                   "val_samples": wandb.Table(columns=["Target text", "Predicted text"],
                                              data=[[val_target_words, val_decoded_words]]),
                   })

    if (epoch % 5 == 0) and (epoch >= 10):
        torch.save(model.state_dict(), f"left{Params.left_context}_right{Params.right_context}_segment{Params.segment_size}_epoch{epoch}_dataset{Params.dataset}_vocab_size{Params.vocab_size}_linear{Params.linear_attention}.pth")
