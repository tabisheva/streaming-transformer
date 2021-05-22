from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import profiler
from config import Params
from data.dataset import LJDataset
from src.modules.utils import Dictionary, to_gpu
from src.metrics import CerWer
from src.modules.model import AugmentedMemoryConvTransformerModel
from src.scheduler import ScheduledOpt
from src.transforms import transforms, collate_fn

torch.manual_seed(24)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(24)

df = pd.read_csv(
    f"{Params.dataset_path}/metadata.csv",
    sep="|",
    quotechar="`",
    index_col=0,
    header=None,
)
train, test = train_test_split(df, test_size=0.1, random_state=10)

train_dataset = LJDataset(train, transform=transforms["train"])
test_dataset = LJDataset(test, transform=transforms["test"])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=Params.batch_size,
    num_workers=Params.num_workers,
    shuffle=True,
    pin_memory=True,
    collate_fn=collate_fn,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=Params.batch_size,
    num_workers=Params.num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
)

device = torch.device(Params.device if torch.cuda.is_available() else "cpu")
print(f"device {device} is used")


@dataclass
class TargetDictHolder:
    target_dictionary: Any
    tgt_dict: Any


DICT_PATH = f"{Params.data_root}/vocabulary_LJ_{Params.vocab_size}.txt"

tgt_dict = Dictionary.load(DICT_PATH)
target_dict = TargetDictHolder(target_dictionary=tgt_dict, tgt_dict=tgt_dict)

model = AugmentedMemoryConvTransformerModel.build_model(Params, target_dict)
if Params.from_pretrained:
    model.load_state_dict(torch.load(Params.model_name, map_location=device))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"model has {total_params} params")

model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = ScheduledOpt(
    Params.encoder_ffn_embed_dim,
    Params.warmup_steps,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
)

num_steps = len(train_dataloader) * Params.num_epochs

cerwer = CerWer(Params.data_root)

if Params.wandb_log:
    wandb.init(project=Params.wandb_name)

start_epoch = Params.start_epoch + 1 if Params.from_pretrained else 1
best_cer = 10.0


for epoch in range(start_epoch, Params.num_epochs + 1):
    train_cer, train_wer, val_wer, val_cer = 0.0, 0.0, 0.0, 0.0
    train_losses = []
    model.train()
    with profiler.profile() as prof:
        for idx, sample in enumerate(train_dataloader):
            sample = to_gpu(sample, device)
            outputs, extra = model(**sample["net_input"])
            outputs = outputs.permute(0, 2, 1)
            optimizer.optimizer.zero_grad()
            loss = criterion(outputs, sample["targets"]).cpu()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Params.clip_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())
            _, max_probs = torch.max(outputs, 1)
            (
                train_epoch_cer,
                train_epoch_wer,
                train_decoded_words,
                train_target_words,
            ) = cerwer(
                max_probs.long().cpu().numpy(),
                sample["targets"].cpu().numpy(),
                sample["net_input"]["src_lengths"],
                sample["target_lengths"],
            )
            train_wer += train_epoch_wer
            train_cer += train_epoch_cer

            if (idx + 1) % 100 == 0:
                if Params.wandb_log:
                    wandb.log({"train_loss": loss.item()})
                    if Params.linear_attention:
                        wandb.log({"training_linear": prof.get_avg("MultiLinearAttn_forward")})
                    else:
                        wandb.log({"training_softmax": prof.get_avg("MultiAttn_forward")})

    model.eval()
    with torch.no_grad():
        val_losses = []
        with profiler.profile() as prof:
            for idx, sample in enumerate(test_dataloader):
                sample = to_gpu(sample, device)
                outputs, extra = model(**sample["net_input"])
                outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, sample["targets"]).cpu()
                val_losses.append(loss.item())
                _, max_probs = torch.max(outputs, 1)
                (
                    val_epoch_cer,
                    val_epoch_wer,
                    val_decoded_words,
                    val_target_words,
                ) = cerwer(
                    max_probs.long().cpu().numpy(),
                    sample["targets"].cpu().numpy(),
                    sample["net_input"]["src_lengths"],
                    sample["target_lengths"],
                )
                val_wer += val_epoch_wer
                val_cer += val_epoch_cer

    if Params.wandb_log:
        wandb.log(
            {
                "val_wer": val_wer / len(test_dataset),
                "train_cer": train_cer / len(train_dataset),
                "val_loss": np.mean(val_losses),
                "train_wer": train_wer / len(train_dataset),
                "val_cer": val_cer / len(test_dataset),
                "testing_linear": prof.get_avg("MultiLinearAttn_forward"),
                "testing_softmax": prof.get_avg("MultiAttn_forward"),
                "train_samples": wandb.Table(
                    columns=["Target text", "Predicted text"],
                    data=[[train_target_words, train_decoded_words]],
                ),
                "val_samples": wandb.Table(
                    columns=["Target text", "Predicted text"],
                    data=[[val_target_words, val_decoded_words]],
                ),
            }
        )
    if val_cer / len(test_dataset) < best_cer:
        best_cer = val_cer / len(test_dataset)
        model_name = f"left{Params.left_context}_" \
                     f"right{Params.right_context}_segment{Params.segment_size}_epoch{epoch}_vocab_size{Params.vocab_size}_linear{Params.linear_attention}.pth"
        torch.save(model.state_dict(), model_name)
        print("we have one more checkpoint!")
