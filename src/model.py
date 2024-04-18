import lightning as L
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.text import WordErrorRate
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Union
from transformers.optimization import get_linear_schedule_with_warmup

from omegaconf import DictConfig

from .utils import phonetic_decode, decode
import numpy as np

from .modules.mamba import mamba_block
from .modules.conv import convolutional_block
from .modules.concat import concatenate_consecutive
from .modules.pooling import consecutive_pooling
from .modules.local_attention import local_attention_block


class B2T_stack(L.LightningModule):
    def __init__(self, layers):
        super(B2T_stack, self).__init__()
        # layers example
        # [
        #   ["mamba", 256, 2, True],  # input_channels, n_layer, bidirectional
        #   ["concat", 256, 512, 2],  # input_dims, output_dims, group_size
        #   ["pooling", "max", 2],  # pooling_type, group_size
        #   ["conv", 256, 512, 2]
        # ]

        modules_list = []

        for l in layers:
            if l[0] == "mamba":
                _, in_channels, n_layer, bidirectional = l
                modules_list.append(
                    mamba_block(
                        d_model=in_channels,
                        n_layer=n_layer,
                        bidirectional=bidirectional,
                    )
                )
            elif l[0] == "concat":
                _, in_dims, out_dims, group_size = l

                modules_list.append(
                    concatenate_consecutive(
                        input_dims=in_dims, output_dims=out_dims, group_size=group_size
                    )
                )
            elif l[0] == "pooling":
                _, pooling_type, group_size = l

                modules_list.append(
                    consecutive_pooling(
                        pooling_type=pooling_type, group_size=group_size
                    )
                )
            elif l[0] == "conv":
                if len(l) == 5:
                    _, in_dims, out_dims, stride, hidden_size = l
                else:
                    _, in_dims, out_dims, stride = l
                    hidden_size = None

                modules_list.append(
                    convolutional_block(
                        input_dims=in_dims,
                        output_dims=out_dims,
                        stride=stride,
                        hidden_size=hidden_size,
                    )
                )
            elif l[0] == "attention":
                _, in_dims, depth, local_attn_window_size, positional_embeding = l
                modules_list.append(
                    local_attention_block(
                        input_dims=in_dims,
                        depth=depth,
                        local_attn_window_size=local_attn_window_size,
                        max_seq_len=1000,
                        positional_embeding=positional_embeding,
                    )
                )
            else:
                raise ValueError(f"unknown layer: {l[0]}")

        self.layers = nn.ModuleList(modules_list)

        self.output_dims = self.layers[-1].output_dims

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class BaseModel(L.LightningModule):
    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = (
            dataset_size
            * self.trainer.max_epochs
            // (self.trainer.accumulate_grad_batches * num_devices)
        )
        return num_steps

    def configure_optimizers(self):

        if self.trainer.max_epochs == -1:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.optimizer.peak_lr
            )

            return self.optimizer

        betas = (self.config.optimizer.beta_1, self.config.optimizer.beta_2)

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimizer.peak_lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=betas,
            eps=self.config.optimizer.eps,
        )

        def get_scheduler(
            optimizer, num_training_steps, warmup_steps, peak_lr, last_lr
        ):

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return current_step / warmup_steps
                progress = (current_step - warmup_steps) / (
                    num_training_steps - warmup_steps
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = last_lr + (peak_lr - last_lr) * cosine_decay
                return lr / peak_lr

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        num_steps = self.num_steps()

        self.scheduler = get_scheduler(
            self.optimizer,
            num_steps,
            int(num_steps * self.config.optimizer.warmup_perc),
            self.config.optimizer.peak_lr,
            self.config.optimizer.last_lr,
        )

        lr_scheduler = {
            "scheduler": self.scheduler,
            "name": "custom_scheduler",
            "interval": "step",  # Ensure learning rate updates per step
            "frequency": 1,  # Optional: If you want to make sure it updates every step
        }

        return [self.optimizer], [lr_scheduler]


class B2T_CTC(BaseModel):
    def __init__(self, config: DictConfig, phoneme_rec=False):
        # phoneme_rec: recognize phoneme or character
        super(B2T_CTC, self).__init__()
        self.save_hyperparameters()

        self.phoneme_rec = phoneme_rec

        self.config = config

        self.encoder = B2T_stack(config.encoder.layers)

        if phoneme_rec:
            self.linear_ph = nn.Linear(
                self.encoder.output_dims, config.decoder.phoneme_vocab_size
            )
        else:
            self.linear_ch = nn.Linear(
                self.encoder.output_dims, config.decoder.character_vocab_size
            )
        # 0: padding
        # 1: input
        # 2: masked
        # since mamba has not support masking out padded inputs
        self.mask_tokens = nn.Embedding(
            num_embeddings=3, embedding_dim=256 #, padding_idx=1
        )

    def forward(self, spikePow, spikePow_mask):
        # _input (batch_size, input_len, input_channels)

        mask_embeddings = self.mask_tokens(spikePow_mask)

        # assume spikePow is padded and masked with 0
        spikePow = spikePow + mask_embeddings

        hidden_states = self.encoder(spikePow)

        if self.phoneme_rec:
            res = self.linear_ph(hidden_states)
        else:
            res = self.linear_ch(hidden_states)

        return res.log_softmax(-1)

    def calc_loss(self, batch):
        res = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
        )

        output_len = [len(a) for a in res]

        output_len = torch.tensor(output_len, device=self.device)

        if self.phoneme_rec:
            loss = F.ctc_loss(
                res.transpose(0, 1),
                batch["phonemize_ids"],
                output_len,
                batch["phonemize_ids_len"] - 1,  # remove the eos token
            )
        else:
            loss = F.ctc_loss(
                res.transpose(0, 1),
                batch["sent_ids"],
                output_len,
                batch["sent_ids_len"] - 1,  # remove the eos token
            )

        return loss, res

    def training_step(self, batch):
        # TODO: mask part of the input

        loss, res = self.calc_loss(batch)

        self.log("train_loss", loss.detach(), prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_pred = []
        self.val_tar = []

    def validation_step(self, batch):

        loss, res = self.calc_loss(batch)

        self.log("valid_loss", loss, batch_size=len(batch["spikePow"]), prog_bar=True)

        self.val_pred += res.argmax(dim=-1).cpu().tolist()

        if self.phoneme_rec:
            self.val_tar += batch["phonemized"]
        else:
            self.val_tar += batch["sent"]

    def on_validation_epoch_end(self):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        raw_pred = [dec(s) for s in self.val_pred]
        pred = [dec(s, True) for s in self.val_pred]

        pred = [s.replace("_", "") for s in pred]

        with open("valid.txt", "w") as txt_file:
            for i in range(len(self.val_pred)):
                txt_file.write(f"{raw_pred[i]}\n{pred[i]}\n{self.val_tar[i]}\n\n")
        self.val_pred = []
        self.val_tar = []

    def on_test_epoch_start(self):
        self.test_pred = []

    def test_step(self, batch):
        res = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
        )

        self.test_pred += res.argmax(dim=-1).cpu().tolist()

    def on_test_epoch_end(self):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        raw_pred = [dec(s) for s in self.test_pred]
        pred = [dec(s, True) for s in self.test_pred]

        pred = [s.replace("_", "") for s in pred]

        with open("test.txt", "w") as txt_file:
            for i in range(len(self.test_pred)):
                txt_file.write(f"{raw_pred[i]}\n{pred[i]}\n\n")
        self.test_pred = []


class B2T_Model(BaseModel):
    def __init__(self, config: DictConfig, phoneme_rec=False):
        # phoneme_rec: recognize phoneme or character
        super(B2T_Model, self).__init__()
        self.save_hyperparameters()

        self.phoneme_rec = phoneme_rec

        self.config = config

        self.encoder = B2T_stack(config.encoder.layers)

        output_dims = self.encoder.output_dims

        if config.decoder.get("layers"):
            self.second_enc = B2T_stack(config.decoder.layers)

            output_dims = self.second_enc.output_dims
        else:
            self.second_enc = None

        if phoneme_rec:
            self.linear_ph = nn.Linear(output_dims, config.decoder.phoneme_vocab_size)
        else:
            self.linear_ch = nn.Linear(output_dims, config.decoder.character_vocab_size)
        
        # 0: padding
        # 1: input
        # 2: masked
        # since mamba has not support masking out padded inputs
        self.mask_tokens = nn.Embedding(
            num_embeddings=3, embedding_dim=256, padding_idx=1
        )

    def forward(self, spikePow, spikePow_mask):
        # spikePow (batch_size, input_len, input_channels)
        mask_embeddings = self.mask_tokens(spikePow_mask)

        # assume spikePow is padded and masked with 0
        spikePow = spikePow + mask_embeddings

        hidden_states = self.encoder(spikePow)

        if self.second_enc is not None:
            hidden_states = self.second_enc(hidden_states)

        if self.phoneme_rec:
            logits = self.linear_ph(hidden_states)
        else:
            logits = self.linear_ch(hidden_states)

        return logits

    def calc_loss(self, batch):
        logits = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
        )

        batch_size, seq_len, vocab_size = logits.shape

        if self.phoneme_rec:
            labels = batch["phonemize_ids"]
        else:
            labels = batch["sent_ids"]

        # pad labels to the same length of logits

        # batch_size, seq_len
        padded_labels = F.pad(
            labels, (0, seq_len - labels.shape[-1], 0, 0), mode="constant", value=-100
        )

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, vocab_size), padded_labels.view(-1))

        return loss, logits

    def training_step(self, batch):
        # TODO: mask part of the input

        loss, logits = self.calc_loss(batch)

        self.log("train_loss", loss.detach(), prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_pred = []
        self.val_tar = []

    def validation_step(self, batch):

        loss, logits = self.calc_loss(batch)

        self.log("valid_loss", loss, batch_size=len(batch["spikePow"]), prog_bar=True)

        self.val_pred += logits.argmax(dim=-1).cpu().tolist()

        if self.phoneme_rec:
            self.val_tar += batch["phonemized"]
        else:
            self.val_tar += batch["sent"]

    def on_validation_epoch_end(self):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        pred = [dec(s) for s in self.val_pred]

        with open("valid.txt", "w") as txt_file:
            for i in range(len(self.val_pred)):
                txt_file.write(f"{pred[i]}\n{self.val_tar[i]}\n\n")

        self.val_pred = []
        self.val_tar = []

    def on_test_epoch_start(self):
        self.test_pred = []

    def test_step(self, batch):
        res = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
        )

        self.test_pred += res.argmax(dim=-1).cpu().tolist()

    def on_test_epoch_end(self):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        pred = [dec(s) for s in self.test_pred]

        with open("test.txt", "w") as txt_file:
            for i in range(len(self.test_pred)):
                txt_file.write(f"{pred[i]}\n")
        self.test_pred = []
