import lightning as L
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.optimization import get_linear_schedule_with_warmup

# from torchmetrics.text import WordErrorRate
import torchmetrics
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Union
from transformers.optimization import get_linear_schedule_with_warmup

from omegaconf import DictConfig

from .utils import phonetic_decode, decode, vocab, phoneme_vocab
import numpy as np
import os

# from .modules.mamba_cu_seqlens import mamba_block

from .modules.mamba import mamba_block
from .modules.highway import Highway
from .modules.others import Pack, UnPack
from .modules.conv import conv_block
from .modules.resnet import resnet_block
import itertools


class modules_stack(L.LightningModule):
    def __init__(self, layers):
        super(modules_stack, self).__init__()
        # layers example
        # [
        #   ["mamba", 256, 2, True, 0.7],  # input_channels, n_layer, bidirectional, update_probs
        #   ["concat", 256, 512, 2],  # input_dims, output_dims, group_size
        #   ["pooling", "max", 2],  # pooling_type, group_size
        #   ["conv", 256, 512, 2]
        # ]

        modules_list = []

        self.output_dims = 0

        for l in layers:
            if l[0] == "mamba":
                _, in_channels, n_layer, bidirectional, update_probs = l
                modules_list.append(
                    mamba_block(
                        d_model=in_channels,
                        n_layer=n_layer,
                        bidirectional=bidirectional,
                        update_probs=update_probs,
                    )
                )
                self.output_dims = in_channels
            elif l[0] == "pooling":
                _, pooling_type, kernel_size, stride = l

                modules_list.append(
                    consecutive_pooling(
                        pooling_type=pooling_type,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                )
            elif l[0] == "resnet":
                if len(l) == 5:
                    _, in_dims, out_dims, stride, hidden_size = l
                else:
                    _, in_dims, out_dims, stride = l
                    hidden_size = None

                modules_list.append(
                    resnet_block(
                        input_dims=in_dims,
                        output_dims=out_dims,
                        stride=stride,
                        hidden_size=hidden_size,
                    )
                )
                self.output_dims = out_dims

            elif l[0] == "highway":
                _, in_channels, n_layer = l
                modules_list.append(Highway(input_dim=in_channels, num_layers=n_layer))
                self.output_dims = in_channels

            elif l[0] == "conv":
                _, input_dims, output_dims, kernel_size, stride, groups = l
                modules_list.append(
                    conv_block(input_dims, output_dims, kernel_size, stride, groups)
                )
                self.output_dims = output_dims

            elif l[0] == "pack":
                modules_list.append(Pack())
            elif l[0] == "unpack":
                modules_list.append(UnPack())
            else:
                raise ValueError(f"unknown layer: {l[0]}")

        self.layers = nn.ModuleList(modules_list)

        if len(self.layers) == 0:
            return

        assert self.output_dims != 0

    def forward(self, hidden_states, spikePow_lens):
        for layer in self.layers:
            hidden_states, spikePow_lens = layer(hidden_states, spikePow_lens)
        return hidden_states, spikePow_lens


class CTC_decoder(L.LightningModule):
    def __init__(self, input_dims=512, n_layer=5, phoneme_rec=False, update_probs=0.7):
        super(CTC_decoder, self).__init__()
        self.phoneme_rec = phoneme_rec
        self.input_dims = input_dims

        self.update_probs = update_probs

        self.layers = [["mamba", input_dims, n_layer, True, update_probs]]

        self.encoder = modules_stack(self.layers)

        if phoneme_rec:
            # remove the eos token
            self.vocab_size = len(phoneme_vocab) - 1
        else:
            self.vocab_size = len(vocab) - 1

        self.linear = nn.Linear(input_dims, self.vocab_size)

    def forward(self, hidden_states, input_lens):

        hidden_states, output_lens = self.encoder(hidden_states, input_lens)

        lm_logits = self.linear(hidden_states)

        return {
            "logits": lm_logits.log_softmax(-1),
            "output_lens": output_lens,
            "hidden_states": hidden_states,
        }

    def disable_grad(self):
        self.orig_requires_grads = [p.requires_grad for p in self.parameters()]

        for p in self.parameters():
            p.requires_grad = False

    def enable_grad(self):
        for p, rg in zip(self.parameters(), self.orig_requires_grads):
            p.requires_grad = rg

    def calc_loss(self, hidden_states, input_lens, batch):
        output = self(hidden_states=hidden_states, input_lens=input_lens)

        if self.phoneme_rec:
            labels = batch["phonemize_ids"]
            label_lens = batch["phonemize_ids_len"]
        else:
            labels = batch["sent_ids"]
            label_lens = batch["sent_ids_len"]

        # remove the eos token
        label_lens -= 1

        loss = F.ctc_loss(
            output["logits"].transpose(0, 1),
            labels,
            output["output_lens"],
            label_lens,
            zero_infinity=True,
        )

        return {
            "loss": loss,
            "logits": output["logits"],
            "output_lens": output["output_lens"],
        }

    def batch_decode(self, ids, output_lens=None, raw_ouput=False):

        if output_lens is not None and not raw_ouput:
            temp = []
            for idx, s in enumerate(ids):
                temp.append(s[: output_lens[idx]])
            ids = temp

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        texts = [dec(s, not raw_ouput) for s in ids]

        if raw_ouput:
            return texts

        texts = [s.replace("|", " ").replace("-", "").replace("_", "") for s in texts]

        return texts

    def get_target_text(self, batch):

        if self.phoneme_rec:
            label = batch["phonemized"]
        else:
            label = batch["sent"]

        label = [s.replace("|", " ").replace("-", "").replace("+", "") for s in label]

        return label


class feature_extractor(L.LightningModule):
    def __init__(
        self,
        output_dims=512,
        conv_size=1024,
        conv_kernel1=7,
        conv_kernel2=3,
        conv_g1=256,
        conv_g2=1,
    ):
        super(feature_extractor, self).__init__()

        self.output_dims = output_dims

        self.layers = [
            ["unpack"],
            ["conv", 256, conv_size, conv_kernel1, 2, conv_g1],
            ["highway", conv_size, 2],
            ["conv", conv_size, output_dims, conv_kernel2, 2, conv_g2],
            ["highway", output_dims, 2],
        ]

        self.extractor = modules_stack(self.layers)

    def forward(self, spikePow, spikePow_lens):
        hidden_states, output_lens = self.extractor(spikePow, spikePow_lens)

        return hidden_states, output_lens


class B2T_Model(L.LightningModule):
    def __init__(
        self,
        conv_size=1024,
        conv_kernel1=7,
        conv_kernel2=3,
        conv_g1=256,
        conv_g2=1,
        hidden_size=512,
        encoder_n_layer=5,
        decoder_n_layer=5,
        decoders=[
            "al",
            "ph",
        ],  # use the alphabet or phonemes, should be both at training
        update_probs=None,
        al_loss_weight=0.5,
        peak_lr=1e-4,
        last_lr=1e-6,
        beta_1=0.9,
        beta_2=0.95,
        weight_decay=0.1,
        eps=1e-08,
        lr_warmup_perc=0.1,  # lr warmup for the first 10% of the training
        **other_args,
    ):
        super(B2T_Model, self).__init__()
        # self.save_hyperparameters()

        self.peak_lr = peak_lr
        self.last_lr = last_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.eps = eps
        self.lr_warmup_perc = lr_warmup_perc

        self.update_probs = update_probs

        self.feature_extractor = feature_extractor(
            output_dims=hidden_size,
            conv_size=conv_size,
            conv_kernel1=conv_kernel1,
            conv_kernel2=conv_kernel2,
            conv_g1=conv_g1,
            conv_g2=conv_g2,
        )

        self.encoder = modules_stack(
            [["mamba", hidden_size, encoder_n_layer, True, update_probs]]
        )

        dec = {}
        loss_weights = {}

        for d in decoders:
            dec[d] = CTC_decoder(
                input_dims=hidden_size,
                n_layer=decoder_n_layer,
                phoneme_rec=d == "ph",
                update_probs=update_probs,
            )
            loss_weights[d] = al_loss_weight if d == "al" else 1 - al_loss_weight

        self.decoders = nn.ModuleDict(dec)

        self.loss_weights = loss_weights

    def forward(self, spikePow, spikePow_mask, spikePow_lens, encoder_only=False):

        hidden_states, output_lens = self.feature_extractor(spikePow, spikePow_lens)

        hidden_states, output_lens = self.encoder(hidden_states, output_lens)

        if encoder_only:
            return hidden_states, output_lens

        for k, d in self.decoders.items():
            outputs[k] = d(hidden_states, output_lens)

        return outputs

    def calc_loss(self, batch):

        hidden_states, output_lens = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
            spikePow_lens=batch["spikePow_lens"],
            encoder_only=True,
        )

        items = {}

        losses = []
        logits = []
        out_lens = []

        for k, d in self.decoders.items():
            # loss, logits, output_lens
            items[k] = d.calc_loss(hidden_states, output_lens, batch)

        return items

    def training_step(self, batch):
        items = self.calc_loss(batch)

        loss = 0
        for k, output in items.items():
            loss += output["loss"] * self.loss_weights[k]

            self.log(
                f"train_{k}_loss",
                output["loss"].detach(),
                batch_size=len(batch["spikePow"]),
            )

        self.log("train_loss", loss.detach(), prog_bar=True)

        # if self.use_addtional_corpus:

        #     add_loss = self.calc_additional_corpus_loss(batch)

        #     loss += add_loss * 0.1

        if torch.isnan(loss):
            raise Exception(f"Loss is NaN")

        return loss

    def on_validation_epoch_start(self):
        logdir = self.trainer.log_dir
        for k in self.decoders.keys():
            # erase the file
            open(os.path.join(logdir, f"valid_{k}.txt"), "w").close()

    def validation_step(self, batch):
        logdir = self.trainer.log_dir

        items = self.calc_loss(batch)

        loss = 0
        for k, output in items.items():
            loss += output["loss"] * self.loss_weights[k]

            self.log(
                f"val_{k}_loss",
                output["loss"].detach(),
                batch_size=len(batch["spikePow"]),
            )

        self.log("valid_loss", loss, batch_size=len(batch["spikePow"]), prog_bar=True)

        for k, output in items.items():

            output_lens = output["output_lens"].cpu().tolist()

            # greedy decode
            ids = output["logits"].argmax(dim=-1).cpu().tolist()

            text = []
            raw_text = []

            text = self.decoders[k].batch_decode(ids, output_lens=output_lens)

            raw_text = self.decoders[k].batch_decode(
                ids, output_lens=output_lens, raw_ouput=True
            )

            target = self.decoders[k].get_target_text(batch)

            with open(os.path.join(logdir, f"valid_{k}.txt"), "a") as txt_file:
                for i in range(len(text)):
                    txt_file.write(f"{raw_text[i]}\n{text[i]}\n{target[i]}\n\n")

    def on_validation_epoch_end(self):
        logdir = self.trainer.log_dir

        total_wer = 0

        for k in self.decoders.keys():
            preds = []
            target = []
            with open(os.path.join(logdir, f"valid_{k}.txt"), "r") as fp:
                # 0 raw_text
                # 1 text
                # 2 target
                # 3 newline
                for idx, l in enumerate(fp):
                    if idx % 4 == 1:
                        preds.append(l)
                    if idx % 4 == 2:
                        target.append(l)

            wer = torchmetrics.functional.text.word_error_rate(preds, target)

            total_wer += wer

            self.log(f"wer_{k}", wer)

        total_wer = total_wer / len(self.decoders)

        self.log("wer", total_wer, prog_bar=True)


        valid_loss = self.trainer.callback_metrics['valid_loss']
        
        score = total_wer + valid_loss/6

        self.log("score", score, prog_bar=True)


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
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr)

            return self.optimizer

        betas = (self.beta_1, self.beta_2)

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.peak_lr,
            weight_decay=self.weight_decay,
            betas=betas,
            eps=self.eps,
        )

        # num_steps = self.num_steps()
        # self.lr_scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=int(num_steps * self.config.optimizer.warmup_perc),
        #     num_training_steps=num_steps,
        # )
        # return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

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
            int(num_steps * self.lr_warmup_perc),
            self.peak_lr,
            self.last_lr,
        )

        lr_scheduler = {
            "scheduler": self.scheduler,
            "name": "custom_scheduler",
            "interval": "step",  # Ensure learning rate updates per step
            "frequency": 1,  # Optional: If you want to make sure it updates every step
        }

        return [self.optimizer], [lr_scheduler]
