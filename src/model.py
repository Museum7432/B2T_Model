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


from .modules.mamba_cu_seqlens import mamba_block, mamba_block_for_input_ids

# from .modules.mamba import mamba_block
from .modules.lstm import lstm_block
from .modules.highway import Highway
from .modules.conv import conv_block
from .modules.resnet import resnet_block
from .modules.pooling import consecutive_pooling
from .modules.local_attention import local_attention_block
from .modules.t5_encoder import t5_encoder
import itertools


class modules_stack(L.LightningModule):
    def __init__(self, layers):
        super(modules_stack, self).__init__()
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
            elif l[0] == "attention":
                _, in_dims, depth, local_attn_window_size, positional_embeding = l
                modules_list.append(
                    local_attention_block(
                        input_dims=in_dims,
                        depth=depth,
                        local_attn_window_size=local_attn_window_size,
                        positional_embeding=positional_embeding,
                    )
                )
            elif l[0] == "highway":
                _, in_channels, n_layer = l
                modules_list.append(Highway(input_dim=in_channels, num_layers=n_layer))

            elif l[0] == "conv":
                _, input_dims, output_dims, kernel_size, stride, groups = l
                modules_list.append(
                    conv_block(input_dims, output_dims, kernel_size, stride, groups)
                )
            elif l[0] == "t5":
                _, input_dims, n_layer = l
                modules_list.append(t5_encoder(input_dims, n_layer))

            elif l[0] == "lstm":
                _, input_size, num_layers, bidirectional = l
                modules_list.append(lstm_block(input_size, num_layers, bidirectional))
            else:
                raise ValueError(f"unknown layer: {l[0]}")

        self.layers = nn.ModuleList(modules_list)

        if len(self.layers) == 0:
            return

        self.output_dims = self.layers[-1].output_dims

    def forward(self, hidden_states, spikePow_lens):
        for layer in self.layers:
            hidden_states, spikePow_lens = layer(hidden_states, spikePow_lens)
        return hidden_states, spikePow_lens


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
                self.parameters(), lr=self.config.optimizer.peak_lr
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


        num_steps = self.num_steps()
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_steps * self.config.optimizer.warmup_perc),
            num_training_steps=num_steps,
        )
        return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]


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


class CTC_decoder(L.LightningModule):
    def __init__(self, layers_config, phoneme_rec=False):
        super(CTC_decoder, self).__init__()
        self.phoneme_rec = phoneme_rec
        self.layers_config = layers_config

        self.layers = modules_stack(layers_config)

        if phoneme_rec:
            # remove the eos token
            self.vocab_size = len(phoneme_vocab) - 1
        else:
            self.vocab_size = len(vocab) - 1

        self.linear = nn.Linear(self.layers.output_dims, self.vocab_size)

    def forward(self, hidden_states, input_lens):

        hidden_states, output_lens = self.layers(hidden_states, input_lens)

        hidden_states = self.linear(hidden_states)

        return hidden_states.log_softmax(-1), output_lens

    def disable_grad(self):
        self.orig_requires_grads = [p.requires_grad for p in self.parameters()]

        for p in self.parameters():
            p.requires_grad = False

    def enable_grad(self):
        for p, rg in zip(self.parameters(), self.orig_requires_grads):
            p.requires_grad = rg

    def calc_loss(self, hidden_states, input_lens, batch):
        logits, output_lens = self(hidden_states=hidden_states, input_lens=input_lens)

        if self.phoneme_rec:
            labels = batch["phonemize_ids"]
            label_lens = batch["phonemize_ids_len"]
        else:
            labels = batch["sent_ids"]
            label_lens = batch["sent_ids_len"]

        # remove the eos token
        label_lens -= 1

        loss = F.ctc_loss(
            logits.transpose(0, 1),
            labels,
            output_lens,
            label_lens,
            zero_infinity=True,
        )

        return loss, logits, output_lens

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


from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5ForConditionalGeneration,
    T5PreTrainedModel,
)


def get_t5_config(input_dims, num_decoder_layers, vocab_size, eos_token_id):
    base_config = T5Config.from_pretrained("google-t5/t5-base")
    base_config.d_model = input_dims
    base_config.num_layers = 0
    base_config.num_decoder_layers = num_decoder_layers
    base_config.vocab_size = vocab_size

    base_config.decoder_start_token_id = 0

    base_config.pad_token_id = 1

    base_config.eos_token_id = eos_token_id

    return base_config


class decoder(L.LightningModule):
    def __init__(self, layers_config, phoneme_rec=False):
        super(decoder, self).__init__()
        self.phoneme_rec = phoneme_rec

        # the last layer is the decoder
        self.layers_config = layers_config[:-1]

        if phoneme_rec:
            self.vocab_size = len(phoneme_vocab)
        else:
            self.vocab_size = len(vocab)

        self.eos_token_id = self.vocab_size - 1

        self.layers = modules_stack(self.layers_config)

        _, input_size, num_layers = layers_config[-1]

        t5_config = get_t5_config(
            input_dims=input_size,
            num_decoder_layers=num_layers,
            vocab_size=self.vocab_size,
            eos_token_id=self.eos_token_id,
        )
        self.t5_model = T5ForConditionalGeneration(t5_config)

    def disable_grad(self):
        self.orig_requires_grads = [p.requires_grad for p in self.parameters()]

        for p in self.parameters():
            p.requires_grad = False

    def enable_grad(self):
        for p, rg in zip(self.parameters(), self.orig_requires_grads):
            p.requires_grad = rg

    def forward(self, hidden_states, input_lens, labels=None):
        hidden_states, output_lens = self.layers(hidden_states, input_lens)

        batch_size, seq_len, hidden_size = hidden_states.shape

        attention_mask = torch.arange(
            seq_len, device=self.device
        ) < output_lens.unsqueeze(1)

        if labels is not None:
            return self.t5_model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                labels=labels,
            )

        logits = self.t5_model.generate(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            max_new_tokens=150,
            output_logits=True,
            return_dict_in_generate=True,
        ).logits

        batch_size, vocab_size = logits[0].shape

        logits = torch.vstack(logits)

        logits = logits.reshape(-1, batch_size, vocab_size).transpose(0, 1)

        return logits, None

    def calc_loss(self, hidden_states, input_lens, batch):
        if self.phoneme_rec:
            labels = batch["phonemize_ids"]
        else:
            labels = batch["sent_ids"]

        out = self(hidden_states=hidden_states, input_lens=input_lens, labels=labels)
        loss = out.loss
        logits = out.logits

        batch_size, seq_len, vocab_size = logits.shape

        return loss, logits, None

    def batch_decode(self, ids, output_lens=None, raw_ouput=False):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        texts = [dec(s) for s in ids]

        if raw_ouput:
            return texts

        for i in range(len(texts)):
            if "_" in texts[i]:
                texts[i] = texts[: texts.index("_")]

        texts = [s.replace("|", " ").replace("-", "").replace("_", "") for s in texts]

        return texts

    def get_target_text(self, batch):

        if self.phoneme_rec:
            label = batch["phonemized"]
        else:
            label = batch["sent"]

        label = [s.replace("|", " ").replace("-", "").replace("+", "") for s in label]

        return label


def get_decoder(decoder_type, layers_config):
    if "phonetic" in decoder_type:
        phoneme_rec = True
    elif "alphabet" in decoder_type:
        phoneme_rec = False
    else:
        raise ValueError(f"unknown decoder type: {decoder_type}")

    if "ctc" in decoder_type:
        use_ctc = True
    elif "decoder" in decoder_type:
        use_ctc = False
    else:
        raise ValueError(f"unknown decoder type: {decoder_type}")

    if use_ctc:
        dec = CTC_decoder(layers_config, phoneme_rec=phoneme_rec)
    else:
        dec = decoder(layers_config, phoneme_rec=phoneme_rec)

    return dec


class joint_Model(BaseModel):
    def __init__(self, config: DictConfig, decoders_conf=None, use_people_speech=None):
        # decoder_conf: if not none, only load those layers
        super(joint_Model, self).__init__()
        self.save_hyperparameters()

        self.config = config

        if use_people_speech is None:
            if config.get("use_people_speech"):
                self.use_people_speech = config.use_people_speech
            else:
                self.use_people_speech = False
        else:
            self.use_people_speech = use_people_speech

        self.word_level = config.get("word_level")

        self.encoder = modules_stack(config.encoder.layers)

        dec_conf = config.decoders_conf

        if decoders_conf is not None:
            dec_conf = [i for i in dec_conf if i[0] in decoders_conf]

        assert len(dec_conf) > 0

        decoder_loss_weights = []
        modules = {}

        for dec_name, dec_type, loss_weights in dec_conf:
            decoder_loss_weights.append(loss_weights)

            decoder_layers_config = config.decoders_layers[dec_name]

            modules[dec_name] = get_decoder(dec_type, decoder_layers_config)

        # normalize loss_weights
        decoder_loss_weights = [
            i / sum(decoder_loss_weights) for i in decoder_loss_weights
        ]

        self.decoder_loss_weights = decoder_loss_weights

        self.decoders = nn.ModuleDict(modules)

        if self.use_people_speech:
            self.people_speech_encoder = mamba_block_for_input_ids(
                d_model=self.encoder.output_dims,
                n_layer=4,
                bidirectional=True,
                vocab_size=len(phoneme_vocab) - 1,
            )

        if self.word_level:
            # 0: input and padding
            # 1: mask
            self.mask_tokens = nn.Embedding(
                num_embeddings=2, embedding_dim=256, padding_idx=0
            )

    def forward(self, spikePow, spikePow_mask, spikePow_lens, encoder_only=False):
        # _input (batch_size, input_len, input_channels)
        if self.word_level:
            mask_embeddings = self.mask_tokens(spikePow_mask)
            spikePow[spikePow_mask != 0, :] = 0
            spikePow = spikePow + mask_embeddings

        hidden_states, output_lens = self.encoder(spikePow, spikePow_lens)

        if encoder_only:
            return hidden_states, output_lens

        outputs = {}
        for k, d in self.decoders.items():
            # (logits, output_lens)
            outputs[k] = d(hidden_states, output_lens)
        return outputs

    def calc_loss(self, batch):

        hidden_states, output_lens = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
            spikePow_lens=batch["spikePow_lens"],
            encoder_only=True,
        )

        losses = []
        logits = []
        out_lens = []

        for k, d in self.decoders.items():
            l, lg, lens = d.calc_loss(hidden_states, output_lens, batch)

            losses.append(l)
            out_lens.append(lens)
            logits.append(lg.detach())

        return losses, logits, out_lens

    def training_step(self, batch):
        losses, logits, output_lens = self.calc_loss(batch)

        loss = 0
        for l, w in zip(losses, self.decoder_loss_weights):
            loss += l * w

        self.log("train_loss", loss.detach(), prog_bar=True)

        if self.use_people_speech:
            ps_hidden_states, ps_output_lens = self.people_speech_encoder(
                input_ids=batch["ps_input_ids"], input_lens=batch["ps_input_ids_lens"]
            )

            for i, (k, d) in enumerate(self.decoders.items()):
                # the label of people speech is text not phonemized text
                if d.phoneme_rec:
                    continue

                disable_grad = np.random.rand() < 0.5

                if disable_grad:
                    d.disable_grad()

                ps_loss, _, _ = d.calc_loss(
                    ps_hidden_states,
                    ps_output_lens,
                    {
                        "sent_ids": batch["ps_label"],
                        "sent_ids_len": batch["ps_label_lens"],
                    },
                )

                if disable_grad:
                    d.enable_grad()

                # losses[i] = losses[i] + ps_loss * 0.5
                loss += ps_loss * 0.1

                self.log(
                    f"ps_{k}_loss", ps_loss.detach(), batch_size=len(batch["spikePow"])
                )

        for k, l in zip(self.decoders.keys(), losses):
            self.log(f"train_{k}_loss", l.detach(), batch_size=len(batch["spikePow"]))

        return loss

    def on_validation_epoch_start(self):
        for k in self.decoders.keys():
            # erase the file
            open(f"valid_{k}.txt", "w").close()

    def validation_step(self, batch):
        losses, logits, output_lens = self.calc_loss(batch)

        for k, l in zip(self.decoders.keys(), losses):
            self.log(f"val_{k}_loss", l, batch_size=len(batch["spikePow"]))

        loss = 0
        for l, w in zip(losses, self.decoder_loss_weights):
            loss += l * w

        self.log("valid_loss", loss, batch_size=len(batch["spikePow"]), prog_bar=True)

        for k, l, ol in zip(self.decoders.keys(), logits, output_lens):
            if ol is not None:
                ol = ol.cpu().tolist()
            # greedy decode
            ids = l.argmax(dim=-1).cpu().tolist()

            text = []
            raw_text = []

            text = self.decoders[k].batch_decode(ids, output_lens=ol)

            raw_text = self.decoders[k].batch_decode(
                ids, output_lens=ol, raw_ouput=True
            )

            target = self.decoders[k].get_target_text(batch)

            with open(f"valid_{k}.txt", "a") as txt_file:
                for i in range(len(text)):
                    txt_file.write(f"{raw_text[i]}\n{text[i]}\n{target[i]}\n\n")

    def on_validation_epoch_end(self):
        total_wer = 0

        for k in self.decoders.keys():
            preds = []
            target = []
            with open(f"valid_{k}.txt", "r") as fp:
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

    def on_test_epoch_start(self):
        for k in self.decoders.keys():
            # erase the file
            open(f"test_{k}.txt", "w").close()
            open(f"test_raw_{k}.txt", "w").close()

    def test_step(self, batch):
        outputs = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
            spikePow_lens=batch["spikePow_lens"],
        )

        for k in self.decoders.keys():

            logits, output_lens = outputs[k]

            ids = logits.argmax(dim=-1).cpu().tolist()

            text = []
            raw_text = []

            text = self.decoders[k].batch_decode(ids, output_lens=output_lens)

            raw_text = self.decoders[k].batch_decode(
                ids, output_lens=output_lens, raw_ouput=True
            )

            with open(f"test_{k}.txt", "a") as txt_file:
                for i in range(len(text)):
                    txt_file.write(f"{text[i]}\n")

            with open(f"test_raw_{k}.txt", "a") as txt_file:
                for i in range(len(raw_text)):
                    txt_file.write(f"{raw_text[i]}\n")
