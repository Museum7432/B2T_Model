import lightning as L
import math
import torch
from torch import nn
from torch.nn import functional as F

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

from .modules.mamba import mamba_block
from .modules.t5_conv_cross_att import t5_conv_cross_att
from .modules.lstm import lstm_block
from .modules.highway import Highway
from .modules.conv import conv_block
from .modules.resnet import resnet_block
from .modules.pooling import consecutive_pooling
from .modules.local_attention import local_attention_block
from .modules.t5_encoder import t5_encoder


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

            elif l[0] == "t5_conv_cross_att":
                _, input_dims, n_layer = l
                modules_list.append(t5_conv_cross_att(input_dims, n_layer))

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

    def forward(self, hidden_states, input_lens, labels=None):
        hidden_states, output_lens = self.layers(hidden_states, input_lens)

        # TODO: create attention mask from output_lens

        if labels is not None:
            return self.t5_model(inputs_embeds=hidden_states, labels=labels)

        logits = self.t5_model.generate(
            inputs_embeds=hidden_states,
            max_new_tokens=150,
            output_logits=True,
            return_dict_in_generate=True,
        ).logits


        batch_size, vocab_size = logits[0].shape

        logits = torch.vstack(logits)
        
        logits = logits.reshape(-1,batch_size,vocab_size).transpose(0,1)

        return logits, None

    def calc_loss(self, hidden_states, input_lens, batch):
        if self.phoneme_rec:
            labels = batch["phonemize_ids"]
        else:
            labels = batch["sent_ids"]

        out = self(
            hidden_states=hidden_states, input_lens=input_lens, labels=labels
        )
        loss = out.loss
        logits = out.logits

        batch_size, seq_len, vocab_size = logits.shape

        return loss, logits, None

    def batch_decode(self, ids,output_lens=None, raw_ouput=False):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        texts = [dec(s) for s in ids]

        if raw_ouput:
            return texts

        texts = [s.replace("|", " ").replace("-", "").replace("_", "") for s in texts]

        return texts


class B2T_CTC(BaseModel):
    def __init__(self, config: DictConfig, phoneme_rec=False):
        # phoneme_rec: recognize phoneme or character
        super(B2T_CTC, self).__init__()
        self.save_hyperparameters()

        self.phoneme_rec = phoneme_rec

        self.config = config

        self.encoder = modules_stack(config.encoder.layers)

        if phoneme_rec:
            # remove the eos token
            self.vocab_size = len(phoneme_vocab) - 1

            self.linear_ph = nn.Linear(self.encoder.output_dims, self.vocab_size)
        else:
            self.vocab_size = len(vocab) - 1

            self.linear_ch = nn.Linear(self.encoder.output_dims, self.vocab_size)
        # 0: padding
        # 1: input
        # 2: masked
        # 3: eos token
        # since mamba has not support masking out padded inputs
        self.mask_tokens = nn.Embedding(
            num_embeddings=4, embedding_dim=256, padding_idx=1
        )

    def forward(self, spikePow, spikePow_mask, spikePow_lens):
        # _input (batch_size, input_len, input_channels)

        mask_embeddings = self.mask_tokens(spikePow_mask)

        # assume spikePow is padded and masked with 0
        spikePow = spikePow + mask_embeddings

        hidden_states, output_lens = self.encoder(spikePow, spikePow_lens)

        if self.phoneme_rec:
            res = self.linear_ph(hidden_states)
        else:
            res = self.linear_ch(hidden_states)

        return res.log_softmax(-1), output_lens

    def calc_loss(self, batch):
        res, output_lens = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
            spikePow_lens=batch["spikePow_lens"],
        )

        if self.phoneme_rec:
            loss = F.ctc_loss(
                res.transpose(0, 1),
                batch["phonemize_ids"],
                output_lens,
                batch["phonemize_ids_len"] - 1,  # remove the eos token
                zero_infinity=True,
            )
        else:
            loss = F.ctc_loss(
                res.transpose(0, 1),
                batch["sent_ids"],
                output_lens,
                batch["sent_ids_len"] - 1,  # remove the eos token
                zero_infinity=True,
            )

        return loss, res, output_lens

    def training_step(self, batch):
        loss, res, output_lens = self.calc_loss(batch)

        self.log("train_loss", loss.detach(), prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_pred = []
        self.val_tar = []

    def validation_step(self, batch):

        loss, res, output_lens = self.calc_loss(batch)

        self.log("valid_loss", loss, batch_size=len(batch["spikePow"]), prog_bar=True)

        preds = res.argmax(dim=-1).cpu().tolist()

        for idx, s in enumerate(preds):
            preds[idx] = s[: output_lens[idx] - 1]

        self.val_pred += preds

        if self.phoneme_rec:
            self.val_tar += batch["phonemized"]
        else:
            self.val_tar += batch["sent"]

    def on_validation_epoch_end(self):

        self.val_tar = [
            s.replace("|", " ").replace("_", "").replace("+", "").replace("-", "")
            for s in self.val_tar
        ]

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        raw_pred = [dec(s) for s in self.val_pred]

        pred = [dec(s, True).replace("|", " ").replace("-", "") for s in self.val_pred]

        wer = torchmetrics.functional.text.word_error_rate(pred, self.val_tar)

        self.log("wer", wer, prog_bar=True)

        with open("valid.txt", "w") as txt_file:
            for i in range(len(self.val_pred)):
                txt_file.write(f"{raw_pred[i]}\n{pred[i]}\n{self.val_tar[i]}\n\n")
        self.val_pred = []
        self.val_tar = []

    def on_test_epoch_start(self):
        self.test_pred = []

    def test_step(self, batch):
        res, output_lens = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
            spikePow_lens=batch["spikePow_lens"],
        )
        preds = res.argmax(dim=-1).cpu().tolist()
        for idx, s in enumerate(preds):
            preds[idx] = s[: output_lens[idx] - 1]

        self.test_pred += preds

    def on_test_epoch_end(self):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        raw_pred = [dec(s) for s in self.test_pred]
        pred = [dec(s, True) for s in self.test_pred]

        pred = [s.replace("-", "").replace("|", " ") for s in pred]

        with open("test.txt", "w") as txt_file:
            for i in range(len(self.test_pred)):
                txt_file.write(f"{pred[i]}\n")

        with open("test_raw.txt", "w") as txt_file:
            for i in range(len(self.test_pred)):
                txt_file.write(f"{raw_pred[i]}\n")
        self.test_pred = []


# from transformers.models.t5.configuration_t5 import T5Config
# from transformers.models.t5.modeling_t5 import (
#     T5Stack,
#     T5ForConditionalGeneration,
#     T5PreTrainedModel,
# )


# def get_t5_config(input_dims, num_layers, num_decoder_layers, vocab_size, eos_token_id):
#     base_config = T5Config.from_pretrained("google-t5/t5-base")
#     base_config.d_model = input_dims
#     base_config.num_layers = num_layers
#     base_config.num_decoder_layers = num_decoder_layers
#     base_config.vocab_size = vocab_size

#     base_config.pad_token_id = pad_token_id

#     base_config.decoder_start_token_id = 0

#     base_config.pad_token_id = 0

#     base_config.eos_token_id = eos_token_id

#     return base_config


class B2T_Model(BaseModel):
    def __init__(self, config: DictConfig, phoneme_rec=False):
        # phoneme_rec: recognize phoneme or character
        super(B2T_Model, self).__init__()
        self.save_hyperparameters()

        self.phoneme_rec = phoneme_rec

        self.config = config

        self.encoder = modules_stack(config.encoder.layers)

        output_dims = self.encoder.output_dims

        if config.decoder.get("layers"):
            self.second_enc = modules_stack(config.decoder.layers)

            output_dims = self.second_enc.output_dims
        else:
            self.second_enc = None

        if phoneme_rec:
            vocab_size = len(phoneme_vocab)
        else:
            vocab_size = len(vocab)

        eos_token_id = vocab_size - 1

        t5_config = get_t5_config(
            input_dims=output_dims,
            num_layers=config.decoder.t5_num_encoder_layers,
            num_decoder_layers=config.decoder.t5_num_decoder_layers,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
        )

        self.t5_model = T5ForConditionalGeneration(t5_config)

        # 0: padding
        # 1: input
        # 2: masked
        # 3: eos token
        # since mamba has not support masking out padded inputs
        self.mask_tokens = nn.Embedding(
            num_embeddings=4, embedding_dim=256, padding_idx=1
        )

    def forward(self, spikePow, spikePow_mask, spikePow_lens, labels=None):
        # spikePow (batch_size, input_len, input_channels)
        mask_embeddings = self.mask_tokens(spikePow_mask)

        # assume spikePow is padded and masked with 0
        spikePow = spikePow + mask_embeddings

        hidden_states, output_lens = self.encoder(spikePow, spikePow_lens)

        if self.second_enc is not None:
            hidden_states, output_lens = self.second_enc(hidden_states, output_lens)

        if labels is not None:
            return self.t5_model(inputs_embeds=hidden_states, labels=labels)

        # logits = self.t5_model.generate(inputs_embeds=hidden_states, max_new_tokens=150, output_scores=True,return_dict_in_generate=True)

        return self.t5_model.generate(inputs_embeds=hidden_states, max_new_tokens=150)

    def calc_loss(self, batch):
        if self.phoneme_rec:
            labels = batch["phonemize_ids"]
        else:
            labels = batch["sent_ids"]

        re = self(
            spikePow=batch["spikePow"],
            spikePow_mask=batch["spikePow_mask"],
            spikePow_lens=batch["spikePow_lens"],
            labels=labels,
        )

        loss = re.loss
        logits = re.logits

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
        self.val_tar = [
            s.replace("_", "").replace("+", "").replace("-", "") for s in self.val_tar
        ]
        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        pred = [dec(s).replace("+", "").replace("_", "") for s in self.val_pred]

        wer = torchmetrics.functional.text.word_error_rate(pred, self.val_tar)

        self.log("wer", wer, prog_bar=True)

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
            spikePow_lens=batch["spikePow_lens"],
        )

        self.test_pred += res

    def on_test_epoch_end(self):

        if self.phoneme_rec:
            dec = phonetic_decode
        else:
            dec = decode

        pred = [dec(s).replace("_", "").replace("+", "") for s in self.test_pred]

        with open("test.txt", "w") as txt_file:
            for i in range(len(self.test_pred)):
                txt_file.write(f"{pred[i]}\n")
        self.test_pred = []


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
    def __init__(self, config: DictConfig, decoders_conf=None):
        # decoder_conf: if not none, only load those layers
        super(joint_Model, self).__init__()
        self.save_hyperparameters()

        self.config = config

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

        # 0: padding
        # 1: input
        # 2: masked
        # since mamba has not support masking out padded inputs
        self.mask_tokens = nn.Embedding(
            num_embeddings=3, embedding_dim=256, padding_idx=1
        )

    def forward(self, spikePow, spikePow_mask, spikePow_lens, encoder_only=False):
        # _input (batch_size, input_len, input_channels)

        mask_embeddings = self.mask_tokens(spikePow_mask)

        # assume spikePow is padded and masked with 0
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

        for k, l in zip(self.decoders.keys(), losses):
            self.log(f"train_{k}_loss", l, batch_size=len(batch["spikePow"]))

        self.log("train_loss", loss.detach(), prog_bar=True)

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

            target = (
                batch["phonemized"] if self.decoders[k].phoneme_rec else batch["sent"]
            )

            target = [
                s.replace("|", " ").replace("-", "").replace("+", "") for s in target
            ]

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
