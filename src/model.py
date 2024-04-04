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

from .mamba import MambaFeatureExtractor, MambaConfig
from omegaconf import DictConfig


# copied from https://github.com/helboukkouri/character-bert/blob/main/modeling/character_cnn.py
class Highway(torch.nn.Module):
    """
    A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    # Parameters

    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation=torch.nn.functional.relu,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class consecutive_pooling(nn.Module):
    def __init__(self, pooling_type="max", group_size=2):
        """perform pooling over group_size consecutive inputs"""
        super(consecutive_pooling, self).__init__()
        self.pooling_type = pooling_type
        self.group_size = group_size

        assert pooling_type in ["max", "mean", "min"]

    def forward(self, hidden_states):
        # hidden_states  (batch_size, seq_len, input_dims)
        batch_size, seq_len, input_dims = hidden_states.shape
        assert seq_len % self.group_size == 0

        # (batch_size, seq_len//group_size, group_size, input_dims)
        hidden_states = hidden_states.reshape(
            (batch_size, seq_len // self.group_size, self.group_size, input_dims)
        )

        # (batch_size, seq_len//2, input_dims, 2)
        hidden_states = hidden_states.transpose(2, 3)

        # (batch_size, seq_len//2, input_dims)
        if self.pooling_type == "max":
            hidden_states = hidden_states.max(-1).values
        elif self.pooling_type == "min":
            hidden_states = hidden_states.min(-1).values
        else:
            hidden_states = hidden_states.mean(-1)

        return hidden_states


class concatenate_consecutive(nn.Module):
    def __init__(self, input_dims, output_dims=None, group_size=2):
        """concatenate group_size consecutive inputs and forward through a highway network"""
        super(concatenate_consecutive, self).__init__()

        self.group_size = group_size
        self.input_dims = input_dims
        self.output_dims = output_dims

        if output_dims == None:
            output_dims = input_dims * group_size

        self._highways = Highway(
            input_dims * group_size, 2, activation=nn.functional.relu
        )

        if input_dims * group_size != output_dims:
            self._projection = nn.Linear(
                input_dims * group_size, output_dims, bias=True
            )
        else:
            self._projection = nn.Identity()

    def forward(self, hidden_states):
        # hidden_states  (batch_size, seq_len, input_dims)

        batch_size, seq_len, input_dims = hidden_states.shape
        assert input_dims == self.input_dims
        assert seq_len % self.group_size == 0

        # (batch_size, seq_len//group_size, input_dims*group_size)
        hidden_states = hidden_states.reshape(
            (batch_size, seq_len // self.group_size, input_dims * self.group_size)
        )

        # (batch_size, seq_len//group_size, input_dims*group_size)
        hidden_states = self._highways(hidden_states)

        # (batch_size, seq_len//group_size, output_dims)
        hidden_states = self._projection(hidden_states)

        return hidden_states


class convolutional_block(nn.Module):
    def __init__(self, input_dims, output_dims=None, stride=2):
        super(convolutional_block, self).__init__()
        # convolution block have a constant stride of 2
        if output_dims is None:
            output_dims = input_dims
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.stride = stride

        # self.conv1 = nn.Conv1d(
        #     in_channels=input_dims,
        #     out_channels=output_dims,
        #     kernel_size=4,
        #     padding_mode="replicate",
        #     padding=(1, 2),
        #     stride=stride,
        # )

        self.conv1 = nn.Conv1d(
            in_channels=input_dims,
            out_channels=output_dims,
            kernel_size=3,
            padding_mode="replicate",
            padding=1,
            stride=stride,
        )

        self.bn1 = nn.BatchNorm1d(output_dims)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=output_dims,
            out_channels=output_dims,
            kernel_size=3,
            padding_mode="replicate",
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm1d(output_dims)

        if input_dims != output_dims:
            self.residual = nn.Conv1d(
                in_channels=input_dims,
                out_channels=output_dims,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.residual = nn.MaxPool1d(kernel_size=1, stride=stride)

    def forward(self, hidden_states):
        # hidden_states  (batch_size, seq_len, input_dims)

        # transpose input for convolution
        hidden_states = hidden_states.transpose(1, 2)

        residual_part = self.residual(hidden_states)

        out = self.conv1(hidden_states)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual_part

        convoluted = self.relu(out + residual_part)

        return convoluted.transpose(1, 2)


class B2T_encoder(L.LightningModule):
    def __init__(self, config):
        super(B2T_encoder, self).__init__()
        # config.encoder.layers example
        # [
        #   ["mamba", 256, 2, True],  # input_channels, n_layer, bidirectional
        #   ["concat", 256, 512, 2],  # input_dims, output_dims, group_size
        #   ["pooling", "max", 2],  # pooling_type, group_size
        # ]

        modules_list = []

        for l in config.encoder.layers:
            if l[0] == "mamba":
                _, in_channels, n_layer, bidirectional = l
                modules_list.append(
                    MambaFeatureExtractor(
                        MambaConfig(
                            d_model=in_channels,
                            n_layer=n_layer,
                            bidirectional=bidirectional,
                        )
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
                _, in_dims, out_dims, stride = l
                modules_list.append(
                    convolutional_block(input_dims=in_dims, output_dims=out_dims, stride=stride)
                )
            else:
                raise ValueError(f"unknown layer: {l[0]}")

        self.layers = nn.Sequential(*modules_list)

    def forward(self, _input, input_len=None):
        hidden_states = self.layers(
            _input,
            # cu_seqlens=input_len
        )
        # cu_seqlens is not available in mamba yet
        return hidden_states


class BasedModel(L.LightningModule):
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


class B2T_Phonemes_CTC(BasedModel):
    def __init__(self, config: DictConfig):
        super(B2T_Phonemes_CTC, self).__init__()
        self.save_hyperparameters()

        self.config = config

        self.encoder = B2T_encoder(config)

        self.linear = nn.Linear(
            config.encoder.output_dims, config.ctc_decoder.phoneme_vocab_size
        )

    def forward(self, _input, input_len):
        # _input (batch_size, input_len, input_channels)
        hidden_states = self.encoder(
            _input,
            # cu_seqlens=input_cu_seqlens
        )
        res = self.linear(hidden_states)
        return res.log_softmax(-1)

    def training_step(self, batch):
        # mask part of the input

        res = self(
            _input=batch["input"],
            input_len=batch["input_len"],
        )

        new_input_len = batch["input_len"] // self.config.encoder.input_reduction_ratio

        loss = F.ctc_loss(
            res.transpose(0, 1),
            batch["phonemize_ids"],
            new_input_len,
            batch["phonemize_ids_len"],
        )

        self.log("train_loss", loss.detach(), prog_bar=True)

        return loss

    def validation_step(self, batch):
        res = self(
            _input=batch["input"],
            input_len=batch["input_len"],
        )

        new_input_len = batch["input_len"] // self.config.encoder.input_reduction_ratio

        loss = F.ctc_loss(
            res.transpose(0, 1),
            batch["phonemize_ids"],
            new_input_len,
            batch["phonemize_ids_len"],
        )

        self.log("valid_loss", loss, batch_size=len(batch["input"]), prog_bar=True)
