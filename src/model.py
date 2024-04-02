import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.text import WordErrorRate
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Union
from transformers.optimization import get_linear_schedule_with_warmup

from mamba import MambaVec2Vec, MambaConfig


@dataclass
class B2TConfig:
    d_model: int = 2560
    n_layer: int = 64
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    tie_embeddings: bool = True
    bidirectional: bool = False
    bidirectional_strategy: Union[str, None] = None


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


class length_reduction_block(L.LightningModule):
    def __init__(self, input_dims, output_dims=None):
        """concatenate 2 consecutive inputs and forward through a highway network"""
        super(length_reduction_block, self).__init__()

        if output_dims == None:
            output_dims = input_dims * 2

        self.input_dims = input_dims
        self.output_dims = output_dims

        self._highways = Highway(input_dims * 2, 1, activation=nn.functional.relu)

        if input_dims * 2 != output_dims:
            self._projection = nn.Linear(input_dims * 2, output_dims, bias=True)
        else:
            self._projection = nn.Identity()

    def forward(self, hidden_states):
        # hidden_states  (batch_size, seq_len, input_dims)
        # output         (batch_size, seq_len//2, input_dims*2)
        batch_size, seq_len, input_dims = hidden_states.shape
        assert input_dims == self.input_dims
        assert seq_len % 2 == 0

        # (batch_size, seq_len//2, input_dims*2)
        hidden_states = hidden_states.reshape(
            (batch_size, seq_len // 2, input_dims * 2)
        )

        # (batch_size, seq_len//2, input_dims*2)
        hidden_states = self._highways(hidden_states)

        # (batch_size, seq_len//2, output_dims)
        hidden_states = self._projection(hidden_states)

        # TODO: add normalization

        return hidden_states


class B2T_Model(L.LightningModule):
    def __init__(
        self,
        lr=1e-5,
        log_dir=None,
        vocab_size=150,
        input_channels=256,
    ):
        super(B2T_Model, self).__init__()
        self.input_channels = input_channels

        # config = MambaConfig(
        #     d_model=input_channels,
        #     n_layer=6,
        #     bidirectional=True,
        # )

        # self.encoder = MambaVec2Vec(config=config)

        self.encoder = nn.Sequential(
            # (batch_size, seq_len, input_dims)
            MambaVec2Vec(
                MambaConfig(
                    d_model=input_channels,
                    n_layer=2,
                    bidirectional=True,
                )
            ),
            # (batch_size, seq_len//2, input_dims*2)
            length_reduction_block(input_dims=input_channels),
            MambaVec2Vec(
                MambaConfig(
                    d_model=input_channels * 2,
                    n_layer=2,
                    bidirectional=True,
                )
            ),
            # (batch_size, seq_len//4, input_dims*2)
            length_reduction_block(input_dims=input_channels * 2, output_dims=input_channels*2),
            MambaVec2Vec(
                MambaConfig(
                    d_model=input_channels * 2,
                    n_layer=2,
                    bidirectional=True,
                )
            ),
            # (batch_size, seq_len//8, input_dims*2)
            length_reduction_block(input_dims=input_channels * 2, output_dims=input_channels * 2),
            MambaVec2Vec(
                MambaConfig(
                    d_model=input_channels * 2,
                    n_layer=2,
                    bidirectional=True,
                )
            ),
        )

        self.linear = nn.Linear(input_channels * 2, vocab_size)

        self.lr = lr
        self.log_dir = log_dir

    def forward(
        self,
        _input,
        input_cu_seqlens,
    ):
        # _input                         (batch_size, number_of_blocks*block_size, input_channels)
        # input_block_attention_mask              (batch_size, number_of_blocks)
        # labels                        (batch_size, sent_length)
        # labels_mask                   (batch_size, sent_length)
        # block_size                    int

        hidden_states = self.encoder(
            _input,
            # cu_seqlens=input_cu_seqlens
        )

        res = self.linear(hidden_states)

        return res.log_softmax(-1)

    def training_step(self, batch):

        res = self(
            _input=batch["input"],
            input_cu_seqlens=batch["input_cu_seqlens"],
        )

        batch["input_cu_seqlens"] = batch["input_cu_seqlens"] // 8

        loss = F.ctc_loss(
            res.transpose(0, 1),
            batch["labels"],
            batch["input_cu_seqlens"],
            batch["labels_len"],
        )

        self.log("train_loss", loss.detach(), prog_bar=True)

        return loss

    def validation_step(self, batch):
        res = self(
            _input=batch["input"],
            input_cu_seqlens=batch["input_cu_seqlens"],
        )

        batch["input_cu_seqlens"] = batch["input_cu_seqlens"] // 8

        loss = F.ctc_loss(
            res.transpose(0, 1),
            batch["labels"],
            batch["input_cu_seqlens"],
            batch["labels_len"],
        )

        self.log("valid_loss", loss, batch_size=len(batch["input"]), prog_bar=True)

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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # return self.optimizer
        num_steps = self.num_steps()

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_steps * 0.20,
            num_training_steps=num_steps,
        )

        return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]
