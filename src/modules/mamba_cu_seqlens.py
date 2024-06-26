# Copyright (c) 2023, Albert Gu, Tri Dao.

from dataclasses import dataclass, field
from typing import Union

from torch.nn import functional as F


@dataclass
class MambaConfig:
    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 1
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    bidirectional: bool = False
    bidirectional_strategy: Union[str, None] = None
    update_probs: float = 0.5


import math
from functools import partial
import json
import os
from typing import Union, Optional

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


"""
unpack function: convert packed_hidden_states (batch_size=1) to hidden_states
"""


def unpack(packed_hidden_states, cu_seqlens):
    batch_size = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]

    packed_hidden_states = packed_hidden_states.squeeze(0)

    ori_indices = torch.arange(batch_size* seq_len, device=cu_seqlens.device).reshape((batch_size, seq_len))

    indices_offset = seq_len - seq_len_list

    last_offset = indices_offset[-1].item()

    indices_offset[-1] = 0

    flatten_indices_offset = indices_offset.roll(1).cumsum(0)
    ori_indices = ori_indices - flatten_indices_offset.unsqueeze(1)
    
    # pad packed_hidden_states
    packed_hidden_states = F.pad(packed_hidden_states, pad=(0, 0, 0, last_offset))
    return packed_hidden_states[ori_indices]

"""
pack function: convert hidden_states to packed_hidden_states (batch_size=1)
"""


def pack(hidden_states, cu_seqlens):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_len_list_3d = seq_len_list.unsqueeze(1).unsqueeze(2)
    indices_3d = (
        torch.arange(seq_len, device=hidden_states.device)
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(batch_size, 1, hidden_dim)
    )
    mask_3d = indices_3d < seq_len_list_3d
    packed_hidden_states = hidden_states[mask_3d].view(-1, hidden_dim)
    return packed_hidden_states


def pack2d(input_ids, cu_seqlens):
    batch_size, seq_len = input_ids.shape

    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]

    seq_len_list_2d = seq_len_list.unsqueeze(1)

    indices_2d = (
        torch.arange(seq_len, device=input_ids.device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    mask_2d = indices_2d < seq_len_list_2d
    packed_input_ids = input_ids[mask_2d].view(-1)
    return packed_input_ids


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        cu_seqlens=None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(
            hidden_states, cu_seqlens=cu_seqlens, inference_params=inference_params
        )
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    bidirectional=False,
    bidirectional_strategy=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
    }
    mixer_cls = partial(
        MambaWrapper,
        layer_idx=layer_idx,
        **ssm_cfg,
        **bidirectional_kwargs,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = False,
        bidirectional_strategy: Optional[str] = None,
        **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(
                f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!"
            )
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        if bidirectional:
            self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, cu_seqlens=None, inference_params=None):
        """Bidirectional-enabled forward pass
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(
            hidden_states, cu_seqlens=cu_seqlens, inference_params=inference_params
        )
        if self.bidirectional:

            # flip the cu_seqlens
            reverse_cu_seqlens = None
            if cu_seqlens is not None:
                reverse_cu_seqlens = torch.cumsum(
                    torch.cat(
                        (
                            torch.tensor([0], device=hidden_states.device),
                            (cu_seqlens[1:] - cu_seqlens[:-1]).flip(dims=(0,)),
                        ),
                        dim=0,
                    ),
                    dim=0,
                )

            out_rev = self.mamba_rev(
                hidden_states.flip(
                    dims=(1,)
                ),  # Flip along the sequence length dimension
                cu_seqlens=reverse_cu_seqlens,
                inference_params=inference_params,
            ).flip(
                dims=(1,)
            )  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
        return out


def stochastic_update(new_state, old_state, mask=None, update_probs=0.5):
    # batch_size, seq_len, hidden_size

    if old_state is None:
        return new_state, mask

    batch_size, seq_len, hidden_size = old_state.shape

    if mask is None:
        mask = torch.rand((batch_size, seq_len), device=old_state.device) < update_probs
        mask = mask.unsqueeze(-1).expand_as(old_state)

    output_state = torch.where(mask, new_state, old_state)

    return output_state, mask


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        bidirectional: bool = False,
        bidirectional_strategy: Optional[str] = None,
        update_probs=0.5,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.update_probs=update_probs

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bidirectional=bidirectional,
                    bidirectional_strategy=bidirectional_strategy,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, cu_seqlens=None, inference_params=None):

        residual = None
        for layer in self.layers:
            n_hidden_states, n_residual = layer(
                hidden_states,
                residual,
                cu_seqlens=cu_seqlens,
                inference_params=inference_params,
            )
            hidden_states, mask = stochastic_update(
                new_state=n_hidden_states, old_state=hidden_states, update_probs=self.update_probs
            )
            residual, _ = stochastic_update(
                new_state=n_residual, old_state=residual, mask=mask
            )

            # if self.training:
            #     hidden_states, mask = stochastic_update(
            #         new_state=n_hidden_states, old_state=hidden_states, update_probs=0.5
            #     )

            #     residual, _ = stochastic_update(
            #         new_state=n_residual, old_state=residual, mask=mask
            #     )
            # else:
            #     hidden_states = n_hidden_states

            #     residual = n_residual


        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaBlock(nn.Module):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer

        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm

        bidirectional = config.bidirectional
        bidirectional_strategy = config.bidirectional_strategy
        update_probs = config.update_probs


        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            bidirectional=bidirectional,
            bidirectional_strategy=bidirectional_strategy,
            update_probs=update_probs,
            **factory_kwargs,
        )
        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        hidden_states,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        cu_seqlens=None,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(
            hidden_states, inference_params=inference_params, cu_seqlens=cu_seqlens
        )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        return hidden_states


class mamba_block_for_input_ids(nn.Module):
    def __init__(self, d_model, n_layer, bidirectional, vocab_size, update_probs=0.5):
        super(mamba_block_for_input_ids, self).__init__()

        self.input_dims = d_model
        self.output_dims = d_model

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.model = MambaBlock(
            MambaConfig(
                d_model=d_model,
                n_layer=n_layer,
                bidirectional=bidirectional,
                update_probs=update_probs
            )
        )

    def forward(self, input_ids, input_lens):

        if input_ids.size(0) == 1:
            cu_seqlens = None
            packed_input_ids = input_ids
        else:
            cu_seqlens = F.pad(input_lens.cumsum(0), pad=(1, 0), value=0).to(
                torch.int32
            )
            packed_input_ids = pack2d(input_ids, cu_seqlens).unsqueeze(0)

        packed_hidden_states = self.embedding(packed_input_ids)

        packed_hidden_states = self.model(packed_hidden_states, cu_seqlens=cu_seqlens)

        if input_ids.size(0) == 1:
            hidden_states = packed_hidden_states
        else:
            hidden_states = unpack(packed_hidden_states, cu_seqlens)

        return hidden_states, input_lens

    # def forward(self, input_ids, input_lens):

    #     hidden_states = self.embedding(input_ids)

    #     hidden_states = self.model(hidden_states)

    #     return hidden_states, input_lens


class mamba_block(nn.Module):
    def __init__(self, d_model, n_layer, bidirectional, update_probs=0.5):
        super(mamba_block, self).__init__()

        self.input_dims = d_model
        self.output_dims = d_model

        self.model = MambaBlock(
            MambaConfig(
                d_model=d_model,
                n_layer=n_layer,
                bidirectional=bidirectional,
                update_probs=update_probs
            )
        )

    def forward(self, hidden_states, input_lens):

        if len(input_lens) == 1:
            cu_seqlens = None
        else:
            cu_seqlens = F.pad(input_lens.cumsum(0), pad=(1, 0), value=0).to(
                torch.int32
            )

        packed_hidden_states = self.model(hidden_states, cu_seqlens=cu_seqlens)

        return packed_hidden_states, input_lens
