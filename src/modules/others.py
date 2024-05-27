from torch.nn import functional as F
import torch
import torch.nn as nn

"""
unpack function: convert packed_hidden_states (batch_size=1) to hidden_states
"""


def unpack(packed_hidden_states, cu_seqlens):
    batch_size = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    packed_hidden_states = packed_hidden_states.squeeze(0)

    ori_indices = (
        torch.arange(seq_len, device=cu_seqlens.device)
        .unsqueeze(0)
        .expand((batch_size, seq_len))
    )

    ori_indices = (ori_indices + cu_seqlens[:-1].unsqueeze(1)) % (
        len(packed_hidden_states)
    )

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


def reverse_hidden_states(hidden_states, seq_lens):
    batch_size, seq_len, hidden_dim = hidden_states.shape

    assert batch_size == len(seq_lens)

    indices = torch.arange(batch_size * seq_len, device=hidden_states.device).reshape(
        batch_size, seq_len
    )

    indices_offset = seq_len - seq_lens

    indices = (indices - indices_offset.unsqueeze(1)) % (seq_len * batch_size)

    indices = indices.flip(1)

    return hidden_states.reshape(batch_size * seq_len, hidden_dim)[indices]


def stochastic_update(new_state, old_state, mask=None, update_probs=None):
    # batch_size, seq_len, hidden_size

    if mask is None and update_probs is None:
        return new_state, mask

    if old_state is None:
        return new_state, mask

    batch_size, seq_len, hidden_size = old_state.shape

    if mask is None:
        mask = torch.rand((batch_size, seq_len), device=old_state.device) < update_probs
        mask = mask.unsqueeze(-1).expand_as(old_state)

    output_state = torch.where(mask, new_state, old_state)

    return output_state, mask


class Pack(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_lens):
        if len(hidden_states) == 1:
            return hidden_states, input_lens

        cu_seqlens = F.pad(input_lens.cumsum(0), pad=(1, 0), value=0).to(torch.int32)
        packed_hidden_states = pack(hidden_states, cu_seqlens).unsqueeze(0)

        return packed_hidden_states, input_lens


class UnPack(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_lens):
        if len(input_lens) == 1:
            return hidden_states, input_lens

        cu_seqlens = F.pad(input_lens.cumsum(0), pad=(1, 0), value=0).to(torch.int32)
        hidden_states = unpack(hidden_states, cu_seqlens)

        return hidden_states, input_lens
