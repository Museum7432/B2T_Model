from torch.nn import functional as F
import torch
import torch.nn as nn

"""
unpack function: convert packed_hidden_states (batch_size=1) to hidden_states
"""


def unpack(packed_hidden_states, cu_seqlens):
    batch_size = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    hidden_dim = packed_hidden_states.shape[2]
    hidden_states = torch.zeros(
        batch_size,
        seq_len,
        hidden_dim,
        dtype=packed_hidden_states.dtype,
        device=packed_hidden_states.device,
    )
    for i in range(batch_size):
        hidden_states[i, : cu_seqlens[i + 1] - cu_seqlens[i], :] = packed_hidden_states[
            :, cu_seqlens[i] : cu_seqlens[i + 1], :
        ]
    return hidden_states


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


class Pack(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, hidden_states, input_lens):
        if len(hidden_states) == 1:
            return hidden_states, input_lens

        cu_seqlens = F.pad(input_lens.cumsum(0), pad=(1, 0), value=0).to(
            torch.int32
        )
        packed_hidden_states = pack(hidden_states, cu_seqlens).unsqueeze(0)
        
        
        return packed_hidden_states, input_lens

class UnPack(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, hidden_states, input_lens):
        if len(input_lens) == 1:
            return hidden_states, input_lens

        cu_seqlens = F.pad(input_lens.cumsum(0), pad=(1, 0), value=0).to(
            torch.int32
        )
        hidden_states = unpack(hidden_states, cu_seqlens)
        
        return hidden_states, input_lens