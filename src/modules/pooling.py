import torch
from torch import nn
from torch.nn import functional as F

class consecutive_pooling(nn.Module):
    def __init__(self, pooling_type="max", group_size=2):
        """perform pooling over group_size consecutive inputs"""
        super(consecutive_pooling, self).__init__()
        self.pooling_type = pooling_type
        self.group_size = group_size

        assert pooling_type in ["max", "mean", "min"]

    def forward(self, hidden_states, input_len):
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
        
        input_len = input_len // self.group_size

        return hidden_states, input_len
