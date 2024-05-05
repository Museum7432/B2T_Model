import torch
from torch import nn
from torch.nn import functional as F

class consecutive_pooling(nn.Module):
    def __init__(self, pooling_type="max", kernel_size=3, stride=2):
        """perform pooling over group_size consecutive inputs"""
        super(consecutive_pooling, self).__init__()
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride

        assert pooling_type in ["max", "mean"]

        if pooling_type == "max":
            self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        elif pooling_type == "mean":
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, hidden_states, input_lens):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.pool(hidden_states)

        output_lens = (spikePow_lens - self.kernel_size) // self.stride + 1
        
        return hidden_states.transpose(1, 2), output_lens
