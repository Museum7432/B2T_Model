import torch
from torch import nn
from torch.nn import functional as F


class conv_block(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size=3, stride=2, groups=1):
        super(conv_block, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims

        assert kernel_size % 2 == 1
        pad = kernel_size//2

        self.conv = nn.Conv1d(
            in_channels=input_dims,
            out_channels=output_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            padding_mode="replicate",
            groups=groups,
        )

        self.bn = nn.BatchNorm1d(output_dims)

        # self.act = nn.ReLU()
        self.act = nn.GELU()

    def forward(self, hidden_states):
        # transpose input for convolution
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.act(self.conv(hidden_states))

        hidden_states = self.bn(hidden_states)

        return hidden_states.transpose(1, 2)
