from torch import nn
from torch.nn import functional as F


class resnet_block(nn.Module):
    def __init__(self, input_dims, output_dims=None, hidden_size=None, stride=2):
        super(resnet_block, self).__init__()
        if output_dims is None:
            output_dims = input_dims
        
        if hidden_size is None:
            hidden_size = output_dims
        
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.output_dims = output_dims
        self.stride = stride

        self.conv1 = nn.Conv1d(
            in_channels=input_dims,
            out_channels=hidden_size,
            kernel_size=3,
            padding_mode="replicate",
            padding=1,
            stride=stride,
        )

        self.bn1 = nn.BatchNorm1d(hidden_size)

        # self.act = nn.ReLU()
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
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
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual_part

        convoluted = self.act(out + residual_part)

        return convoluted.transpose(1, 2)
