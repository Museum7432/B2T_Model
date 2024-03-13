import lightning as L
import torch
from torch import nn
import transformers
from torch.nn import functional as F



class CNN_block(L.LightningModule):
    def __init__(self, input_channels, output_channels):

        # TODO: try larger kernel size
        self.conv1 = nn.conv1d(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding="same"
        )

        self.bn1 = nn.BatchNorm1d(output_channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.conv1d(
            input_channels=output_channels,
            output_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding="same"
        )

        self.bn2 = nn.BatchNorm1d(output_channels)

        self.residual = nn.conv1d(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding="same"
        )
    
    def forward(self, _input):
        # input: (batch_size, input_channels, number_of_blocks*block_size)

        # ==> (batch_size, output_channels, number_of_blocks*block_size)
        residual_part = self.residual(_input)

        out = self.conv1(_input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual_part

        out = self.relu(out)

        return out


class Signal_CNN(L.LightningModule):
    def __init__(self, input_channels = 256, embedding_size=512):
        self.embedding_size = embedding_size
        self.input_channels = input_channels

        self.conv = nn.Sequential(
            CNN_block(input_channels, embedding_size),
            CNN_block(embedding_size, embedding_size),
            CNN_block(embedding_size, embedding_size),
        )

    def forward(self, _input, block_size):
        # _input  (batch_size, number_of_blocks*block_size, input_channels)
        # output (batch_size, number_of_blocks, embedding_size)

        batch_size, seq_length, input_channels = _input.shape

        # transpose input for convolution
        # ==> (batch_size, input_channels, number_of_blocks*block_size)
        _input = _input.transpose(1, 2)

        # TODO: do convolution here with padding='same'
        # ==> (batch_size, conv_output_channels, number_of_blocks*block_size)

        convoluted = self.conv(_input)

        # partionning input into blocks of size block_size
        # ==> (batch_size, conv_output_channels, number_of_blocks, block_size)
        convoluted = convoluted.reshape((batch_size, -1, seq_length // block_size, block_size))

        # perform max pooling per block
        # TODO: we could try average pooling
        # ==> (batch_size, conv_output_channels, number_of_blocks)
        blocks_prep = torch.max(convoluted, dim=-1)

        # transpose input again
        # ==> (batch_size, number_of_blocks, conv_output_channels)
        blocks_prep = blocks_prep.transpose(1, 2)

        # TODO: use highway and projection layers here to convert conv_output_channels into embedding_size
        # ==> (batch_size, seq_length, embedding_size)

        return blocks_prep


class Signal_Embeddings(L.LightningModule):
    def __init__(self, input_channels, embedding_size, max_block_index=511):

        self.signal_embeddings = Signal_CNN(
            input_channels=input_channels,
            embedding_size=embedding_size
        )

        self.position_embeddings = nn.Embedding(max_block_index, embedding_size)

    def forward()



class B2T_Model(L.LightningModule):
    def __init__(self, input_size=256):
        self.input_size = input_size


    def forward(self, batch):
        # batch["input"]                (batch_size, number_of_blocks*block_size, input_channels)
        # batch["input_block_mask"]     (batch_size, number_of_blocks)
        # batch["labels"]               (batch_size, sent_length)
        # batch["labels_mask"]          (batch_size, sent_length)
        # batch["block_size"]           int




