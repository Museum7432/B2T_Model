import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5ForConditionalGeneration,
    T5PreTrainedModel,
    T5Model,
)


def get_t5_config(input_dims, num_layers):
    base_config = T5Config.from_pretrained("google-t5/t5-base")

    # we dont need vocab
    base_config.vocab_size = 1

    base_config.d_model = input_dims
    # we only need the decoder
    base_config.num_layers = 1
    base_config.num_decoder_layers = num_layers

    return base_config


class t5_conv_cross_att(nn.Module):
    def __init__(self, input_dims, num_layers):
        super(t5_conv_cross_att, self).__init__()
        self.output_dims = input_dims

        t5_conf = get_t5_config(input_dims=input_dims, num_layers=num_layers)

        self.model = T5Model(t5_conf)

        self.conv = nn.Conv1d(
            in_channels=input_dims,
            out_channels=input_dims,
            kernel_size=5,
            stride=2,
            padding=2,
            padding_mode="replicate",
            # bias=False
        )
        # self.act = nn.ReLU()
        self.act = nn.GELU()

    def forward(self, hidden_states, input_lens):

        batch_size, seq_len, hidden_size = hidden_states.shape

        keys = self.conv(hidden_states.transpose(1, 2))
        keys = self.act(keys).transpose(1, 2)

        keys_lens = (input_lens - 1) // 2 + 1

        keys_attention_mask = torch.arange(
            keys.shape[1], device=self.device
        ) < keys_lens.unsqueeze(1)

        input_attention_mask = torch.arange(
            seq_len, device=self.device
        ) < input_lens.unsqueeze(1)

        # outputs = self.model(encoder_outputs=(hidden_states), decoder_inputs_embeds=keys)
        outputs = self.model(
            inputs_embeds=hidden_states,
            # encoder_outputs=(hidden_states)
            attention_mask=input_attention_mask,
            decoder_inputs_embeds=keys,
            decoder_attention_mask=keys_attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        output_lens = (input_lens - 1) // 2

        return hidden_states, output_lens
