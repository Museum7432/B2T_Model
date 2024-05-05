from torch import nn
from torch.nn import functional as F
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5ForConditionalGeneration,
    T5PreTrainedModel,
    T5EncoderModel
)

class t5_encoder(nn.Module):
    def __init__(self, input_dims, n_layer):
        super(t5_encoder, self).__init__()

        self.input_dims = input_dims
        self.output_dims = input_dims

        base_config = T5Config.from_pretrained("google-t5/t5-base")
        base_config.d_model = input_dims
        base_config.num_layers = n_layer
        base_config.vocab_size = 1
        self.model = T5EncoderModel(base_config)
    def forward(self, hidden_states, input_lens):
        hidden_states = self.model(inputs_embeds=hidden_states).last_hidden_state

        return hidden_states, input_lens