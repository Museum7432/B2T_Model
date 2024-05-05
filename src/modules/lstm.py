from torch import nn
from torch.nn import functional as F

class lstm_block(nn.Module):
    def __init__(self, input_size, num_layers, bidirectional):
        super(lstm_block, self).__init__()

        self.input_dims = input_size
        self.output_dims = input_size

        if bidirectional:
            self.output_dims = 2 * input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.05,
            bidirectional=bidirectional
        )
    def forward(self, hidden_states, input_lens):
        
        hidden_states, (h_n, c_n) = self.lstm(hidden_states)

        return hidden_states, input_lens
