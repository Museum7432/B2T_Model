import torch
from torch import nn
from torch.nn import functional as F

# copied from https://github.com/helboukkouri/character-bert/blob/main/modeling/character_cnn.py
class Highway(torch.nn.Module):
    """
    A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    # Parameters

    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation=torch.nn.functional.relu,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input

class concatenate_consecutive(nn.Module):
    def __init__(self, input_dims, output_dims=None, group_size=2):
        """concatenate group_size consecutive inputs and forward through a highway network"""
        super(concatenate_consecutive, self).__init__()

        self.group_size = group_size
        self.input_dims = input_dims
        self.output_dims = output_dims

        if output_dims == None:
            output_dims = input_dims * group_size

        self._highways = Highway(
            input_dims * group_size, 2, activation=nn.functional.relu
        )

        if input_dims * group_size != output_dims:
            self._projection = nn.Linear(
                input_dims * group_size, output_dims, bias=True
            )
        else:
            self._projection = nn.Identity()

    def forward(self, hidden_states):
        # hidden_states  (batch_size, seq_len, input_dims)

        batch_size, seq_len, input_dims = hidden_states.shape
        assert input_dims == self.input_dims
        assert seq_len % self.group_size == 0

        # (batch_size, seq_len//group_size, input_dims*group_size)
        hidden_states = hidden_states.reshape(
            (batch_size, seq_len // self.group_size, input_dims * self.group_size)
        )

        # (batch_size, seq_len//group_size, input_dims*group_size)
        hidden_states = self._highways(hidden_states)

        # (batch_size, seq_len//group_size, output_dims)
        hidden_states = self._projection(hidden_states)

        return hidden_states
