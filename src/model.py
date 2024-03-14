import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torchmetrics.text import WordErrorRate
import pathlib
from datetime import datetime
from transformers.optimization import get_linear_schedule_with_warmup

# adapted from https://github.com/helboukkouri/character-bert/blob/main/modeling/character_cnn.py


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


class CNN_block(L.LightningModule):
    def __init__(self, input_channels, output_channels):
        super(CNN_block, self).__init__()

        # TODO: try larger kernel size
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.bn1 = nn.BatchNorm1d(output_channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.bn2 = nn.BatchNorm1d(output_channels)

        if input_channels != output_channels:
            self.residual = nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding="same",
            )
        else:
            self.residual = nn.Identity()

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
    def __init__(self, input_channels=256, hidden_size=768, embedding_size=512):
        super(Signal_CNN, self).__init__()

        self.embedding_size = embedding_size
        self.input_channels = input_channels
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            CNN_block(input_channels, hidden_size),
            CNN_block(hidden_size, hidden_size),
            CNN_block(hidden_size, hidden_size),
        )

        self._highways = Highway(hidden_size, 2, activation=nn.functional.relu)
        self._projection = torch.nn.Linear(hidden_size, embedding_size, bias=True)

    def forward(self, _input, block_size):
        # _input  (batch_size, number_of_blocks*block_size, input_channels)
        # output (batch_size, number_of_blocks, embedding_size)

        batch_size, seq_length, input_channels = _input.shape

        number_of_blocks = seq_length // block_size

        # transpose input for convolution
        # ==> (batch_size, input_channels, number_of_blocks*block_size)
        _input = _input.transpose(1, 2)

        # DONE TODO: do convolution here with padding='same'
        # differ from characterCNN, in SignalCNN embbeding, convolutions are not done per block
        # ==> (batch_size, hidden_size, number_of_blocks*block_size)

        convoluted = self.conv(_input)

        # partionning input into blocks of size block_size
        # ==> (batch_size, hidden_size, number_of_blocks, block_size)
        convoluted = convoluted.reshape((batch_size, self.hidden_size, number_of_blocks, block_size))

        # perform max pooling per block
        # TODO: we could try average pooling
        # ==> (batch_size, hidden_size, number_of_blocks)
        blocks_prep = torch.max(convoluted, dim=-1).values
        # blocks_prep = torch.mean(convoluted, dim=-1).values

        # transpose input again
        # ==> (batch_size, number_of_blocks, hidden_size)
        blocks_prep = blocks_prep.transpose(1, 2)

        # ==> (batch_size*number_of_blocks, hidden_size)
        blocks_prep = blocks_prep.reshape((batch_size*number_of_blocks, self.hidden_size))

        # DONE TODO: use highway and projection layers here to convert hidden_size into embedding_size

        blocks_prep = self._highways(blocks_prep)

        # ==> (batch_size*number_of_blocks, embedding_size)
        embedding = self._projection(blocks_prep)

        # ==> (batch_size, number_of_blocks, embedding_size)
        embedding = embedding.reshape(
            (batch_size, number_of_blocks, self.embedding_size)
        )

        return embedding


# add absolute positional embedding to Signal_CNN
# not needed if relative postional embedding is implemented in the encoder
class Signal_Embeddings(L.LightningModule):
    def __init__(
        self,
        input_channels=256,
        hidden_size=768,
        embedding_size=512,
        max_block_index=511,
    ):
        super(Signal_Embeddings, self).__init__()

        self.embedding_size = embedding_size
        self.input_channels = input_channels
        self.hidden_size = hidden_size

        self.signal_embeddings = Signal_CNN(
            input_channels=input_channels,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
        )
        # TODO: we should try relative positional embedding
        self.position_embeddings = nn.Embedding(max_block_index, embedding_size)
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, _input, block_size):
        # _input  (batch_size, number_of_blocks*block_size, input_channels)
        # convert input into block representation
        # ==> (batch_size, number_of_blocks, embedding_size)

        blocks_embedding = self.signal_embeddings(_input, block_size)

        batch_size, number_of_blocks, embedding_size = blocks_embedding.shape

        position_ids = (
            torch.arange(number_of_blocks)
            .unsqueeze(0)
            .expand_as(blocks_embedding[:, :, 0])
        )

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = blocks_embedding + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class B2T_Model(L.LightningModule):
    def __init__(
        self,
        lr=1e-5,
        log_dir=None,

        input_channels=256,
        conv_hidden_size=None,
        seq2seq_model="google-t5/t5-small",
    ):
        super(B2T_Model, self).__init__()
        self.input_channels = input_channels

        # DONE TODO: try relative positional embeding
        self.encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model)

        self.embedding_size = self.encoder_decoder.get_input_embeddings().embedding_dim

        if conv_hidden_size:
            self.conv_hidden_size = conv_hidden_size
        else:
            self.conv_hidden_size = self.embedding_size



        self.embeddings = Signal_CNN(
            input_channels=input_channels,
            hidden_size=self.conv_hidden_size,
            embedding_size=self.embedding_size,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(seq2seq_model)

        self.wer = WordErrorRate()

        self.lr = lr
        self.log_dir = log_dir

    def forward(
        self,
        _input,
        input_block_attention_mask,
        block_size=16,
        labels=None,
        labels_mask=None,
    ):
        # _input                         (batch_size, number_of_blocks*block_size, input_channels)
        # input_block_attention_mask              (batch_size, number_of_blocks)
        # labels                        (batch_size, sent_length)
        # labels_mask                   (batch_size, sent_length)
        # block_size                    int

        # (batch_size, number_of_blocks, embedding_size)
        embedding_output = self.embeddings(_input=_input, block_size=block_size)

        if labels is None:
            return self.encoder_decoder.generate(
                inputs_embeds=embedding_output,
                attention_mask=input_block_attention_mask,
                max_new_tokens=30
            )

        return self.encoder_decoder(
            inputs_embeds=embedding_output,
            attention_mask=input_block_attention_mask,
            labels=labels,
        )

    def training_step(self, batch):
        loss = self(
            _input=batch["input"],
            input_block_attention_mask=batch["input_block_attention_mask"],
            block_size=batch["block_size"],
            labels=batch["labels"],
            labels_mask=batch["labels_mask"],
        ).loss

        self.log("train_loss", loss.detach(), prog_bar=True)

        return loss

    def validation_step(self, batch):
        res = self(
            _input=batch["input"],
            input_block_attention_mask=batch["input_block_attention_mask"],
            block_size=batch["block_size"],
        )

        pred_sents = self.tokenizer.batch_decode(res, skip_special_tokens=True)

        # batch["sent"]

        w_score = self.wer(pred_sents, batch["sent"])

        self.log("batch_val_wer", w_score, batch_size=len(batch["input"]))

    def on_validation_epoch_end(self):
        w_score = self.wer.compute()
        self.log("val_wer", w_score, prog_bar=True)
        self.wer.reset()

    def on_test_epoch_start(self):
        self.test_res = []


    @torch.no_grad()
    def test_step(self, batch):
        res = self(
            _input=batch["input"],
            input_block_attention_mask=batch["input_block_attention_mask"],
            block_size=batch["block_size"],
        )
        pred_sents = self.tokenizer.batch_decode(res, skip_special_tokens=True)

        self.test_res += pred_sents
    
    def on_test_epoch_end(self):
        # self.global_step
        pathlib.Path(self.log_dir, "eval").mkdir(exist_ok=True)

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        output_file = pathlib.Path(
            self.log_dir, "eval", datetime.now().strftime("%H_%M_%S") + ".txt"
        )

        with open(output_file, "w") as txt_file:
            for i in self.test_res:
                txt_file.write(str(i) + "\n")

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = (
            dataset_size
            * self.trainer.max_epochs
            // (self.trainer.accumulate_grad_batches * num_devices)
        )
        return num_steps

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # return self.optimizer
        num_steps = self.num_steps()
        
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_steps * 0.20,
            num_training_steps=num_steps,
        )

        return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]