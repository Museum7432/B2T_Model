# about
This is a model that convert EGG signal to text for the task of decoding silent speech (for the [Brain-to-text '24](https://eval.ai/web/challenges/challenge-page/2099/overview) competetion).

# architecture
### feature extractor:
multiple designs were tried but it seem like a simple depth wise convolution followed by a highway network and finally a convolution layer perform best.
### encoder:
for the enocder, a modified version of [mamba](https://github.com/state-spaces/mamba) that support bi-directionality and variable sequence length. I did try to use the T5 encoder for it relative positional embedding but it didn't work well.
### decoder
CTC loss was use to train the decoder. And for better performance, the model are made to decode both the text and the phonemized version of that text. In older commit, the model was also trained with a auto-regressive decoder but the fintuning the hyperparameters with that decoder was quite hard so it is omitted in newer version.

# setup
this project requires python 3.11 (python 3.12 is not support by mamba yet). Additionally, `festival` is aslo require for the data preparation step.
the dataset should be downloaded on first run. For the sake of simplicity, it will download the formated version of the dataset which is the result of the prepare data script.

# training
run `python train.py model=b1` to start the training.

# result
the raw output of the model achived by the training script (with greedy ctc decoder) on the validation set:
wer|wer (phonemized version)
--|--
0.38|0.35

then using the CTC decoder with the 4-gram kenlm model provide by the PyTorch [tutorial](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#sphx-glr-tutorials-asr-inference-with-ctc-decoder-tutorial-py) will give you a wer of 24% (or 18.8% on the test set).
# work in progress
