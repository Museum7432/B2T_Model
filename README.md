# about
This is a model that convert EGG signal to text for the task of decoding silent speech (for the [Brain-to-text '24](https://eval.ai/web/challenges/challenge-page/2099/overview) competetion).

# architecture
### feature extractor:
multiple designs were tried but it seem like a simple depth wise convolution followed by a highway network and finally a convolutional layer perform best.
### encoder:
for the enocder, a modified version of [mamba](https://github.com/state-spaces/mamba) that support bi-directionality and variable sequence length is used (check `src/modules/mamba.py` if you are interested). I did try to use the T5 encoder for its relative positional embedding but it didn't work as well as mamba. It could be that a RNN is better suited for this task than a transformer.
### decoder
CTC loss was use to train the decoder. And for better performance, the model are made to decode both the text and the phonemized version of that text. In older commit, the model was also trained with a auto-regressive decoder (the T5 model) but the fintuning the hyperparameters with that decoder was quite hard so it is omitted in newer version.

# setup
this project requires python 3.11 (python 3.12 is not support by mamba yet). Additionally, `festival` is aslo require for the data preparation step.
the dataset should be downloaded on first run. And for the sake of simplicity, it will download the formated version of the dataset which is the output of the prepare data script.

# training
run `python train.py model=b1` to start the training. It should take 20 minute on a rtx 3060 (or 1 hour on my laptop with a gtx 1650).

# result
the raw output of the model achieved with the training script (with greedy ctc decoder) on the validation set:
wer|wer (phonemized version)
--|--
0.38|0.35

then using the CTC decoder with the 4-gram kenlm model provide by the PyTorch [tutorial](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#sphx-glr-tutorials-asr-inference-with-ctc-decoder-tutorial-py) will give you a wer of 24% (or 18.8% on the test set).

14th place!. Which is quite obvious since there's just me working on this project. There are still multiple ideas i could try but since free time is quite limited as a college student so if someone is interested, you could fork this project. 
# work in progress
* use additional corpus: inspired by the GAN network, basically train a separate encoder that project text into the same latent space as the output of the encoder by making the decoder (with gradient off) ouput the disired text. then a second decoder is train with the second enocder ouput (detached). In theory, the decoder should learn how to spell better due to the addtional data but in practice the second loss converge way too fast before anything of use was learn, so tunning the loss weight could work. There should be a (not) working prototype of this idea in the `src` folder.
* mask word: use a trained neural decoder to algin the text label to the neural recording. Then, in each training step we mask a work and its associate spikepower region. It should force the model to predict a word without relying on the context.
