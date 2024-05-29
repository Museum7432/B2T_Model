import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io as sio
from scipy import signal
import scipy
import os
import numpy as np
import torch
import random
import samplerate
import pickle

from .utils import (
    phonemize_text,
    phonetic_tokenize,
    clean_text,
    tokenize,
    phoneme_vocab,
    vocab,
)


def get_bound(_mean, _std, ep=1e-8):
    part1 = np.sqrt(-np.log(ep * _std * np.pi) * 2)

    high = _mean + _std * part1
    low = _mean - _std * part1

    return low, high


def filter_noises(spikePow, block_mean, block_std, ep=1e-8):
    # _mean = spikePow.mean(0)
    # _std = spikePow.std(0)
    # low, high = get_bound(_mean=_mean, _std=_std, ep=1e-8)

    low, high = get_bound(_mean=block_mean, _std=block_std, ep=ep)
    spikePow = np.clip(spikePow, low, high)

    return spikePow


def project(x, ratio=1.0):
    return np.rint(np.array(x) * ratio)


def flip_(spikePow, sentenceText, phonemizedText):
    spikePow = np.flip(spikePow, axis=0)

    if sentenceText is None:
        return spikePow, sentenceText, phonemizedText

    sentenceText = sentenceText[::-1]

    t = phonemizedText.replace("|", " | ")
    t = t.replace("-", " - ")
    t = t.replace("+", " + ")
    t = t.split()[::-1]
    phonemizedText = "".join(t)

    return spikePow, sentenceText, phonemizedText


class dataset(Dataset):
    def __init__(
        self,
        data_dir,
        debugging=False,
        has_label=True,
        add_noises=False,
        word_level=False,
        use_addtional_corpus=False,
        sp_noise_std=None,
        features_noise_std=None,
        gaussian_filter_sigma=0.8,
    ):
        self.debugging = debugging
        self.add_noises = add_noises
        self.word_level = word_level
        self.has_label = has_label
        self.use_addtional_corpus = use_addtional_corpus

        self.gaussian_filter_sigma = gaussian_filter_sigma

        self.sp_noise_std = sp_noise_std

        self.features_noise_std = features_noise_std

        if data_dir is None:
            return

        if use_addtional_corpus:
            # text extract from the people_speech dataset
            # force the model to learn grammar
            # not working yet
            self.addtional_corpus_path = os.path.join(data_dir, "people_speech.npy")

        # self.spikePows = np.load(os.path.join(data_dir, "spikePows.npy"),mmap_mode="r")
        # reconstruct the np.mmap every itteration is way more memory efficient
        self.spikePows_path = os.path.join(data_dir, "spikePows.npy")

        # start and end indices of spikePow for each sentence
        self.spikePow_indices = np.load(os.path.join(data_dir, "spikePow_indices.npy"))

        self.spikePow_mean_stds = np.load(
            os.path.join(data_dir, "spikePow_mean_stds.npy")
        )

        if word_level:
            # start and end indices of each word in spikepow
            with open(os.path.join(data_dir, "word_sp_indices.npy"), "rb") as handle:
                self.word_sp_indices = pickle.load(handle)

        if has_label:
            self.sentenceTexts = np.load(os.path.join(data_dir, "sentenceTexts.npy"))
            self.phonemizedTexts = np.load(
                os.path.join(data_dir, "phonemizedTexts.npy")
            )

        if debugging:
            # only use the first 500 entries for debugging
            self.spikePow_indices = self.spikePow_indices[:500]
            if word_level:
                self.word_sp_indices = self.word_sp_indices[:1000]

    def __len__(self):
        return len(self.spikePow_indices)

    def get_addtional_courpus(self):
        texts = np.load(self.addtional_corpus_path, mmap_mode="r")
        idx = np.random.randint(len(texts))

        text, ph_text = texts[idx]

        ph_text = ph_text.replace(" ", "|")
        text = text.replace(" ", "|")

        # generate a string that look similar to the output
        # of greedy ctc decoder with a lot of noises

        eos_id = len(vocab) - 1
        label = tokenize(text) + [eos_id]

        ph_tokenized = phonetic_tokenize(ph_text)

        repeats = np.ones(len(ph_tokenized), dtype="int")

        while len(label) * 1.5 > repeats.sum():
            # repeats[np.random.rand(len(repeats)) < 0.1] += 1
            repeats += 1

        input_ids = np.repeat(ph_tokenized, repeats)

        # repalce with blank token
        # temp[np.random.rand(len(temp)) < 0.05] = 0

        # # replace with random token
        # t = np.random.rand(len(temp)) < 0.03
        # temp[t] = np.random.randint(len(phoneme_vocab) - 1, size=sum(t))

        input_ids = np.pad(input_ids, (np.random.randint(15), 0))

        assert len(input_ids) > len(label)

        return input_ids, label

    def mask_word(self, idx, spikePow, spikePow_mask, sentenceText, phonemizedText):
        assert self.word_level

        words_indices = self.word_sp_indices[idx]

        sent_words = sentenceText.split("|")
        ph_words = phonemizedText.split("|")

        mp = np.random.rand(len(words_indices))

        mp[np.random.randint(len(mp))] = 1

        mask = mp < 0.2

        sentenceText = []
        phonemizedText = []

        for m, sw, phw, (_start, _end) in zip(
            mask, sent_words, ph_words, words_indices
        ):
            if not m:
                sentenceText.append(sw)
                phonemizedText.append(phw)
                continue

            spikePow_mask[_start:_end] = 1

        sentenceText = "|".join(sentenceText)
        phonemizedText = "|".join(phonemizedText)

        return spikePow, spikePow_mask, sentenceText, phonemizedText

    def __getitem__(self, idx):

        spikePows = np.load(self.spikePows_path, mmap_mode="r")
        _start, _end = self.spikePow_indices[idx]
        spikePow = spikePows[_start:_end].copy()
        spikePows = None

        block_mean, block_std = self.spikePow_mean_stds[idx]

        sentenceText = None
        phonemizedText = None

        if self.has_label:
            sentenceText = self.sentenceTexts[idx].replace(" ", "|")
            phonemizedText = self.phonemizedTexts[idx].replace(" ", "|")

        # spikePow = filter_noises(spikePow, block_mean, block_std, ep=1e-8)

        if self.add_noises and self.sp_noise_std and np.random.rand() < 0.75:
            noise = np.random.normal(
                loc=1, scale=self.sp_noise_std, size=spikePow.shape
            ).astype("float32")
            spikePow = spikePow * noise

        # block normalization
        spikePow = (spikePow - block_mean) / block_std

        if self.add_noises and self.features_noise_std and np.random.rand() < 0.75:
            noise = np.random.normal(
                loc=0, scale=self.features_noise_std, size=256
            ).astype("float32")

            spikePow += noise

        # smoothing
        sigma = 0.8
        # if self.add_noises:
        #     sigma = np.random.normal(loc=sigma, scale=0.1)

        if self.gaussian_filter_sigma is not None and self.gaussian_filter_sigma != 0:
            spikePow = scipy.ndimage.gaussian_filter1d(
                spikePow, self.gaussian_filter_sigma, axis=0
            )
        # spikePow = scipy.signal.savgol_filter(spikePow, 20, 2, axis=0)


        # if self.add_noises:
        #     # add more noises
        #     ratio = np.random.normal(loc=1, scale=0.1)

        #     spikePow = np.hstack([
        #         samplerate.resample(spikePow[:, 0:128], ratio),
        #         samplerate.resample(spikePow[:, 128:256], ratio)
        #     ])

        spikePow_mask = np.zeros(len(spikePow), dtype=int)

        if self.add_noises and self.word_level:
            spikePow, spikePow_mask, sentenceText, phonemizedText = self.mask_word(
                idx, spikePow, spikePow_mask, sentenceText, phonemizedText
            )

        # if self.add_noises and np.random.rand() < 0.25:
        #     # blank out quater of the input channels randomly
        #     _start = np.random.randint(3) * 64
        #     _end = _start + 64
        #     spikePow[:, _start:_end] = 0

        # shift spikePow randomly
        if self.add_noises:
            _start = np.random.randint(8)
            spikePow = spikePow[_start:]
            spikePow_mask = spikePow_mask[_start:]

        re = {
            "spikePow": spikePow,
            "spikePow_mask": spikePow_mask,
            "spikePow_lens": len(spikePow),
        }

        if not self.has_label:
            return re

        tokenized = tokenize(sentenceText)

        eos_id = len(vocab) - 1

        tokenized = [1] + tokenized + [1, eos_id]
        # tokenized = tokenized + [eos_id]

        re["sent"] = sentenceText

        re["sent_ids"] = tokenized
        re["sent_ids_len"] = len(tokenized)

        ph = phonetic_tokenize(phonemizedText)

        ph_eos_id = len(phoneme_vocab) - 1

        ph = [1] + ph + [1, ph_eos_id]
        # ph = ph + [ph_eos_id]

        re["phonemized"] = phonemizedText
        re["phonemize_ids"] = ph
        re["phonemize_ids_len"] = len(ph)

        if not self.use_addtional_corpus:
            return re

        ps_input_ids, ps_label = self.get_addtional_courpus()

        re["ps_input_ids"] = ps_input_ids
        re["ps_input_ids_lens"] = len(ps_input_ids)
        re["ps_label"] = ps_label
        re["ps_label_lens"] = len(ps_label)

        return re


def collate_fn_factory(add_noises=False):

    batch_fields = [
        "spikePow",
        "spikePow_mask",
        "spikePow_lens",
        "sent_ids",
        "sent_ids_len",
        "phonemize_ids",
        "phonemize_ids_len",
        "sent",
        "phonemized",
        "ps_input_ids",
        "ps_input_ids_lens",
        "ps_label",
        "ps_label_lens",
    ]

    fields_to_pad = ["sent_ids", "phonemize_ids", "ps_input_ids", "ps_label"]

    tensor_fields = [
        "spikePow",
        "spikePow_mask",
        "spikePow_lens",
        "sent_ids",
        "sent_ids_len",
        "phonemize_ids",
        "phonemize_ids_len",
        "ps_input_ids",
        "ps_input_ids_lens",
        "ps_label",
        "ps_label_lens",
    ]
    # only scalar is allowed
    # ignore_index=-100
    pad_values = [-100, -100, 0, -100]

    def _pad(arrs, constant_values=0, pad_width_fn=lambda l: ((0, l))):
        target_length = max([len(i) for i in arrs])

        return np.array(
            [
                np.pad(
                    i,
                    pad_width_fn(target_length - len(i)),
                    "constant",
                    constant_values=constant_values,
                )
                for i in arrs
            ]
        )

    def collate_fn(items):
        batch = {}

        for f in batch_fields:
            if f in items[0].keys():
                batch[f] = [i[f] for i in items]

        for f, v in zip(fields_to_pad, pad_values):
            if f in batch.keys():
                batch[f] = _pad(batch[f], constant_values=v)

        # concatenate spikePow
        # 1, seq_len, 256
        batch["spikePow"] = np.expand_dims(np.concatenate(batch["spikePow"]), axis=0)
        batch["spikePow_mask"] = np.concatenate(batch["spikePow_mask"])
        
        # target_length = max([len(i) for i in batch["spikePow"]])
        # for i in range(len(batch["spikePow"])):
        #     additional = target_length - len(batch["spikePow"][i])

        #     batch["spikePow"][i] = np.pad(
        #         batch["spikePow"][i],
        #         ((0, additional), (0, 0)),
        #         "constant",
        #         constant_values=0,
        #     )
        #     batch["spikePow_mask"][i] = np.pad(
        #         batch["spikePow_mask"][i], (0, additional), constant_values=0
        #     )

        for f in tensor_fields:
            if f in batch.keys():
                batch[f] = torch.tensor(batch[f])

        return batch

    return collate_fn


class B2T_DataModule(L.LightningDataModule):
    def __init__(
        self,
        add_noises=False,
        train_data_dir="./dataset/train",
        val_data_dir="./dataset/valid",
        test_data_dir="./dataset/test",
        word_level=False,
        use_addtional_corpus=False,
        sp_noise_std=None,
        features_noise_std=None,
        gaussian_filter_sigma=0.8,
        debugging=False,
        train_batch_size=4,
        valid_batch_size=4,
        num_workers=4,
        **other_args
    ):
        super().__init__()

        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir

        self.word_level = word_level

        self.sp_noise_std = sp_noise_std
        self.gaussian_filter_sigma = gaussian_filter_sigma
        self.debugging = debugging

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

        self.use_addtional_corpus = use_addtional_corpus

        self.add_noises = add_noises

        self.features_noise_std = features_noise_std

    def setup(self, stage: str):

        self.train_dataset = dataset(
            data_dir=self.train_data_dir,
            debugging=self.debugging,
            sp_noise_std=self.sp_noise_std,
            word_level=self.word_level,
            add_noises=self.add_noises,
            has_label=True,
            use_addtional_corpus=self.use_addtional_corpus,
            gaussian_filter_sigma=self.gaussian_filter_sigma,
            features_noise_std=self.features_noise_std
        )

        self.val_dataset = dataset(
            data_dir=self.val_data_dir,
            has_label=True,
            debugging=self.debugging,
            word_level=False,
            add_noises=False,
            gaussian_filter_sigma=self.gaussian_filter_sigma,
        )

        self.test_dataset = dataset(
            data_dir=self.test_data_dir,
            has_label=False,
            debugging=self.debugging,
            word_level=False,
            add_noises=False,
            gaussian_filter_sigma=self.gaussian_filter_sigma,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=collate_fn_factory(add_noises=True),
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(),
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(),
            shuffle=False,
            num_workers=self.num_workers,
        )
