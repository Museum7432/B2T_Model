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

import gdown
import os


def download_files():
    dataset_dir = "./dataset"

    os.makedirs(dataset_dir, exist_ok=True)

    f_list = [
        ["test", "folder", "1v6lMnlrPsG0f71_FgsBhx712lLa695Qe"],
        ["train", "folder", "179Qx7YxLs1X1-uMR2Z5OhUeMIQrs0RsK"],
        ["valid", "folder", "1UaE9sCBn5xxJ4EiRrpnlcKC6_rIcf4Jp"],
    ]

    for dir_name, ftype, gd_id in f_list:
        output_path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(output_path) or os.path.isfile(output_path):
            continue

        if ftype == "folder":
            gdown.download_folder(id=gd_id, output=output_path)
        else:
            gdown.download(id=gd_id, output=output_path)


download_files()


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
        add_noises=False,
        word_level=False,
        has_label=True,
        use_people_speech=False,
    ):
        self.debugging = debugging
        self.add_noises = add_noises
        self.word_level = word_level
        self.has_label = has_label
        self.use_people_speech = use_people_speech

        if data_dir is None:
            return

        if use_people_speech:
            # text extract from the people_speech dataset
            # force the model to learn grammar
            self.people_speech_text_path = os.path.join(data_dir, "people_speech.npy")

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

    def get_people_speech(self):
        texts = np.load(self.people_speech_text_path, mmap_mode="r")
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

        spikePow = filter_noises(spikePow, block_mean, block_std, ep=1e-8)

        ############################
        noise = 0
        if self.add_noises:
            #     noise = np.random.normal(loc=1, scale=0.05, size=spikePow.shape).astype(
            #         "float32"
            #     )
            #     spikePow = spikePow * noise

            #     noise2 = np.random.normal(loc=1, scale=0.1, size=spikePow.shape).astype(
            #         "float32"
            #     )

            #     block_mean = block_mean*noise2

            # scale = block_std * 0.1
            # noise = np.random.normal(loc=0, scale=scale, size=spikePow.shape).astype(
            #     "float32"
            # )
            # spikePow = spikePow + noise

            noise_ratio = np.random.rand() / 2
            n_std = block_std * noise_ratio

            noise = np.random.normal(loc=0, scale=n_std, size=spikePow.shape).astype(
                "float32"
            )

        #     spikePow = spikePow + noise

        # if self.add_noises:
        #     n_std = block_std * 0.5
        #     noise = np.random.normal(loc=0, scale=n_std, size=spikePow.shape).astype(
        #         "float32"
        #     )
        ############################

        # block normalization
        spikePow = (spikePow - block_mean + noise) / block_std

        # smoothing
        sigma = 0.8
        # if self.add_noises:
        #     sigma = np.random.normal(loc=sigma, scale=0.1)
        spikePow = scipy.ndimage.gaussian_filter1d(spikePow, sigma, axis=0)
        # spikePow = scipy.signal.savgol_filter(spikePow, 20, 2, axis=0)

        # if self.add_noises:
        #     noise_ratio = np.random.rand()/80
        #     n_std = 1 * noise_ratio

        #     noise = np.random.normal(loc=0, scale=n_std, size=spikePow.shape).astype(
        #         "float32"
        #     )

        #     spikePow = spikePow + noise

        # if self.add_noises and np.random.rand() < 0.15:
        #     spikePow, sentenceText, phonemizedText = flip_(spikePow, sentenceText, phonemizedText)

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
        # shift spikePow randomly
        if self.add_noises:
            _start = np.random.randint(32)
            # spikePow = spikePow[_start:]
            # spikePow_mask = spikePow_mask[_start:]

            spikePow = np.pad(
                spikePow,
                ((_start, 0), (0, 0)),
                # mode="edge"
                constant_values=0,
            )
            spikePow_mask = np.pad(spikePow_mask, (_start, 0), constant_values=0)

        re = {
            "spikePow": spikePow,
            "spikePow_mask": spikePow_mask,
            "spikePow_lens": len(spikePow),
        }

        if not self.has_label:
            return re

        tokenized = tokenize(sentenceText)

        eos_id = len(vocab) - 1

        # tokenized = [1] + tokenized + [1, eos_id]
        tokenized = tokenized + [eos_id]

        re["sent"] = sentenceText

        re["sent_ids"] = tokenized
        re["sent_ids_len"] = len(tokenized)

        ph = phonetic_tokenize(phonemizedText)

        ph_eos_id = len(phoneme_vocab) - 1

        # ph = [1] + ph + [1, ph_eos_id]
        ph = ph + [ph_eos_id]

        re["phonemized"] = phonemizedText
        re["phonemize_ids"] = ph
        re["phonemize_ids_len"] = len(ph)

        if not self.use_people_speech:
            return re

        ps_input_ids, ps_label = self.get_people_speech()

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

        # pad spikePow and spikePow_mask
        target_length = max([len(i) for i in batch["spikePow"]])

        if add_noises:
            target_length += np.random.randint(10)

        for i in range(len(batch["spikePow"])):
            additional = target_length - len(batch["spikePow"][i])

            _start = 0

            # if not add_noises and additional > 0:
            #     _start = np.random.randint(additional)
            #     _start = _start % 10

            _end = additional - _start

            batch["spikePow"][i] = np.pad(
                batch["spikePow"][i],
                ((_start, _end), (0, 0)),
                # mode="edge"
                constant_values=0,
            )
            batch["spikePow_mask"][i] = np.pad(
                batch["spikePow_mask"][i], (_start, _end), constant_values=0
            )

            batch["spikePow_lens"][i] + _start

        batch["spikePow"] = np.array(batch["spikePow"])
        batch["spikePow_mask"] = np.array(batch["spikePow_mask"])

        for f in tensor_fields:
            if f in batch.keys():
                batch[f] = torch.tensor(batch[f])

        return batch

    return collate_fn


class B2T_DataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.get("debugging"):
            self.debugging = self.config.debugging
            print("debugging mode")
        else:
            self.debugging = False

    def setup(self, stage: str):

        if self.config.train_data_dir is not None:
            self.train_dataset = dataset(
                data_dir=self.config.train_data_dir,
                debugging=self.debugging,
                add_noises=True,
                word_level=self.config.word_level,
                has_label=True,
                use_people_speech=self.config.use_people_speech,
            )
        else:
            self.train_dataset = None

        if self.config.val_data_dir is not None:
            self.val_dataset = dataset(
                data_dir=self.config.val_data_dir,
                has_label=True,
                debugging=self.debugging,
                word_level=False,
                add_noises=False,
            )
        else:
            self.val_dataset = None

        if self.config.test_data_dir is not None:
            self.test_dataset = dataset(
                data_dir=self.config.test_data_dir,
                has_label=False,
                debugging=self.debugging,
                word_level=False,
                add_noises=False,
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=collate_fn_factory(add_noises=True),
            pin_memory=True,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.valid_batch_size,
            collate_fn=collate_fn_factory(),
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.valid_batch_size,
            collate_fn=collate_fn_factory(),
            shuffle=False,
            num_workers=self.config.num_workers,
        )
