import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io as sio
import scipy
import os
import numpy as np
import torch
import random

from .utils import (
    phonemize_text,
    phonetic_tokenize,
    clean_text,
    tokenize,
)


# https://stackoverflow.com/questions/3589214/generate-random-numbers-summing-to-a-predefined-value
def constrained_sum_sample_pos(n, total, low=0):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(
        np.random.choice(np.arange(1, total - (low - 1) * n), n - 1, replace=False)
    )

    return [
        a - b + low - 1
        for a, b in zip(dividers + [total - (low - 1) * n], [0] + dividers)
    ]


def plit_sections(num_sections, size, low=0):
    lens = [0] + constrained_sum_sample_pos(num_sections, size, low)

    _starts = np.cumsum(lens[:-1])
    _ends = np.cumsum(lens[1:])

    return np.vstack([_starts, _ends]).T


def plit_spikePow(spikePow, spikePow_mask, total_pad):
    # use pad token to partition input into sections
    seq_len, channel_dim = spikePow.shape

    min_section_len = int(seq_len * 0.08)
    num_sections = np.random.randint(6) + 1

    section_ids = plit_sections(num_sections, seq_len, min_section_len)

    pad_lens = constrained_sum_sample_pos(num_sections + 1, total_pad)

    padded_sections = []
    padded_sections_mask = []

    for i in range(num_sections):
        pad_len = pad_lens[i + 1]

        s_start, s_end = section_ids[i]

        section = np.pad(
            spikePow[s_start:s_end],
            ((0, pad_len), (0, 0)),
            "constant",
            constant_values=0,
        )

        section_mask = np.pad(
            spikePow_mask[s_start:s_end], (0, pad_len), constant_values=0
        )

        padded_sections.append(section)
        padded_sections_mask.append(section_mask)

    spikePow = np.vstack(padded_sections)
    spikePow_mask = np.concatenate(padded_sections_mask)

    spikePow = np.pad(
        spikePow, ((pad_lens[0], 0), (0, 0)), "constant", constant_values=0
    )
    spikePow_mask = np.pad(spikePow_mask, (pad_lens[0], 0), constant_values=0)

    return spikePow, spikePow_mask


def add_noises_to_input(spikePow, spikePow_mask):
    seq_len, channel_dim = spikePow.shape

    if np.random.rand() < 0.025:
        return spikePow, spikePow_mask

    # assume spikePow has mean=0 and std=1 after block normalization
    min_section_len = int(seq_len * 0.06)
    num_sections = np.random.randint(6) + 1
    section_ids = plit_sections(num_sections, seq_len, min_section_len)

    sections = []
    for s, e in section_ids:
        new_mean = np.random.normal(loc=0.0, scale=0.35, size=channel_dim).astype(
            spikePow.dtype
        )
        new_std = np.random.normal(loc=1.0, scale=0.35, size=channel_dim).astype(
            spikePow.dtype
        )

        sections.append(
            spikePow[s:e] * new_std + new_mean
        )

    base = np.vstack(sections)

    return base, spikePow_mask

    # and finally mask section of the input

    if np.random.rand() < 0.05:
        return base, spikePow_mask

    # for _ in range(np.random.randint(2) + 1):

    mask_length = np.random.randint(64) + 4
    assert seq_len > mask_length

    _start = np.random.randint(seq_len - mask_length)
    _end = _start + mask_length

    base[_start:_end] = 0

    spikePow_mask[_start:_end] = 2

    return base, spikePow_mask


class B2T_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        has_labels=True,
        debugging=False,
        add_noises=False,
    ):

        self.has_labels = has_labels

        self.add_noises = add_noises

        # self.spikePows = np.load(os.path.join(data_dir, "spikePows.npy"),mmap_mode="r")
        # reconstruct the np.mmap every itteration is way more memory efficient
        self.spikePows_path = os.path.join(data_dir, "spikePows.npy")

        self.spikePow_indices = np.load(os.path.join(data_dir, "spikePow_indices.npy"))

        if self.has_labels:
            self.sentenceTexts = np.load(os.path.join(data_dir, "sentenceTexts.npy"))

            self.phonemizedTexts = np.load(
                os.path.join(data_dir, "phonemizedTexts.npy")
            )

        if debugging:
            # only use the first 500 entries for debugging
            self.spikePow_indices = self.spikePow_indices[:500]

    def __len__(self):
        return len(self.spikePow_indices)

    def _get(self, idx):
        spikePows = np.load(self.spikePows_path, mmap_mode="r")

        _start, _end = self.spikePow_indices[idx]
        spikePow = spikePows[_start:_end]

        if self.add_noises and np.random.rand() < 0.92 and self.has_labels:
            # concat spikePow with another one
            idx2 = np.random.randint(len(self.spikePow_indices))
        else:
            idx2 = None

        # idx2 = None

        if idx2 is not None:
            _start, _end = self.spikePow_indices[idx2]
            spikePow2 = spikePows[_start:_end]

            spikePow = np.vstack([spikePow, spikePow2])

        # 0: padding
        # 1: input
        # 2: masked
        spikePow_mask = np.ones(len(spikePow), dtype=int)

        if self.has_labels:
            sentenceText = self.sentenceTexts[idx]
            phonemizedText = self.phonemizedTexts[idx]
        else:
            sentenceText = None
            phonemizedText = None

        if idx2 is not None:
            sentenceText2 = self.sentenceTexts[idx2]
            # assume there is a silent before and after the text is spoken
            sentenceText = sentenceText + " " + sentenceText2

        if idx2 is not None:
            phonemizedText2 = self.phonemizedTexts[idx2]
            phonemizedText = phonemizedText + " " + phonemizedText2

        return spikePow, spikePow_mask, sentenceText, phonemizedText

    def __getitem__(self, idx):

        spikePow, spikePow_mask, sentenceText, phonemizedText = self._get(idx)

        # introduce noises into spikePow
        if self.add_noises:
            spikePow, spikePow_mask = add_noises_to_input(spikePow, spikePow_mask)

        re = {
            "spikePow": spikePow,
            "spikePow_mask": spikePow_mask,
        }

        if not self.has_labels:
            return re

        tokenized = tokenize(sentenceText)

        tokenized = [1] + tokenized + [1, 0]

        re["sent"] = sentenceText

        re["sent_ids"] = tokenized
        re["sent_ids_len"] = len(tokenized)

        ph = phonetic_tokenize(phonemizedText)

        ph = [1] + ph + [1, 0]

        re["phonemized"] = phonemizedText
        re["phonemize_ids"] = ph
        re["phonemize_ids_len"] = len(ph)

        return re


def collate_fn_factory(add_noises=False):

    batch_fields = [
        "spikePow",
        "spikePow_mask",
        "sent_ids",
        "sent_ids_len",
        "phonemize_ids",
        "phonemize_ids_len",
        "sent",
        "phonemized",
    ]

    fields_to_pad = ["sent_ids", "phonemize_ids"]

    tensor_fields = [
        "spikePow",
        "spikePow_mask",
        "sent_ids",
        "sent_ids_len",
        "phonemize_ids",
        "phonemize_ids_len",
    ]
    # only scalar is allowed
    # ignore_index=-100
    pad_values = [0, 0, -100, -100]

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
        # TODO: use variable block_size to reduce overfitting

        batch = {}

        for f in batch_fields:
            if f in items[0].keys():
                batch[f] = [i[f] for i in items]

        for f, v in zip(fields_to_pad, pad_values):
            if f in batch.keys():
                batch[f] = _pad(batch[f], constant_values=v)

        # pad spikePow and spikePow_mask
        # spikePow should always be padded
        target_length = max([len(i) for i in batch["spikePow"]]) + 20

        for i in range(len(batch["spikePow"])):
            additional = target_length - len(batch["spikePow"][i])

            if add_noises:
                spikePow, spikePow_mask = plit_spikePow(
                    batch["spikePow"][i], batch["spikePow_mask"][i], additional
                )
                batch["spikePow"][i] = spikePow
                batch["spikePow_mask"][i] = spikePow_mask

            else:
                _start = additional // 2
                _end = additional - _start

                batch["spikePow"][i] = np.pad(
                    batch["spikePow"][i],
                    ((_start, _end), (0, 0)),
                    "constant",
                    constant_values=0,
                )

                batch["spikePow_mask"][i] = np.pad(
                    batch["spikePow_mask"][i], (_start, _end), constant_values=0
                )

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

        self.train_dataset = B2T_Dataset(
            data_dir=self.config.train_data_dir,
            has_labels=True,
            debugging=self.debugging,
            add_noises=True,
        )

        self.val_dataset = B2T_Dataset(
            data_dir=self.config.val_data_dir,
            has_labels=True,
            debugging=self.debugging,
            add_noises=False,
        )

        self.test_dataset = B2T_Dataset(
            data_dir=self.config.test_data_dir,
            has_labels=False,
            debugging=self.debugging,
            add_noises=False,
        )

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
