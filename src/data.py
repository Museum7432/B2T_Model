import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io as sio
import scipy
import os
from transformers import AutoTokenizer
import numpy as np
import torch

from .utils import (
    unscrambleChans,
    phonemize_text,
    phonetic_tokenize,
    clean_text,
    tokenize,
)


def add_noises_to_input(spikePow, spikePow_mask):
    if np.random.rand() < 0.025:
        return spikePow, spikePow_mask

    # assume spikePow has mean=0 and std=1 after block normalization
    seq_len, channel_dim = spikePow.shape

    new_mean = np.random.normal(loc=0.0, scale=0.35, size=channel_dim).astype(
        spikePow.dtype
    )
    new_std = np.random.normal(loc=1.0, scale=0.35, size=channel_dim).astype(
        spikePow.dtype
    )

    base = spikePow * new_std + new_mean

    # now add more noises into part of the array

    if np.random.rand() < 0.025:
        return base, spikePow_mask

    for _ in range(6):
        selected = np.random.rand(seq_len) < 0.5

        if selected.sum() == 0:
            continue

        new_mean = np.random.normal(loc=0.0, scale=0.35, size=channel_dim).astype(
            spikePow.dtype
        )
        new_std = np.random.normal(loc=1.0, scale=0.35, size=channel_dim).astype(
            spikePow.dtype
        )

        # base[selected] = spikePow[selected] * new_std + new_mean
        base[selected] = base[selected] * new_std + new_mean


    # and finally mask section of the input

    if np.random.rand() < 0.05:
        return base, spikePow_mask

    for _ in range(np.random.randint(4) + 1):
        mask_length = np.random.randint(16) + 4

        assert seq_len > mask_length

        _start = np.random.randint(seq_len - mask_length)
        _end = _start + mask_length

        base[_start:_end] = 0

        spikePow_mask[_start:_end] = 2

    return base, spikePow_mask


def load_file(path, has_labels=True):
    data = sio.loadmat(path)
    # sentenceText spikePow blockIdx

    block_ids = list(set(data["blockIdx"].squeeze()))

    spikePows = []
    sentenceTexts = []

    for b_idx in block_ids:
        selected_ids = data["blockIdx"].squeeze() == b_idx

        spikePow_block = data["spikePow"][0][selected_ids]

        spikePow_block_lens = [len(a) for a in spikePow_block][:-1]

        spikePow_block_start_indices = np.cumsum(spikePow_block_lens)

        # block normalization
        spikePow_block = np.vstack(spikePow_block)

        spikePow_block = scipy.stats.zscore(spikePow_block)

        spikePow_block = np.split(
            spikePow_block, indices_or_sections=spikePow_block_start_indices
        )

        spikePows += spikePow_block

        if has_labels:
            sentenceText_block = data["sentenceText"][selected_ids]
            sentenceTexts += [clean_text(s) for s in sentenceText_block]

    data = None
    return spikePows, sentenceTexts


def load_files(paths, has_labels=True):

    spikePows = []
    sentenceTexts = []

    for f in paths:
        sp, st = load_file(path=f, has_labels=has_labels)

        spikePows += sp

        if has_labels:
            sentenceTexts += st

    return spikePows, sentenceTexts


class B2T_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        has_labels=True,
        phonemize_target=False,
        debugging=False,
        input_length_multiplier=4,
        add_noises=False,
    ):

        self.has_labels = has_labels
        self.phonemize_target = phonemize_target
        self.input_length_multiplier = input_length_multiplier

        self.add_noises = add_noises

        data_file_paths = [
            os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
        ]

        if debugging:
            # only load 1 file for debugging
            data_file_paths = data_file_paths[:1]

        spikePows, sentenceTexts = load_files(
            paths=data_file_paths, has_labels=has_labels
        )

        spikePows = [unscrambleChans(sp) for sp in spikePows]

        self.spikePows = spikePows

        self.sentenceTexts = sentenceTexts

        if self.phonemize_target and self.has_labels:
            self.phonemized = phonemize_text(sentenceTexts)

    def __len__(self):
        return len(self.spikePows)

    def __getitem__(self, idx):

        spikePow = self.spikePows[idx]

        # 0: padding
        # 1: input
        # 2: masked
        spikePow_mask = np.ones(len(spikePow)).astype(int)

        # TODO: introduce noises into spikePow
        if self.add_noises:
            spikePow, spikePow_mask = add_noises_to_input(spikePow, spikePow_mask)

        # cut bits at the start and end of spikePow to a multiple of input_length_multiplier
        if len(spikePow) % self.input_length_multiplier != 0:

            remainder = len(spikePow) % self.input_length_multiplier

            _start = np.random.randint(remainder + 1)

            _end = len(spikePow) - (remainder - _start)

            spikePow = spikePow[_start:_end]

            spikePow_mask = spikePow_mask[_start:_end]

            # pad_width = (
            #     self.input_length_multiplier
            #     - len(spikePow) % self.input_length_multiplier
            # )

            # _start = np.random.randint(pad_width + 1)

            # _end = pad_width - _start

            # spikePow = np.pad(
            #     spikePow, ((_start, _end), (0, 0)), "constant", constant_values=0
            # )

        assert len(spikePow) % self.input_length_multiplier == 0

        re = {
            "spikePow": spikePow,
            "spikePow_len": len(spikePow),
            "spikePow_mask": spikePow_mask,
        }

        if not self.has_labels:
            return re

        sent = self.sentenceTexts[idx]

        tokenized = tokenize(sent)

        re["sent"] = sent

        re["sent_ids"] = tokenized
        re["sent_ids_len"] = len(tokenized)

        if not self.phonemize_target:
            return re

        phonemized = self.phonemized[idx]

        ph = phonetic_tokenize(phonemized)

        re["phonemized"] = phonemized
        re["phonemize_ids"] = ph
        re["phonemize_ids_len"] = len(ph)

        # TODO: pad spikePow to atleast input_length_multiplier * phonemize_ids_len
        # pad spikePow to a multiple of input_length_multiplier

        # if len(ph) * self.input_length_multiplier > len(spikePow):
        #     additional_input_length = len(ph) * self.input_length_multiplier - len(
        #         spikePow
        #     )

        #     additional_input_length = (
        #         additional_input_length // self.input_length_multiplier + 1
        #     ) * self.input_length_multiplier

        #     _start = np.random.randint(additional_input_length + 1)
        #     _end = additional_input_length - _start

        #     spikePow = np.pad(
        #         spikePow, ((_start, _end), (0, 0)), "constant", constant_values=0
        #     )

        #     re["spikePow"] = spikePow
        #     re["spikePow_len"] = len(spikePow)

        # assert len(ph) * self.input_length_multiplier <= len(spikePow)

        return re


def collate_fn_factory():

    batch_fields = [
        "spikePow",
        "spikePow_len",
        "spikePow_mask",
        "sent_ids",
        "sent_ids_len",
        "phonemize_ids",
        "phonemize_ids_len",
        "sent",
        "phonemized",
    ]

    fields_to_pad = ["spikePow", "spikePow_mask", "sent_ids", "phonemize_ids"]

    tensor_fields = [
        "spikePow",
        "spikePow_len",
        "sent_ids",
        "sent_ids_len",
        "phonemize_ids",
        "phonemize_ids_len",
        "spikePow_mask",
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
                if f == "spikePow":
                    # TODO: use 'edge' pad mode
                    batch[f] = _pad(
                        batch[f],
                        constant_values=v,
                        pad_width_fn=lambda l: ((0, l), (0, 0)),
                    )
                else:
                    batch[f] = _pad(batch[f], constant_values=v)

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
            phonemize_target=self.config.phonemize_target,
            input_length_multiplier=self.config.input_length_multiplier,
            debugging=self.debugging,
            add_noises=True,
        )

        self.val_dataset = B2T_Dataset(
            data_dir=self.config.val_data_dir,
            has_labels=True,
            phonemize_target=self.config.phonemize_target,
            input_length_multiplier=self.config.input_length_multiplier,
            debugging=self.debugging,
            add_noises=False,
        )

        self.test_dataset = B2T_Dataset(
            data_dir=self.config.test_data_dir,
            has_labels=False,
            phonemize_target=False,
            input_length_multiplier=self.config.input_length_multiplier,
            debugging=self.debugging,
            add_noises=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=collate_fn_factory(),
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
