import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io as sio
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


class B2T_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        has_labels=True,
        phonemize_target=False,
        debugging=False,
        input_length_multiplier=4,
    ):

        self.has_labels = has_labels
        self.phonemize_target = phonemize_target
        self.input_length_multiplier = input_length_multiplier

        data_file_paths = [
            os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
        ]

        if debugging:
            # only load 1 file for debugging
            data_file_paths = data_file_paths[:1]

        entries = [e for fpath in data_file_paths for e in self._load_data(fpath)]

        self.entries = self._preprocess_data(entries)

    def _load_data(self, data_files_path):
        data = sio.loadmat(data_files_path)

        entries = []

        for i in range(len(data["spikePow"][0])):
            entry = {
                "input": unscrambleChans(data["spikePow"][0][i]),
            }

            if self.has_labels:
                entry["sent"] = clean_text(data["sentenceText"][i])

            entries.append(entry)

        return entries

    def _preprocess_data(self, entries):

        if self.phonemize_target and self.has_labels:
            sentences = [e["sent"] for e in entries]

            phs = phonemize_text(sentences)

            for i in range(len(entries)):
                entries[i]["phonemized"] = phs[i]

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        _input = item["input"]

        # pad _input to a multiple of input_length_multiplier
        if len(_input) % self.input_length_multiplier != 0:
            pad_width = (
                self.input_length_multiplier
                - len(_input) % self.input_length_multiplier
            )

            _start = np.random.randint(pad_width + 1)

            _end = pad_width - _start

            _input = np.pad(_input, ((_start, _end), (0, 0)),  "constant", constant_values=0)

        re = {
            "input": _input,
            "input_len": len(_input),
        }

        if not self.has_labels:
            return re

        tokenized = tokenize(item["sent"])

        re["sent"] = item["sent"]

        re["sent_ids"] = tokenized
        re["sent_ids_len"] = len(tokenized)

        if not self.phonemize_target:
            return re

        ph = phonetic_tokenize(item["phonemized"])

        re["phonemized"] = item["phonemized"]
        re["phonemize_ids"] = ph
        re["phonemize_ids_len"] = len(ph)

        # TODO: pad _input to atleast input_length_multiplier * phonemize_ids_len
        # pad _input to a multiple of input_length_multiplier

        if len(ph) * self.input_length_multiplier > len(_input):
            additional_input_length = len(ph) * self.input_length_multiplier - len(
                _input
            )

            additional_input_length = (
                additional_input_length // self.input_length_multiplier + 1
            ) * self.input_length_multiplier
            
            _start = np.random.randint(additional_input_length + 1)
            _end = additional_input_length - _start

            _input = np.pad(_input, ((_start, _end), (0, 0)), "constant", constant_values=0)

            re["input"] = _input
            re["input_len"] = len(_input)

        assert len(ph) * self.input_length_multiplier <= len(_input)

        return re


def collate_fn_factory():

    batch_fields = [
        "input",
        "input_len",
        "sent_ids",
        "sent_ids_len",
        "phonemize_ids",
        "phonemize_ids_len",
    ]

    fields_to_pad = ["input", "sent_ids", "phonemize_ids"]

    # only scalar is allowed
    # ignore_index=-100
    pad_values = [0, -100, -100]

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
                if f == "input":
                    # TODO: use 'edge' pad mode
                    batch[f] = _pad(
                        batch[f],
                        constant_values=v,
                        pad_width_fn=lambda l: ((0, l), (0, 0)),
                    )
                else:
                    batch[f] = _pad(batch[f], constant_values=v)

        for f in batch_fields:
            if f in batch.keys():
                batch[f] = torch.tensor(batch[f])

        if "sent" in items[0].keys():
            batch["sent"] = [i["sent"] for i in items]

        if "phonemized" in items[0].keys():
            batch["phonemized"] = [i["phonemized"] for i in items]

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
        )

        self.val_dataset = B2T_Dataset(
            data_dir=self.config.val_data_dir,
            has_labels=True,
            phonemize_target=self.config.phonemize_target,
            input_length_multiplier=self.config.input_length_multiplier,
            debugging=self.debugging,
        )

        self.test_dataset = B2T_Dataset(
            data_dir=self.config.test_data_dir,
            has_labels=False,
            phonemize_target=False,
            input_length_multiplier=self.config.input_length_multiplier,
            debugging=self.debugging,
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
