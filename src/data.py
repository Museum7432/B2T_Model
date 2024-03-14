import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io as sio
import os
from transformers import AutoTokenizer
import numpy as np
import torch


class B2T_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer_name,
        block_size=16,
        has_labels=True,
        debugging=False,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, model_max_length=512
        )

        self.has_labels = has_labels

        data_file_paths = [
            os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
        ]

        if debugging:
            # only load 1 file for debugging
            data_file_paths = data_file_paths[:1]

        self.entries = [
            e for fpath in data_file_paths for e in self._process_raw(fpath)
        ]

        self.block_size = block_size

        input_channels = len(self.entries[0]["input"][0])

    def _process_raw(self, data_files_path):
        data = sio.loadmat(data_files_path)

        entries = []

        for i in range(len(data["spikePow"][0])):

            # TODO: tolist might take up too much time
            entry = {
                "input": data["spikePow"][0][i],
            }

            if self.has_labels:
                entry["sent"] = data["sentenceText"][i].strip()

            entries.append(entry)

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        _input = item["input"]

        # pad _input to a multiple of block_size
        if len(_input) % self.block_size != 0:
            pad_width = self.block_size - len(_input) % self.block_size
            _input = np.pad(
                _input, ((0, pad_width), (0, 0)), "constant", constant_values=0
            )

        re = {
            "input": _input,
            "input_block_attention_mask": [1] * (len(_input) // self.block_size),
        }

        if not self.has_labels:
            return re

        tokenized = self.tokenizer(item["sent"])

        re["labels"] = tokenized["input_ids"]
        re["labels_mask"] = tokenized["attention_mask"]
        re["sent"] = item["sent"]

        return re


def collate_fn_factory(label_padding, block_size):

    fields_to_pad = ["input", "input_block_attention_mask", "labels", "labels_mask"]

    # only scalar is allowed
    pad_values = [0, True, label_padding, 0]

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

        for f in fields_to_pad:
            if f in items[0].keys():
                batch[f] = [i[f] for i in items]

        for f, v in zip(fields_to_pad, pad_values):
            if f in batch.keys():
                if f == "input":
                    batch[f] = _pad(
                        batch[f],
                        constant_values=v,
                        pad_width_fn=lambda l: ((0, l), (0, 0)),
                    )
                else:
                    batch[f] = _pad(batch[f], constant_values=v)

        for f in fields_to_pad:
            if f in batch.keys():
                batch[f] = torch.from_numpy(batch[f])

        batch["block_size"] = block_size

        if "sent" in items[0].keys():
            batch["sent"] = [i["sent"] for i in items]

        return batch

    return collate_fn


class B2T_DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir,
        val_data_dir,
        test_data_dir,
        block_size=16,
        tokenizer_name="google-t5/t5-base",
        train_batch_size: int = 2,
        valid_batch_size: int = 4,
        num_workers: int = 8,
        debugging=False,
    ):
        super().__init__()

        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir

        self.tokenizer_name = tokenizer_name

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        self.num_workers = num_workers

        self.debugging = debugging

        self.block_size = block_size

    def setup(self, stage: str):

        self.train_dataset = B2T_Dataset(
            data_dir=self.train_data_dir,
            tokenizer_name=self.tokenizer_name,
            block_size=self.block_size,
            has_labels=True,
            debugging=self.debugging,
        )

        self.val_dataset = B2T_Dataset(
            data_dir=self.val_data_dir,
            tokenizer_name=self.tokenizer_name,
            block_size=self.block_size,
            has_labels=True,
            debugging=self.debugging,
        )

        self.test_dataset = B2T_Dataset(
            data_dir=self.test_data_dir,
            tokenizer_name=self.tokenizer_name,
            block_size=self.block_size,
            has_labels=False,
            debugging=self.debugging,
        )

        # only tested with T5
        self.label_padding = -100

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=collate_fn_factory(
                self.label_padding, self.block_size
            ),
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(
                self.label_padding, self.block_size
            ),
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(
                self.label_padding, self.block_size
            ),
            shuffle=False,
            num_workers=self.num_workers,
        )
