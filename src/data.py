import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io as sio
import os
from transformers import AutoTokenizer


class B2T_Dataset(Dataset):
    def __init__(self, data_dir, tokenizer_name, has_labels=True, debugging=False):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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

    def _process_raw(self, data_files_path):
        data = sio.loadmat(data_files_path)

        entries = []

        for i in range(len(data["spikePow"][0])):

            entry = {
                "input": data["spikePow"][0][i],
            }

            if self.has_labels:
                entry["sents"] = data["sentenceText"][i].strip()

            entries.append(entry)

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        re = {
            "input": item["input"],
            "input_mask": [1] * len(item["input"]),
        }

        if self.has_labels:
            # TODO: tokenize sents
            re["labels"] = None
            re["labels_mask"] = None

        return re


def collate_fn_factory(pad_token_id):
    fields_to_pad = [
        "input",
        "input_mask",
        "labels",
        "labels_mask"
    ]

    pad_values = [
        [0] * 256,
        0,
        pad_token_id,
        0
    ]

    def _pad(arr, pad_value):
        target = max([len(i) for i in arr])
        return [i + [pad_value] * (target - len(i)) for i in arr]

    def collate_fn(items):
        batch = {}

        for f in items[0].keys():
            batch[f] = [i[f] for i in items]

        for f, v in zip(fields_to_pad, pad_values):
            if f in batch.keys():
                batch[f] = _pad(batch[f], v)

        for f in batch.keys():
            batch[f] = torch.tensor(batch[f])
        return batch

    return collate_fn


class B2T_DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir,
        val_data_dir,
        test_data_dir,
        tokenizer_name="google-t5/t5-base",
        train_batch_size: int = 2,
        valid_batch_size: int = 4,
        num_workers: int = 8,
        debugging=False
    ):
        super().__init__()

        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir

        
        self.tokenizer_name = tokenizer_name


        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size



        self.debugging = debugging


        self.num_workers = num_workers


    def setup(self, stage: str):

        self.train_dataset = SemevalDataset(
            data_dir=self.train_data_dir,
            tokenizer_name=self.tokenizer_name,
            has_labels=True,
            debugging=self.debugging
        )


        self.val_dataset = SemevalDataset(
            data_dir=self.val_data_dir,
            tokenizer_name=self.tokenizer_name,
            has_labels=True
            debugging=self.debugging
        )

        self.test_dataset = SemevalDataset(
            data_dir=self.test_data_dir,
            tokenizer_name=self.tokenizer_name,
            has_labels=False
            debugging=self.debugging
        )

        self.pad_token_id = self.train_dataset.tokenizer.pad_token_id
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            shuffle=False,
            num_workers=self.num_workers,
        )


        
