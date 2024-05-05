import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import scipy.io as sio
from scipy import signal
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
    phoneme_vocab,
    vocab,
)

correct_pos = np.array(
    [
        [192, 193, 208, 216, 160, 165, 178, 185, 62, 51, 43, 35, 94, 87, 79, 78],
        [194, 195, 209, 217, 162, 167, 180, 184, 60, 53, 41, 33, 95, 86, 77, 76],
        [196, 197, 211, 218, 164, 170, 177, 189, 63, 54, 47, 44, 93, 84, 75, 74],
        [198, 199, 210, 219, 166, 174, 173, 187, 58, 55, 48, 40, 92, 85, 73, 72],
        [200, 201, 213, 220, 168, 176, 183, 186, 59, 45, 46, 38, 91, 82, 71, 70],
        [202, 203, 212, 221, 172, 175, 182, 191, 61, 49, 42, 36, 90, 83, 69, 68],
        [204, 205, 214, 223, 161, 169, 181, 188, 56, 52, 39, 34, 89, 81, 67, 66],
        [206, 207, 215, 222, 163, 171, 179, 190, 57, 50, 37, 32, 88, 80, 65, 64],
        [129, 144, 150, 158, 224, 232, 239, 255, 125, 126, 112, 103, 31, 28, 11, 8],
        [128, 142, 152, 145, 226, 233, 242, 241, 123, 124, 110, 102, 29, 26, 9, 5],
        [130, 135, 148, 149, 225, 234, 244, 243, 121, 122, 109, 101, 27, 19, 18, 4],
        [131, 138, 141, 151, 227, 235, 246, 245, 119, 120, 108, 100, 25, 15, 12, 6],
        [134, 140, 143, 153, 228, 236, 248, 247, 117, 118, 107, 99, 23, 13, 10, 3],
        [132, 146, 147, 155, 229, 237, 250, 249, 115, 116, 106, 97, 21, 20, 7, 2],
        [133, 137, 154, 157, 230, 238, 252, 251, 113, 114, 105, 98, 17, 24, 14, 0],
        [136, 139, 156, 159, 231, 240, 254, 253, 127, 111, 104, 96, 30, 22, 16, 1],
    ]
)
correct_pos = np.array(
    [correct_pos[:8, :8], correct_pos[8:, 8:], correct_pos[:8, 8:], correct_pos[8:, :8]]
).flatten()


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
    num_sections = np.random.randint(3) + 1

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

    spikePow = spikePow.copy()

    # # add more noises
    # new_len = seq_len + np.random.randint(3) - 1
    # spikePow = signal.resample(spikePow, new_len)
    # spikePow_mask = np.ones(new_len, dtype=int)
    # seq_len = new_len
    # # return spikePow, spikePow_mask

    if np.random.rand() < 0.025:
        return spikePow, spikePow_mask

    # assume spikePow has mean=0 and std=1 after block normalization
    new_mean = np.random.uniform(low=-0.5, high=0.5, size=channel_dim).astype(
        spikePow.dtype
    )

    new_std = np.random.uniform(low=0.7, high=1.3, size=channel_dim).astype(
        spikePow.dtype
    )

    spikePow = spikePow * new_std + new_mean

    # min_section_len = int(seq_len * 0.1)
    # num_sections = np.random.randint(3) + 1
    # section_ids = plit_sections(num_sections, seq_len, min_section_len)

    # sections = []
    # for s, e in section_ids:
    #     new_mean = np.random.normal(loc=0.0, scale=0.35, size=channel_dim).astype(
    #         spikePow.dtype
    #     )
    #     new_std = np.random.normal(loc=1.0, scale=0.35, size=channel_dim).astype(
    #         spikePow.dtype
    #     )

    #     sections.append(spikePow[s:e] * new_std + new_mean)

    # spikePow = np.vstack(sections)

    # return spikePow, spikePow_mask

    # and finally mask section of the input

    if np.random.rand() < 0.25:
        return spikePow, spikePow_mask

    # for _ in range(np.random.randint(2) + 1):

    mask_length = int(seq_len * (np.random.rand() / 8))
    assert seq_len > mask_length

    _start = np.random.randint(seq_len - mask_length)
    _end = _start + mask_length

    spikePow[_start:_end] = 0

    spikePow_mask[_start:_end] = 2

    return spikePow, spikePow_mask


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

        self.spikePow_mean_stds = np.load(
            os.path.join(data_dir, "spikePow_mean_stds.npy")
        )

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

    def _get_one(self, idx):
        spikePows = np.load(self.spikePows_path, mmap_mode="r")
        _start, _end = self.spikePow_indices[idx]

        # spikePow = spikePows[_start:_end][:, correct_pos]
        spikePow = spikePows[_start:_end]

        # if self.add_noises:
        # add more noises
        # new_len = len(spikePows) - 10
        # spikePow = signal.resample(spikePow, new_len)

        _mean, _std = self.spikePow_mean_stds[idx]

        noise = 0

        if self.add_noises:
            # noise1 = np.random.uniform(low=0.95, high=1.05, size=256).astype(
            #     spikePow.dtype
            # )
            # noise2 = np.random.uniform(low=0.95, high=1.05, size=256).astype(
            #     spikePow.dtype
            # )
            # _mean = _mean * noise1
            # _std = _std * noise2
            # spikePow = spikePow * noise1

            noise = np.random.normal(loc=0.0, scale=0.05, size=256).astype(
                spikePow.dtype
            )

        # block normalization
        spikePow = (spikePow - _mean) / _std + noise

        # filter noises
        # assume each input channel follows a gaussian distribution
        # with std=1 after block normalization

        high = 4
        low = -4
        spikePow = np.clip(spikePow, low, high)

        spikePow_mask = np.ones(len(spikePow), dtype=int)

        # gaussian blur
        sig = 0.8
        if self.add_noises:
            sig = np.random.uniform(low=sig, high=sig + 0.3)
        spikePow = scipy.ndimage.gaussian_filter1d(spikePow, sig, axis=0, radius=10)

        if self.has_labels:
            sentenceText = self.sentenceTexts[idx]
            phonemizedText = self.phonemizedTexts[idx]
        else:
            sentenceText = None
            phonemizedText = None

        # if self.add_noises and np.random.rand() < 0.1:
        #     spikePow = np.flip(spikePow, axis=0)

        #     if self.has_labels:
        #         sentenceText = sentenceText[::-1]

        #         # t = sentenceText.split()[::-1]
        #         # sentenceText = " ".join(t)

        #         # phonemizedText = phonemizedText[::-1]
        #         t = phonemizedText.replace(" ", " | ")
        #         t = t.replace("-", " - ")
        #         t = t.replace("+", " + ")
        #         t = t.split()[::-1]
        #         phonemizedText = "".join(t).replace("|", " ")

        return spikePow, spikePow_mask, sentenceText, phonemizedText

    def _get(self, idx):
        seq_ids = [idx]

        # if self.add_noises and self.has_labels:
        #     add_len = np.random.randint(4) + 1
        #     seq_ids = np.random.choice(
        #         len(self.spikePow_indices), size=add_len, replace=False
        #     )
        #     if idx not in seq_ids:
        #         seq_ids[0] = idx
        #     np.random.shuffle(seq_ids)

        items = [self._get_one(i) for i in seq_ids]

        spikePow = []
        spikePow_mask = []
        sentenceText = []
        phonemizedText = []

        for sp, spm, st, pt in items:
            spikePow.append(sp)
            spikePow_mask.append(spm)
            sentenceText.append(st)
            phonemizedText.append(pt)

        spikePow = np.vstack(spikePow)
        spikePow_mask = np.concatenate(spikePow_mask)

        if self.has_labels:
            # assume there is a silent before and after the text is spoken
            sentenceText = " ".join(sentenceText)
            phonemizedText = " ".join(phonemizedText)

        return spikePow, spikePow_mask, sentenceText, phonemizedText

    def __getitem__(self, idx):

        spikePow, spikePow_mask, sentenceText, phonemizedText = self._get(idx)

        # mask part of spikePow
        if self.add_noises and np.random.rand() < 0.5:
            seq_len, channel_dim = spikePow.shape
            mask_length = np.random.randint(4) + 4

            _start = np.random.randint(seq_len - mask_length)
            _end = _start + mask_length

            spikePow[_start:_end] = 0

            spikePow_mask[_start:_end] = 2

        # shift spikePow randomly
        if self.add_noises:
            _start = np.random.randint(5)
            spikePow = spikePow[_start:]
            spikePow_mask = spikePow_mask[_start:]

        if self.add_noises and np.random.rand() < 0.1:
            # blank out quater of the input channels randomly
            # _start = np.random.randint(3) * 64
            # _end = _start + 64
            # spikePow[:, _start:_end] = 0

            
            selected_channels = np.random.rand(256) < 0.2
            spikePow[:, selected_channels] = 0
        
        
        re = {
            "spikePow": spikePow,
            "spikePow_mask": spikePow_mask,
            "spikePow_lens": len(spikePow),
        }

        if not self.has_labels:
            return re

        sentenceText = sentenceText.replace(" ", "|")
        phonemizedText = phonemizedText.replace(" ", "|")

        tokenized = tokenize(sentenceText)

        eos_id = len(vocab) - 1

        # tokenized = [1] + tokenized + [1, eos_id]
        tokenized = tokenized + [eos_id]

        re["sent"] = sentenceText

        re["sent_ids"] = tokenized
        re["sent_ids_len"] = len(tokenized)

        ph = phonetic_tokenize(phonemizedText)

        # ph = [2] + ph + [2, 1]
        # 0: pad
        # 1: eos
        # ph = ph + [1]

        ph_eos_id = len(phoneme_vocab) - 1

        # ph = [1] + ph + [1, ph_eos_id]
        ph = ph + [ph_eos_id]

        re["phonemized"] = phonemizedText
        re["phonemize_ids"] = ph
        re["phonemize_ids_len"] = len(ph)

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
    ]

    fields_to_pad = ["sent_ids", "phonemize_ids"]

    tensor_fields = [
        "spikePow",
        "spikePow_mask",
        "spikePow_lens",
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
        batch = {}

        for f in batch_fields:
            if f in items[0].keys():
                batch[f] = [i[f] for i in items]

        for f, v in zip(fields_to_pad, pad_values):
            if f in batch.keys():
                batch[f] = _pad(batch[f], constant_values=v)

        # pad spikePow and spikePow_mask
        # spikePow should always be padded
        target_length = max([len(i) for i in batch["spikePow"]]) + 4
        if add_noises:
            target_length += np.random.randint(10)

        for i in range(len(batch["spikePow"])):
            additional = target_length - len(batch["spikePow"][i])

            batch["spikePow"][i] = np.pad(
                batch["spikePow"][i],
                ((0, additional), (0, 0)),
                "constant",
                constant_values=0,
            )
            batch["spikePow_mask"][i] = np.pad(
                batch["spikePow_mask"][i], (0, additional), constant_values=0
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
