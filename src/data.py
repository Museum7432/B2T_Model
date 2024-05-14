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

    low, high = get_bound(_mean=block_mean, _std=block_std, ep=1e-8)
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
    ):
        self.debugging = debugging
        self.add_noises = add_noises
        self.word_level = word_level
        self.has_label = has_label

        if data_dir is None:
            return

        # self.spikePows = np.load(os.path.join(data_dir, "spikePows.npy"),mmap_mode="r")
        # reconstruct the np.mmap every itteration is way more memory efficient
        self.spikePows_path = os.path.join(data_dir, "spikePows.npy")

        # start and end indices of spikePow for each sentence
        self.spikePow_indices = np.load(os.path.join(data_dir, "spikePow_indices.npy"))

        self.spikePow_mean_stds = np.load(
            os.path.join(data_dir, "spikePow_mean_stds.npy")
        )

        if word_level:
            # start and end indices of spikePow for each word
            self.word_sp_indices = np.load(
                os.path.join(data_dir, "word_sp_indices.npy")
            )
            # indices of the sentence each word belonged to
            self.word_sentence_id = np.load(os.path.join(data_dir, "sentence_id.npy"))

            self.words_list = np.load(os.path.join(data_dir, "words_list.npy"))
            self.ph_words_list = np.load(os.path.join(data_dir, "ph_words_list.npy"))

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
    
    def __getitem__(self, idx):

        spikePows = np.load(self.spikePows_path, mmap_mode="r")
        _start, _end = self.spikePow_indices[idx]
        spikePow = spikePows[_start:_end].copy()
        spikePows=None

        block_mean, block_std = self.spikePow_mean_stds[idx]

        sentenceText = None
        phonemizedText = None

        if self.has_label:
            sentenceText = self.sentenceTexts[idx].replace(" ", "|")
            phonemizedText = self.phonemizedTexts[idx].replace(" ", "|")


        # if self.add_noises:
        #     noise_ratio = np.random.rand()/40
        #     n_std = block_std * noise_ratio

        #     noise = np.random.normal(loc=0, scale=n_std, size=spikePow.shape).astype(
        #         "float32"
        #     )

        #     spikePow = spikePow + noise

        spikePow = filter_noises(spikePow, block_mean, block_std, ep=1e-8)
        
        if self.add_noises:
            noise = np.random.normal(loc=1, scale=0.01, size=spikePow.shape).astype(
                "float32"
            )

            spikePow = spikePow * noise

            noise2 = np.random.normal(loc=1, scale=0.05, size=spikePow.shape).astype(
                "float32"
            )

            block_mean = block_mean*noise2

        # block normalization
        spikePow = (spikePow - block_mean) / block_std



        # smoothing
        sigma = 1
        # if self.add_noises:
        #     sigma = np.random.normal(loc=sigma, scale=0.1)
        spikePow = scipy.ndimage.gaussian_filter1d(spikePow, sigma, axis=0)
        # spikePow = scipy.signal.savgol_filter(spikePow, 20, 2, axis=0)



        # if self.add_noises and np.random.rand() < 0.15:
        #     spikePow, sentenceText, phonemizedText = flip_(spikePow, sentenceText, phonemizedText)


        # if self.add_noises:
        #     # add more noises
        #     ratio = np.random.normal(loc=1, scale=0.1)

        #     spikePow = np.hstack([
        #         samplerate.resample(spikePow[:, 0:128], ratio),
        #         samplerate.resample(spikePow[:, 128:256], ratio)
        #     ])

        spikePow_mask = np.ones(len(spikePow), dtype=int)
        # shift spikePow randomly
        if self.add_noises:
            _start = np.random.randint(5)
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

        return re



    def _get(self, idx):

        if not self.word_level:
            spikePow, spikePow_mask, sentenceText, phonemizedText = (
                self.get_raw_sentence(idx)
            )
        else:
            spikePow, spikePow_mask, sentenceText, phonemizedText = (
                self.get_continuous_sentence(idx, masked_one=True)
            )
            # spikePow, spikePow_mask, sentenceText, phonemizedText = self.get_original_sentence(idx)
            # spikePow, spikePow_mask, sentenceText, phonemizedText = self.get_random_sent(idx)

        if self.add_noises:
            # nscale = np.random.uniform(low=0.005, high=0.05)

            noise = np.random.normal(loc=1, scale=0.01, size=spikePow.shape).astype(
                "float32"
            )
            # noise = np.random.uniform(low=-0.1, high=0.1, size=256).astype(
            #     "float32"
            # )
            # new_mean = np.random.normal(loc=0, scale=0.1, size=256).astype(
            #     spikePow.dtype
            # )
            # new_std = np.random.normal(loc=1, scale=0.1, size=256).astype(
            #     spikePow.dtype
            # )
            # spikePow = (spikePow*new_std + new_mean)*noise

            spikePow = spikePow * noise

        return spikePow, spikePow_mask, sentenceText, phonemizedText

    # def get_one_word(self, idx, masked=False):
    #     spikePows = np.load(self.spikePows_path, mmap_mode="r")
    #     _start, _end = self.word_sp_indices[idx]

    #     # spikePow has mean of 0 and std of 1
    #     spikePow = spikePows[_start:_end]

    #     # ratio = np.random.rand()/4 + 0.9
    #     # spikePow = np.hstack([
    #     #     samplerate.resample(spikePow[:, 0:128], ratio),
    #     #     samplerate.resample(spikePow[:, 128:256], ratio)
    #     # ])

    #     if masked:
    #         spikePow_mask = np.ones(len(spikePow), dtype=int) * 2
    #         # only mask half word
    #         mask_len = int((np.random.rand() / 2 + 0.25) * len(spikePow))

    #         if np.random.rand() < 0.5:
    #             spikePow_mask[:mask_len] = 1
    #         else:
    #             spikePow_mask[mask_len:] = 1
    #     else:
    #         spikePow_mask = np.ones(len(spikePow), dtype=int)

    #     word = self.words_list[idx]
    #     ph_word = self.ph_words_list[idx]

    #     return spikePow, spikePow_mask, word, ph_word

    # def get_sentence(self, seq_ids, masked=None):
    #     if masked is None:
    #         items = [self.get_one_word(i) for i in seq_ids]
    #     else:
    #         items = [self.get_one_word(i, m) for i, m in zip(seq_ids, masked)]

    #     spikePow = []
    #     spikePow_mask = []
    #     sentenceText = []
    #     phonemizedText = []

    #     for sp, spm, st, pt in items:
    #         spikePow.append(sp)
    #         spikePow_mask.append(spm)
    #         sentenceText.append(st)
    #         phonemizedText.append(pt)

    #     spikePow = np.vstack(spikePow)
    #     spikePow_mask = np.concatenate(spikePow_mask)
    #     sentenceText = "|".join(sentenceText)
    #     phonemizedText = "|".join(phonemizedText)

    #     return spikePow, spikePow_mask, sentenceText, phonemizedText

    # def get_random_sent(self, force_idx, num_words=None):

    #     seq_ids = [force_idx]

    #     if num_words == None:
    #         num_words = np.random.randint(10) + 1

    #     seq_ids = np.random.choice(len(self), size=num_words, replace=False)

    #     if force_idx not in seq_ids:
    #         seq_ids[0] = force_idx

    #     np.random.shuffle(seq_ids)

    #     return self.get_sentence(seq_ids)

    # def get_continuous_sentence(self, start_id, num_words=None, masked_one=False):
    #     if num_words is None:
    #         num_words = np.random.randint(10) + 1

    #     mask = None
    #     if masked_one and np.random.rand() < 0.8 and num_words > 4:
    #         # masked 1 word
    #         mask_id = np.random.randint(num_words)

    #         mask = np.zeros(num_words)
    #         mask[mask_id] = 1

    #     seq_ids = [i for i in range(start_id, start_id + num_words)]

    #     seq_ids = [i % len(self) for i in seq_ids]

    #     return self.get_sentence(seq_ids, mask)

    # def get_original_sentence(self, idx, masked_one=False):
    #     # idx is the index of the sentence
    #     idx = idx % len(self.spikePow_indices)

    #     (seq_ids,) = np.where(self.word_sentence_id == idx)

    #     num_words = len(seq_ids)

    #     mask = None
    #     if masked_one and np.random.rand() < 0.7:
    #         # masked 1 word
    #         mask_id = np.random.randint(num_words)

    #         mask = np.zeros(num_words)
    #         mask[mask_id] = 1

    #     spikePow, spikePow_mask, sentenceText, phonemizedText = self.get_sentence(
    #         seq_ids, mask
    #     )

    #     return spikePow, spikePow_mask, sentenceText, phonemizedText


    # def __getitem__(self, idx):
    #     spikePow, spikePow_mask, sentenceText, phonemizedText = self._get(idx)

    #     # if self.add_noises and np.random.rand() < 0.1:
    #     #     # blank out quater of the input channels randomly
    #     #     _start = np.random.randint(8) * 32
    #     #     _end = _start + 32
    #     #     spikePow[:, _start:_end] = 0

    #     #     # selected_channels = np.random.rand(256) < 0.2
    #     #     # spikePow[:, selected_channels] = 0

    #     # shift spikePow randomly
    #     if self.add_noises:
    #         _start = np.random.randint(5)
    #         spikePow = spikePow[_start:]
    #         spikePow_mask = spikePow_mask[_start:]

    #     re = {
    #         "spikePow": spikePow,
    #         "spikePow_mask": spikePow_mask,
    #         "spikePow_lens": len(spikePow),
    #     }

    #     if not self.has_label:
    #         return re

    #     sentenceText = sentenceText.replace(" ", "|")
    #     phonemizedText = phonemizedText.replace(" ", "|")

    #     tokenized = tokenize(sentenceText)

    #     eos_id = len(vocab) - 1

    #     # tokenized = [1] + tokenized + [1, eos_id]
    #     tokenized = tokenized + [eos_id]

    #     re["sent"] = sentenceText

    #     re["sent_ids"] = tokenized
    #     re["sent_ids_len"] = len(tokenized)

    #     ph = phonetic_tokenize(phonemizedText)

    #     ph_eos_id = len(phoneme_vocab) - 1

    #     # ph = [1] + ph + [1, ph_eos_id]
    #     ph = ph + [ph_eos_id]

    #     re["phonemized"] = phonemizedText
    #     re["phonemize_ids"] = ph
    #     re["phonemize_ids_len"] = len(ph)

    #     return re


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
                mode="edge"
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
