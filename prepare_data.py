import scipy.io as sio
import scipy
import os
import numpy as np
from scipy.stats import norm
from argparse import ArgumentParser
from tqdm import tqdm

from src.utils import (
    phonemize_text,
    phonetic_tokenize,
    clean_text,
    tokenize,
    correct_channels,
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


def fix_text(text):

    text = text.lower().strip()

    replace_list = [
        [".", " "],
        ["-", " "],
        ["?", " "],
        [",", " "],
        ['"', " "],
        ["in' ", "ing "],
        ["in'.", "ing."],
        ["' ", " "],  # 8 instances is not enough
        ["'em", "them"],
        [" 'll", "'ll"],
        [" 'd", "'d"],
        [" '", " "],
        ["evidence'", "evidence"],  # unique case
        ["!", " "],
        ['"', " "],
        [":", " "],
        [";", " "],
        ["stirrin", "stirring"],
        [" hey've", "they've"],
        ["their's", "theirs"],
        ["cccountability", "accountability"],
        ["anythingt", "anything"],
        ["pricy", "pricey"],
        ["cucbers", "cucumbers"],
        ["premis", "premises"],
        [" aking ", " making "],
        [" aking ", " making "],
        [" ain't ", " aren't "]
    ]

    for fro, to in replace_list:
        text = text.replace(fro, to)

    text = " ".join(text.split())

    cl_text = clean_text(text)

    return cl_text


def load_file(path, has_labels=True):
    data = sio.loadmat(path)
    # sentenceText spikePow blockIdx

    block_ids = list(set(data["blockIdx"].squeeze()))

    spikePows = []
    sentenceTexts = []

    mean_stds = []

    for b_idx in block_ids:
        selected_ids = data["blockIdx"].squeeze() == b_idx

        spikePow_block = data["spikePow"][0][selected_ids]
        spikePow_block = [correct_channels(i) for i in spikePow_block]
        
        # spikePow_block = [np.sqrt(i) for i in spikePow_block]

        spikePow_block_lens = [len(a) for a in spikePow_block]

        spikePow_block_start_indices = np.cumsum(spikePow_block_lens[:-1])

        # block normalization
        features = np.vstack(spikePow_block)

        _mean = np.median(features, axis=0)
        _std = np.median(np.abs(features - _mean), axis=0)

        features = filter_noises(features, _mean, _std, ep=1e-8)

        block_mean_std = np.vstack([_mean, _std])

        block_mean_std = np.expand_dims(block_mean_std, 0)

        block_mean_std = np.broadcast_to(
            block_mean_std,
            (len(spikePow_block), block_mean_std.shape[1], len(features[0])),
        )

        spikePow_block = np.split(
            features, indices_or_sections=spikePow_block_start_indices
        )

        spikePows += spikePow_block

        mean_stds += [block_mean_std]

        if has_labels:
            sentenceText_block = data["sentenceText"][selected_ids]
            sentenceTexts += [fix_text(s) for s in sentenceText_block]
            # sentenceTexts += sentenceText_block.tolist()

    data = None
    return spikePows, sentenceTexts, mean_stds


def load_dir(dir_path, has_labels=True):

    data_file_paths = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))]

    spikePows = []
    sentenceTexts = []
    mean_stds = []

    for f in tqdm(data_file_paths):
        sp, st, ms = load_file(path=f, has_labels=has_labels)

        spikePows += sp
        mean_stds += ms

        if has_labels:
            sentenceTexts += st

    return spikePows, sentenceTexts, mean_stds


def prepare_data(input_dir, output_dir, has_labels=True):
    spikePows, sentenceTexts, mean_stds = load_dir(
        dir_path=input_dir, has_labels=has_labels
    )

    spikePow_lens = [0] + [len(a) for a in spikePows]

    start_indices = np.cumsum(spikePow_lens[:-1])
    end_indices = np.cumsum(spikePow_lens[1:])

    spikePow_indices = np.vstack([start_indices, end_indices]).T

    spikePows = np.vstack(spikePows)

    mean_stds = np.vstack(mean_stds)

    # save spikePows
    np.save(os.path.join(output_dir, "spikePows.npy"), spikePows)

    np.save(os.path.join(output_dir, "spikePow_indices.npy"), spikePow_indices)

    np.save(os.path.join(output_dir, "spikePow_mean_stds.npy"), mean_stds)

    if has_labels:
        phonemized = phonemize_text(sentenceTexts)

        sentenceTexts = np.array(sentenceTexts)
        phonemized = np.array(phonemized)

        np.save(os.path.join(output_dir, "sentenceTexts.npy"), sentenceTexts)
        np.save(os.path.join(output_dir, "phonemizedTexts.npy"), phonemized)


def main():
    input_dir = [
        "./dataset/competitionData/train",
        "./dataset/competitionData/test",
        "./dataset/competitionData/competitionHoldOut",
    ]
    output_dir = [
        "./dataset/train",
        "./dataset/valid",
        "./dataset/test",
    ]

    has_labels = [True, True, False]

    for d in output_dir:
        os.makedirs(d, exist_ok=True)

    for inp, out, hl in zip(input_dir, output_dir, has_labels):
        print(inp)
        prepare_data(inp, out, hl)


if __name__ == "__main__":
    main()
