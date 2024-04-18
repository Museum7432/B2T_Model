import scipy.io as sio
import scipy
import os
import numpy as np

from src.utils import (
    phonemize_text,
    phonetic_tokenize,
    clean_text,
    tokenize,
)

def load_file(path, has_labels=True):
    data = sio.loadmat(path)
    # sentenceText spikePow blockIdx

    block_ids = list(set(data["blockIdx"].squeeze()))

    spikePows = []
    sentenceTexts = []

    for b_idx in block_ids:
        selected_ids = data["blockIdx"].squeeze() == b_idx

        spikePow_block = data["spikePow"][0][selected_ids]

        spikePow_block_lens = [len(a) for a in spikePow_block]

        spikePow_block_start_indices = np.cumsum(spikePow_block_lens[:-1])

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


def load_dir(dir_path, has_labels=True):

    data_file_paths = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))]

    spikePows = []
    sentenceTexts = []

    for f in data_file_paths:
        sp, st = load_file(path=f, has_labels=has_labels)

        spikePows += sp

        if has_labels:
            sentenceTexts += st

    return spikePows, sentenceTexts


def prepare_data(input_dir, output_dir, has_labels=True):
    spikePows, sentenceTexts = load_dir(dir_path=input_dir, has_labels=has_labels)

    spikePow_lens = [0] + [len(a) for a in spikePows]

    start_indices = np.cumsum(spikePow_lens[:-1])
    end_indices = np.cumsum(spikePow_lens[1:])

    spikePow_indices = np.vstack([start_indices, end_indices]).T

    spikePows = np.vstack(spikePows)

    # save spikePows
    np.save(os.path.join(output_dir, "spikePows.npy"), spikePows)

    np.save(os.path.join(output_dir, "spikePow_indices.npy"), spikePow_indices)

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

    has_labels = [
        True,
        True,
        False
    ]

    for d in output_dir:
        os.makedirs(d, exist_ok=True)

    for inp, out, hl in zip(input_dir, output_dir, has_labels):
        print(inp)
        prepare_data(inp, out, hl)

if __name__ == "__main__":
    main()
