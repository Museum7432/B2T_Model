import scipy.io as sio
import os
import numpy as np
import pandas as pd
# need to pip install requests
from phonemizer import phonemize
from phonemizer.separator import Separator
from tqdm import tqdm

from src.utils import phonemize_text


# wget -O ./dataset/peoples_speech.json  "https://huggingface.co/datasets/MLCommons/peoples_speech/resolve/main/train/clean.json?download=true"

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

def main():
    dataset = pd.read_json("./dataset/peoples_speech.json", lines=True)

    print("peoples_speech.json loaded")
    raw_text = []
    for index, r in dataset.iterrows():
        raw_text += r["training_data"]["label"]

    text = []
    for t in raw_text:
        w = t.split()

        n_len = np.random.randint(10) + 4
        
        if n_len >= len(w):
            text.append(t)
            continue

        _start = np.random.randint(len(w) - n_len)

        _end = _start + n_len

        text.append(
            " ".join(w[_start: _end])
        )


    sample_id = np.random.choice(range(len(text)), size=100000, replace=False)

    final_text = []
    for i in sample_id:
        final_text.append(text[i])
    
    ph_text = phonemize_text(final_text)

    text = [
        [final_text[i], ph_text[i]]
        for i in range(len(sample_id))
    ]
    
    text = np.array(text)

    np.save("./dataset/train/people_speech.npy", text)

    print("done")


if __name__ == "__main__":
    main()
