import numpy as np
from phonemizer import phonemize
from phonemizer.separator import Separator
import itertools


import gdown
import os
import wget
from jsonargparse import ArgumentParser


def download_files():
    dataset_dir = "./dataset"

    os.makedirs(dataset_dir, exist_ok=True)

    f_list = [
        ["test", "gfolder", "1v6lMnlrPsG0f71_FgsBhx712lLa695Qe"],
        ["train", "gfolder", "179Qx7YxLs1X1-uMR2Z5OhUeMIQrs0RsK"],
        ["valid", "gfolder", "1UaE9sCBn5xxJ4EiRrpnlcKC6_rIcf4Jp"],
        [
            "en_us_cmudict_forward.pt",
            "file",
            "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt",
        ],
    ]

    for dir_name, ftype, gd_id in f_list:
        output_path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(output_path) or os.path.isfile(output_path):
            continue

        if ftype == "gfolder":
            gdown.download_folder(id=gd_id, output=output_path)
        elif ftype == "gfile":
            gdown.download(id=gd_id, output=output_path)
        elif ftype == "file":
            wget.download(gd_id, output_path)


download_files()


# require "festival" to be installed

# el, em, en, pau not found in trainning dataset
# '-': blank token/pad token
# '|': silent token
# '_': eos token (not used for ctc)
phoneme_vocab = "-,|,aa,ae,ah,ao,aw,ax,ay,eh,el,em,en,er,ey,ih,iy,ow,oy,uh,uw,b,ch,d,dh,f,g,hh,jh,k,l,m,n,ng,p,r,s,sh,t,th,v,w,y,z,zh,pau,_".split(
    ","
)
ctc_phoneme_vocab = phoneme_vocab[:-1]

whitelist_char = "nuclear oktsdyifwhbvxpm'gqjz"


def clean_text(text):
    text = text.strip().lower()
    return "".join([c for c in text if c in whitelist_char])


def phonemize_text(texts):
    texts = [clean_text(s) for s in texts]
    # nuclear rockets
    # n-uw+k-l-iy+er r-aa+k-ax-t-s
    return phonemize(
        texts,
        language="en-us",
        backend="festival",
        separator=Separator(phone="-", word=" ", syllable="+"),
        strip=True,
        preserve_punctuation=False,
        njobs=8,
    )


def flatten(xss):
    return [x for xs in xss for x in xs]


def phonetic_tokenize(text):
    # input:
    # nuclear rockets
    # n-uw+k-l-iy+er|r-aa+k-ax-t-s

    text = text.replace("|", ",|,")

    text = text.replace("-", ",")
    text = text.replace("+", ",")

    tokens = text.split(",")
    return [phoneme_vocab.index(c) for c in tokens]


def phonetic_decode(ids, remove_consecutive=False):
    if remove_consecutive:
        ids = [i for i, _ in itertools.groupby(ids)]

    return "".join([phoneme_vocab[i] for i in ids])


# for text generation
vocab = [c for c in "-|etaonihsrdlumwcfgypbvk'xjqz_"]

ctc_vocab = vocab[:-1]


def tokenize(text):
    return [vocab.index(c) for c in text]


def decode(ids, remove_consecutive=False):
    if remove_consecutive:
        ids = [i for i, _ in itertools.groupby(ids)]

    return "".join([vocab[i] for i in ids])


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


def correct_channels(spikePow):
    return spikePow[:, correct_pos]
