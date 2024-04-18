import numpy as np
from phonemizer import phonemize
from phonemizer.separator import Separator
import itertools


# require "festival" to be installed

# el, em, en, pau not found in trainning dataset
# _: blank token
# ' ': silent token
phoneme_vocab = "_, ,aa,ae,ah,ao,aw,ax,ay,eh,el,em,en,er,ey,ih,iy,ow,oy,uh,uw,b,ch,d,dh,f,g,hh,jh,k,l,m,n,ng,p,r,s,sh,t,th,v,w,y,z,zh,pau".split(
    ","
)

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
    # n-uw+k-l-iy+er r-aa+k-ax-t-s

    text = text.replace(' ', ', ,')

    text = text.replace('-', ',')
    text = text.replace('+', ',')

    tokens = text.split(',')

    return [phoneme_vocab.index(c) for c in tokens]


def phonetic_decode(ids, remove_consecutive=False):
    if remove_consecutive:
        ids = [i for i, _ in itertools.groupby(ids)]
    return "".join([phoneme_vocab[i] for i in ids])


# for text generation
vocab = "_ nuclearoktsdyifwhbvxpm'gqjz"

def tokenize(text):
    return [vocab.index(c) for c in text]


def decode(ids, remove_consecutive=False):
    if remove_consecutive:
        ids = [i for i, _ in itertools.groupby(ids)]
    return "".join([vocab[i] for i in ids])