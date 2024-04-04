import numpy as np
from phonemizer import phonemize
from phonemizer.separator import Separator


# require "festival" to be installed

# el, em, en, pau not found in trainning dataset
# _: blank token
# |: silent token
phoneme_vocab = "_,|,aa,ae,ah,ao,aw,ax,ay,eh,el,em,en,er,ey,ih,iy,ow,oy,uh,uw,b,ch,d,dh,f,g,hh,jh,k,l,m,n,ng,p,r,s,sh,t,th,v,w,y,z,zh,pau".split(
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

    # 'n-uw+k-l-iy+er'    '|'    'r-aa+k-ax-t-s'
    text = flatten([[c, "|"] for c in text.split()])[:-1]

    # 'n-uw'    'k-l-iy'    'er'    '|'    'r-aa'   'k-ax-t-s'
    text = flatten([c.split("+") for c in text])

    # n uw k l iy er | r aa k ax t s
    text = flatten([c.split("-") for c in text])

    text = ["|"] + text

    return [phoneme_vocab.index(c) for c in text]


# for text generation
vocab = "#nuclear oktsdyifwhbvxpm'gqjz"


def tokenize(text):
    return [vocab.index(c) for c in text]



def unscrambleChans(timeSeriesDat):
    chanToElec = [63, 64, 62, 61, 59, 58, 60, 54, 57, 50, 53, 49, 52, 45, 55, 44, 56, 39, 51, 43,
                  46, 38, 48, 37, 47, 36, 42, 35, 41, 34, 40, 33, 96, 90, 95, 89, 94, 88, 93, 87,
                  92, 82, 86, 81, 91, 77, 85, 83, 84, 78, 80, 73, 79, 74, 75, 76, 71, 72, 68, 69,
                  66, 70, 65, 67, 128, 120, 127, 119, 126, 118, 125, 117, 124, 116, 123, 115, 122, 114, 121, 113,
                  112, 111, 109, 110, 107, 108, 106, 105, 104, 103, 102, 101, 100, 99, 97, 98, 32, 30, 31, 29,
                  28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 17, 7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 8]
    chanToElec = np.array(chanToElec).astype(np.int32)-1
    
    unscrambledDat = timeSeriesDat.copy()
    for x in range(len(chanToElec)):
        unscrambledDat[:,chanToElec[x]] = timeSeriesDat[:,x]
        
    return unscrambledDat