import scipy.io as sio
import scipy
import os
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import os
from argparse import ArgumentParser
import pickle
import torchaudio.functional as F

from src.model import joint_Model
from src.data import B2T_DataModule

from src.utils import phonemize_text


def align(emission, targets):

    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = (
        alignments[0],
        scores[0],
    )  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()

    ckpt_path = args.ckpt

    model = joint_Model.load_from_checkpoint(
        ckpt_path, decoders_conf=["ctc_al"], strict=False
    )

    model = model.cuda()

    input_dir = "./dataset/train"

    cfg = OmegaConf.create(
        {
            "train_data_dir": None,
            "val_data_dir": input_dir,
            "test_data_dir": None,
            "train_batch_size": 1,
            "valid_batch_size": 1,  # no padding
            # "debugging": True,
            "num_workers": 2,
            "word_level":False
        }
    )

    dm = B2T_DataModule(cfg)
    dm.setup("")

    dataloader = dm.val_dataloader()

    word_sp_indices = []

    for batch in tqdm(dataloader):

        emission = (
            model(
                spikePow=batch["spikePow"].cuda(),
                spikePow_mask=batch["spikePow_mask"].cuda(),
                spikePow_lens=batch["spikePow_lens"].cuda(),
            )["ctc_al"][0]
            .detach()
            .cpu()
        )

        targets_ids = batch["sent_ids"][:, :-1]

        target_text = batch["sent"][0]

        TRANSCRIPT = target_text.split("|")
        aligned_tokens, alignment_scores = align(emission, targets_ids)

        token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

        token_spans = [i for i in token_spans if i.token != 1]

        word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])


        # index of the start and end of each word in spikePow
        word_indices = []

        for w_span in word_spans:
            # convert to index of spikePow
            s_start = w_span[0].start * 4
            s_end = w_span[-1].end * 4 + 3

            word_indices.append([
                s_start,
                s_end
            ])

        word_sp_indices.append(word_indices)

    with open(os.path.join(input_dir, "word_sp_indices.npy"), 'wb') as handle:
        pickle.dump(word_sp_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
