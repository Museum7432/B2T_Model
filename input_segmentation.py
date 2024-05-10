import scipy.io as sio
import scipy
import os
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import os

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


def main():
    ckpt_path = "/home/arch/projects/brain2text/brain2text-main/outputs/2024-05-08_20-59-44/ckpts/last.ckpt"

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
        }
    )

    dm = B2T_DataModule(cfg)
    dm.setup("")

    dataloader = dm.val_dataloader()

    spikePow_indices = np.load(os.path.join(input_dir, "spikePow_indices.npy"))
    sp_idx = 0

    sentence_id = []

    words_list = []

    word_sp_indices = []

    for batch in tqdm(dataloader):
        _start, _end = spikePow_indices[sp_idx]

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

        aligned_tokens, alignment_scores = align(emission, targets_ids)

        token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

        seperators_id = []
        for t in token_spans:
            if t.token == 1:
                seperators_id.append((t.start + t.end + 1) // 2)

        # seperators_id = [0] + seperators_id + [len(emission[0]) - 1]

        # convert to index of spikePow
        seperators_id = [i * 4 + 2 + _start for i in seperators_id]

        seperators_id = [_start] + seperators_id + [_end]

        assert max(seperators_id) <= _end

        word_spans = [
            [seperators_id[i], seperators_id[i + 1]]
            for i in range(0, len(seperators_id) - 1)
        ]

        words = target_text.split("|")

        assert len(words) == len(word_spans)

        sentence_id += [sp_idx] * len(words)
        words_list += words

        word_sp_indices.append(word_spans)

        sp_idx += 1

    word_sp_indices = np.vstack(word_sp_indices)

    ph_words_list = phonemize_text(words_list)

    words_list = np.array(words_list)

    sentence_id = np.array(sentence_id)

    np.save(os.path.join(input_dir, "words_list.npy"), words_list)
    np.save(os.path.join(input_dir, "ph_words_list.npy"), ph_words_list)
    np.save(os.path.join(input_dir, "word_sp_indices.npy"), word_sp_indices)
    np.save(os.path.join(input_dir, "sentence_id.npy"), sentence_id)


if __name__ == "__main__":
    main()
