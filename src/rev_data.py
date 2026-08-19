"""Reverse-string task ("hello>" -> "olleh") in sft_data's packed format, so
sft.py's train/_windows/estimate_loss run on it unchanged.

Encoded char-by-char, not with greedy BPE: greedy merges common chunks
("hello" -> "he","llo"), which would turn reversal into a spelling task the
base model cannot know. Every lowercase letter and ">" is a single token in
the vocab, so char-level encoding reproduces the original sft.ipynb task
exactly -- position reflection, nothing else.
"""

from __future__ import annotations

import random
import string

import numpy as np
import torch

from main import device, tok

CHAR_IDS = np.array([tok.encode(c)[0] for c in string.ascii_lowercase])
GT = tok.encode(">")[0]
EOT = tok.encode("<|endoftext|>")  # not a special token in this vocab: 3 ids


def build_reverse(
    n_train: int = 200_000,
    n_val: int = 5_000,
    lmin: int = 3,
    lmax: int = 8,
    seed: int = 1337,
    weights: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Same keys as sft_data.build_or_load: {split}_{ids,mask,starts}.
    mask is True on the completion (reversed chars + eot), False on "word>".

    weights: relative sampling weight per length lmin..lmax; None -> uniform.
    Per-length accuracy tracks gradient share (9-10 hit 99% under 5-10 uniform
    but 85-90% under 1-10 uniform), so this is the rebalancing knob."""
    rng = np.random.default_rng(seed)
    lengths = np.arange(lmin, lmax + 1)
    p = np.ones(len(lengths)) if weights is None else np.asarray(weights, float)
    assert len(p) == len(lengths), (len(p), len(lengths))
    p = p / p.sum()
    out: dict[str, np.ndarray] = {}
    for split, n in (("train", n_train), ("val", n_val)):
        ids: list[int] = []
        mask: list[bool] = []
        starts: list[int] = []
        for _ in range(n):
            letters = CHAR_IDS[rng.integers(26, size=rng.choice(lengths, p=p))]
            prompt = [*letters, GT]
            comp = [*letters[::-1], *EOT]
            starts.append(len(ids))
            ids += prompt + comp
            mask += [False] * len(prompt) + [True] * len(comp)
        out[f"{split}_ids"] = np.array(ids, dtype=np.uint16)
        out[f"{split}_mask"] = np.array(mask, dtype=bool)
        out[f"{split}_starts"] = np.array(starts, dtype=np.int64)
    return out


@torch.no_grad()
def evaluate_reverse(
    model, n: int = 200, lmin: int = 3, lmax: int = 8, slack: int = 0
) -> tuple[float, list[tuple[str, str]]]:
    """Exact-match accuracy on freshly sampled words, greedy decoding.

    Generation gets l+slack tokens and the prediction is cut at the first eot
    token, so accuracy no longer depends on the generation budget. Termination
    stays part of the task: a correct reversal that never emits eot keeps its
    trailing junk and scores wrong. lmin/lmax outside the training range
    measures length generalization."""
    was_training = model.training
    model.eval()
    correct, samples = 0, []
    char_ids = {c: int(i) for c, i in zip(string.ascii_lowercase, CHAR_IDS)}
    for _ in range(n):
        l = random.randint(lmin, lmax)
        w = "".join(random.choices(string.ascii_lowercase, k=l))
        prompt = torch.tensor([[*(char_ids[c] for c in w), GT]], device=device)
        out = model.generate(prompt, max_new_tokens=l + slack, temperature=0.0)
        gen = out[0, l + 1 :].tolist()
        pred = tok.decode(gen[: gen.index(EOT[0])] if EOT[0] in gen else gen)
        correct += pred == w[::-1]
        r = [w, len(w), pred, len(pred)]
        r += [tok.decode(gen), len(gen)]
        samples.append(r)
    if was_training:
        model.train()
    return correct / n, samples[:5]
