"""The first Task: "cat>" -> "tac<eot>". It goes before sft.py as the trainer's
unit test -- a 2-layer model solves it, so a low score is a bug in the loop.

Char-level ids, not greedy BPE: merged chunks would make reversal a spelling
task. Synthetic, so no on-disk cache -- 200k examples build in 0.3s.
"""

from __future__ import annotations

import string

import numpy as np
import torch

from generate import generate
from task import Packed
from tokenizer import ENDOFTEXT, tokenizer_for

LETTERS = string.ascii_lowercase
SEP = ">"


class Reverse:
    """Random lowercase words, lengths lmin..lmax. weights re-mixes the lengths;
    per-length accuracy follows each length's share of the kept tokens."""

    name = "reverse"

    def __init__(
        self,
        tok=None,
        n_train: int = 200_000,
        n_val: int = 5_000,
        lmin: int = 3,
        lmax: int = 8,
        seed: int = 1337,
        weights: list[float] | None = None,
    ):
        # the corpus the base was pretrained on picks the vocab, not GPT-2's
        self.tok = tokenizer_for() if tok is None else tok
        self.n_train, self.n_val, self.seed = n_train, n_val, seed
        self.eot = self.tok.specials[ENDOFTEXT]

        encoded = [self.tok.encode(c) for c in LETTERS + SEP]
        assert all(len(ids) == 1 for ids in encoded), (
            "this vocab has no single token for some character -- reversal "
            "would become a spelling task"
        )
        self.char_ids = np.array([ids[0] for ids in encoded[:-1]])  # a..z
        self.sep_id = encoded[-1][0]

        self.lengths = np.arange(lmin, lmax + 1)
        p = (
            np.ones(len(self.lengths))
            if weights is None
            else np.asarray(weights, float)
        )
        assert len(p) == len(self.lengths), (len(p), len(self.lengths))
        self.p = p / p.sum()

    def build(self, split: str) -> Packed:
        """ "word>" + reversed(word) + eot, mask True on the second half."""
        assert split in ("train", "val")
        n = self.n_train if split == "train" else self.n_val
        # split is part of the seed: train and val draw from different streams
        rng = np.random.default_rng([self.seed, ("train", "val").index(split)])

        lens = rng.choice(self.lengths, size=n, p=self.p)
        letters = self.char_ids[rng.integers(26, size=int(lens.sum()))]
        ids = np.empty(int((2 * lens + 2).sum()), dtype=np.uint16)
        mask = np.zeros(len(ids), dtype=bool)
        starts = np.empty(n, dtype=np.int64)

        i = j = 0
        for e, l in enumerate(lens):
            w = letters[j : j + l]
            j += l
            starts[e] = i
            ids[i : i + l] = w
            ids[i + l] = self.sep_id
            ids[i + l + 1 : i + 2 * l + 1] = w[::-1]
            ids[i + 2 * l + 1] = self.eot
            mask[i + l + 1 : i + 2 * l + 2] = True
            i += 2 * l + 2

        return Packed(
            ids=ids,
            mask=mask,
            starts=starts,
            vocab_size=self.tok.vocab_size,
            eot_id=self.eot,
        )

    def reward(self, prompt: str, completion: str) -> float:
        """Fraction of characters in the right place, 1.0 for the exact
        reversal. Graded because GRPO ranks within a group; dividing by the
        longer string makes a truncated and a rambling answer both lose."""
        target = prompt.strip().removesuffix(SEP)[::-1]
        pred = completion.split(ENDOFTEXT)[0].strip()
        hits = sum(a == b for a, b in zip(target, pred))
        return hits / max(len(target), len(pred), 1)

    def evaluate(
        self,
        model,
        n: int = 200,
        lmin: int | None = None,
        lmax: int | None = None,
        slack: int = 1,
        seed: int = 0,
    ) -> dict[str, float]:
        """Greedy decode n fresh words: {exact_match, reward, stop_rate}.

        Budget is l + slack, one token past the reversal, so failing to stop
        costs length. lmin/lmax outside the training range measures length
        generalisation -- fresh words alone are not held out, length 3 is only
        17576 words.
        """
        lo = self.lengths[0] if lmin is None else lmin
        hi = self.lengths[-1] if lmax is None else lmax
        rng = np.random.default_rng(seed)
        device = next(model.parameters()).device

        rewards, stops = [], []
        lens = rng.integers(lo, hi + 1, size=n)
        for l in np.unique(lens):
            # one call per length: same-length prompts need no padding
            idx = rng.integers(26, size=(int((lens == l).sum()), int(l)))
            words = ["".join(LETTERS[i] for i in row) for row in idx]
            prompt = np.concatenate(
                [self.char_ids[idx], np.full((len(idx), 1), self.sep_id)], axis=1
            )
            x = torch.from_numpy(prompt).to(device)
            # uncached: crops instead of asserting, so any backend works
            out = generate(model, x, max_new_tokens=int(l) + slack, temperature=0.0)
            for w, row in zip(words, out[:, int(l) + 1 :].tolist()):
                text = self.tok.decode(row)
                rewards.append(self.reward(w + SEP, text))
                stops.append(ENDOFTEXT in text)

        return {
            "exact_match": float(
                np.mean([r == 1.0 for r in rewards])
            ),  # reward's ceiling
            "reward": float(np.mean(rewards)),
            "stop_rate": float(np.mean(stops)),
        }


if __name__ == "__main__":
    # assertions are in test_reverse.py; this just prints the data
    task = Reverse(n_train=6, n_val=2, lmin=3, lmax=5)
    packed = task.build("train")
    tok = task.tok

    print(f"{len(packed)} tokens, {len(packed.starts)} examples\n")
    print("prompt | completion   (| is where the loss starts counting)")
    for s, e in zip(packed.starts, [*packed.starts[1:], len(packed)]):
        text = tok.decode(packed.ids[s:e].tolist()).replace(ENDOFTEXT, "<eot>")
        cut = int(packed.mask[s:e].argmax())
        print(f"  {text[:cut]:>6} | {text[cut:]}")

    print("\nreward")
    for completion in ["tac", "tac<|endoftext|>", "ta", "tacx", "tca", "cat", "zzz"]:
        print(f"  cat> -> {completion:<18} {task.reward('cat>', completion):.2f}")

    for weights in (None, [1, 2, 4]):
        p = Reverse(tok, n_train=4000, lmin=3, lmax=5, weights=weights).build("train")
        lens = (np.diff([*p.starts, len(p)]) - 2) // 2
        mix = " ".join(f"{l}: {(lens == l).mean():.0%} " for l in (3, 4, 5))
        print(f"\nweights={weights!s:<12} {mix}")
