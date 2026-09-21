"""One interface for every finetuning dataset. sft/lora/dpo take a Task and
never name a dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class Packed:
    """One split as a flat token stream plus two index arrays."""

    ids: np.ndarray  # uint16
    mask: np.ndarray  # bool, True where loss counts
    starts: np.ndarray  # int64, each example's first token
    vocab_size: int  # which tokenizer wrote these ids
    eot_id: int

    def __post_init__(self):
        assert self.ids.dtype == np.uint16 and self.mask.dtype == bool
        assert self.ids.shape == self.mask.shape
        assert np.all(np.diff(self.starts) > 0), "starts must be sorted and unique"
        assert self.starts[0] == 0 and self.starts[-1] < len(self.ids)
        assert self.ids.max(initial=0) < self.vocab_size
        assert self.eot_id < self.vocab_size

    def __len__(self) -> int:
        return len(self.ids)


def concat(a: Packed, b: Packed) -> Packed:
    """Mix two tasks. Windows open at starts, so start count sets the ratio."""
    assert a.vocab_size == b.vocab_size and a.eot_id == b.eot_id
    return Packed(
        ids=np.concatenate([a.ids, b.ids]),
        mask=np.concatenate([a.mask, b.mask]),
        starts=np.concatenate([a.starts, b.starts + len(a.ids)]),
        vocab_size=a.vocab_size,
        eot_id=a.eot_id,
    )


@runtime_checkable
class Task(Protocol):
    """A dataset, what counts as a good completion, and what to report."""

    name: str

    def build(self, split: str) -> Packed: ...

    def reward(self, prompt: str, completion: str) -> float:
        """Higher is better. Float not bool -- GRPO/RLOO need the magnitude."""
        ...

    def evaluate(self, model, n: int) -> dict[str, float]:
        """The task's own scoreboard, not val loss."""
        ...


def windows(
    packed: Packed,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """(x, y, keep) -- a batch of windows opening at example starts, so a
    prompt sits at position 0 as it does at inference. keep is read at i+1
    alongside y: it marks the token being predicted. y comes back unmasked so
    one pass can be scored on completion, prompt and everything."""
    ids, mask, starts = packed.ids, packed.mask, packed.starts
    # sorted, so the "fits in a window" cutoff is a binary search
    n_ok = np.searchsorted(starts, len(ids) - block_size - 1, side="right")
    assert n_ok > 0, f"no example fits in block_size {block_size}"
    ix = starts[rng.integers(n_ok, size=batch_size)]
    x = np.stack([ids[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([ids[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    keep = np.stack([mask[i + 1 : i + 1 + block_size] for i in ix])
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(keep).to(device),
    )


def masked_targets(y: Tensor, keep: Tensor) -> Tensor:
    """Targets with non-completion positions set to cross_entropy's ignore_index."""
    return y.masked_fill(~keep, -100)


def check_vocab(packed: Packed, model) -> None:
    """A vocab mismatch is silent: wrong-vocab ids are still valid ids. Read
    off the embedding -- GPT takes kwargs and keeps no config."""
    rows = model.token_embedding_table.weight.shape[0]
    assert packed.vocab_size == rows, (
        f"packed data is {packed.vocab_size}-vocab, model embedding has {rows} rows"
    )


if __name__ == "__main__":
    # assertions are in test_task.py; this just prints the data
    from tokenizer import load_tokenizer

    tok = load_tokenizer()
    eot = tok.specials["<|endoftext|>"]

    ids: list[int] = []
    mask: list[bool] = []
    starts: list[int] = []
    for w in ["cat", "dog", "bird"]:
        prompt = [tok.encode(c)[0] for c in w] + [tok.encode(">")[0]]
        comp = [tok.encode(c)[0] for c in reversed(w)] + [eot]
        starts.append(len(ids))
        ids += prompt + comp
        mask += [False] * len(prompt) + [True] * len(comp)

    packed = Packed(
        ids=np.array(ids, dtype=np.uint16),
        mask=np.array(mask, dtype=bool),
        starts=np.array(starts, dtype=np.int64),
        vocab_size=tok.vocab_size,
        eot_id=eot,
    )

    def d(t):
        return tok.decode([int(t)]).replace("<|endoftext|>", "<eot>")

    print(f"{len(packed)} tokens, {len(packed.starts)} examples\n")
    print(f"{'i':>3} {'id':>6}  {'text':<7} {'loss?':<6} start?")
    for i, (t, m) in enumerate(zip(packed.ids, packed.mask)):
        start = "<- start" if i in packed.starts else ""
        print(f"{i:>3} {t:>6}  {d(t)!r:<7} {m!s:<6} {start}")

    x, y, keep = windows(packed, 1, 8, np.random.default_rng(1))
    print("\none window, block_size 8:")
    print(f"{'t':>3}  {'x[t]':<7} {'-> y[t]':<8} trained?")
    for t in range(8):
        print(
            f"{t:>3}  {d(x[0, t])!r:<7} {d(y[0, t])!r:<8}"
            f" {'TRAIN' if keep[0, t] else '  .'}"
        )

    # '>' is a prompt token but predicts a completion one, so keep is True there.
    # The last column predicts the next example: out of the loss, not out of
    # attention -- that is mask.py's job.
