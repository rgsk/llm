"""Two Tasks trained as one stream.

The mix is a share of STARTS, not of tokens: windows() opens a window at each
start, so start count is the sampling ratio. Sizing is done by dropping starts
rather than by asking a task for fewer examples -- nothing in the Task protocol
exposes its own size.

Dropped starts keep their tokens in the stream; they are just never opened on,
so they survive as context for the window before them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from instruct import Instruct
from reverse import Reverse
from task import Packed, Task, concat
from tokenizer import ENDOFTEXT

DEFAULT_FRACS = (0.67, 0.33)  # the old track's instruct/reverse mix


def take_starts(packed: Packed, n: int, rng: np.random.Generator) -> Packed:
    """Keep n of the starts, chosen uniformly. Index 0 always stays -- Packed
    asserts the first example begins at token 0."""
    assert 0 < n <= len(packed.starts), f"want {n} of {len(packed.starts)} starts"
    if n == len(packed.starts):
        return packed
    rest = rng.choice(len(packed.starts) - 1, size=n - 1, replace=False) + 1
    return replace(packed, starts=np.sort(np.concatenate([[0], rest])))


class Joint:
    """A mix of Tasks. build() concatenates them at the requested ratio;
    evaluate() returns every member's scoreboard, prefixed by task name."""

    def __init__(
        self,
        tasks: tuple[Task, ...] | None = None,
        fracs: tuple[float, ...] = DEFAULT_FRACS,
        seed: int = 1337,
        name: str = "joint",
    ):
        self.tasks = (Instruct(), Reverse()) if tasks is None else tuple(tasks)
        self.fracs = tuple(fracs)
        assert len(self.tasks) == len(self.fracs), (len(self.tasks), len(self.fracs))
        assert abs(sum(self.fracs) - 1.0) < 1e-6, f"fracs sum to {sum(self.fracs)}"
        assert all(f > 0 for f in self.fracs), self.fracs
        self.seed = seed
        self.name = name

    def build(self, split: str) -> Packed:
        packs = [t.build(split) for t in self.tasks]
        # the biggest total whose every share is actually available
        total = int(min(len(p.starts) / f for p, f in zip(packs, self.fracs)))
        rng = np.random.default_rng([self.seed, ("train", "val").index(split)])
        sized = [
            take_starts(p, max(1, round(f * total)), rng)
            for p, f in zip(packs, self.fracs)
        ]
        out = sized[0]
        for p in sized[1:]:
            out = concat(out, p)
        return out

    def reward(self, prompt: str, completion: str) -> float:
        raise NotImplementedError(
            "a joint stream has no single reward -- call the member task's "
            "reward, which knows what its own prompts mean"
        )

    def evaluate(self, model, n: int = 200) -> dict[str, float]:
        """Every member's scoreboard. Names are prefixed, so two tasks that both
        report stop_rate do not overwrite each other."""
        return {
            f"{t.name}_{k}": v
            for t in self.tasks
            for k, v in t.evaluate(model, n).items()
        }


if __name__ == "__main__":
    # assertions are in test_joint.py; this prints the mix
    task = Joint()
    for split in ("train", "val"):
        packed = task.build(split)
        starts = [len(t.build(split).starts) for t in task.tasks]
        sized = [round(f * int(min(s / f for s, f in zip(starts, task.fracs))))
                 for f in task.fracs]  # fmt: skip
        print(f"\n{split}: {len(packed):,} tokens, {len(packed.starts):,} starts")
        for t, avail, got, f in zip(task.tasks, starts, sized, task.fracs):
            print(
                f"  {t.name:<9} {got:>8,} of {avail:>8,} starts"
                f"   {got / len(packed.starts):>6.1%} (asked {f:.0%})"
            )

    p = task.build("val")
    tok = task.tasks[0].tok
    print("\nfirst start of each half:")
    for i in (0, len(p.starts) - 1):
        s = int(p.starts[i])
        text = tok.decode(p.ids[s : s + 24].tolist()).replace(ENDOFTEXT, "<eot>")
        print(f"  start {i:>6}: {text!r}")
