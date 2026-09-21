import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gpt import GPT
from task import Packed, Task, check_vocab, concat, masked_targets, windows


def test_packed_is_one_flat_stream(build, chars):
    p = build(["cat", "dog", "bird"])
    assert chars(p.ids) == [
        *"cat>tac", "<eot>",
        *"dog>god", "<eot>",
        *"bird>drib", "<eot>",
    ]  # fmt: skip
    assert p.starts.tolist() == [0, 8, 16]
    assert p.mask[:8].tolist() == [False] * 4 + [True] * 4  # "cat>" then "tac<eot>"


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"starts": np.array([0, 20, 10])}, id="unsorted-starts"),
        pytest.param({"vocab_size": 8}, id="ids-out-of-range"),
        pytest.param({"mask": np.ones(3, dtype=bool)}, id="mask-length-mismatch"),
    ],
)
def test_packed_rejects(build, bad):
    p = build(["cat", "dog", "bird"])
    with pytest.raises(AssertionError):
        Packed(**{**p.__dict__, **bad})


def test_windows_open_at_example_starts(build, chars):
    p = build(["cat", "dog", "bird"])
    x, _, _ = windows(p, 16, 8, np.random.default_rng(0))
    assert set(chars(x[:, 0])) <= {"c", "d", "b"}  # never mid-word


def test_keep_marks_the_token_being_predicted(build, chars):
    # two 8-token examples at block_size 8 leaves exactly one legal window
    p = build(["cat", "dog"])
    x, y, keep = windows(p, 1, 8, np.random.default_rng(0))
    assert chars(x[0]) == [*"cat>tac", "<eot>"]
    assert chars(y[0]) == [*"at>tac", "<eot>", "d"]
    # '>' is a prompt token, but what it predicts is the first completion token
    assert keep[0].tolist() == [False] * 3 + [True] * 4 + [False]


def test_window_overhangs_into_the_next_example(build, chars):
    p = build(["cat", "dog"])
    _, y, keep = windows(p, 1, 8, np.random.default_rng(0))
    assert chars(y[0])[-1] == "d"  # "dog" -- a different example
    assert not keep[0, -1]  # out of the loss, still inside attention


def test_masked_targets_hides_the_prompt(build, chars):
    p = build(["cat", "dog"])
    _, y, keep = windows(p, 1, 8, np.random.default_rng(0))
    ym = masked_targets(y, keep)
    assert (ym[0] == -100).tolist() == [True] * 3 + [False] * 4 + [True]
    assert chars(ym[0][3:7]) == [*"tac", "<eot>"]  # what survives is the reversal


def test_masked_loss_equals_loss_on_kept_tokens(build, tok):
    p = build(["cat", "dog"])
    _, y, keep = windows(p, 1, 8, np.random.default_rng(0))
    logits = torch.randn(8, tok.vocab_size)
    flat, fkeep = y.reshape(-1), keep.reshape(-1)
    masked = F.cross_entropy(
        logits, masked_targets(y, keep).reshape(-1), ignore_index=-100
    )
    by_hand = F.cross_entropy(logits[fkeep], flat[fkeep])
    assert torch.allclose(masked, by_hand)


def test_loss_counts_kept_tokens_not_examples(build):
    _, _, short = windows(build(["ab", "cd"]), 1, 6, np.random.default_rng(0))
    _, _, longer = windows(build(["abcde", "fghij"]), 1, 12, np.random.default_rng(0))
    # one mean over all kept targets, so the longer completion pulls twice as hard
    assert int(short.sum()) == 3
    assert int(longer.sum()) == 6


def test_concat_reindexes_the_second_stream(build, chars):
    m = concat(build(["cat"]), build(["dog"]))
    assert m.starts.tolist() == [0, 8]
    assert chars(m.ids) == [*"cat>tac", "<eot>", *"dog>god", "<eot>"]


def test_concat_mixes_by_start_count(build, chars):
    random.seed(0)
    a = build(["".join(random.choices("abcdefghijklm", k=4)) for _ in range(200)])
    b = build(["".join(random.choices("nopqrstuvwxyz", k=4)) for _ in range(100)])
    m = concat(a, b)
    assert len(m.starts) == 300

    first = chars(windows(m, 2000, 10, np.random.default_rng(0))[0][:, 0])
    from_b = sum(c in "nopqrstuvwxyz" for c in first) / len(first)
    assert 0.30 < from_b < 0.37  # b is 100 of 300 starts, so 1/3 of the windows


def test_check_vocab_catches_a_mismatch(build):
    p = build(["cat"])
    small = GPT(vocab_size=4096, block_size=16, n_embed=32, n_head=4, n_layer=2)
    with pytest.raises(AssertionError, match="4096 rows"):
        check_vocab(p, small)


def test_task_is_a_protocol(build):
    class Reverse:
        name = "reverse"

        def build(self, split):
            return build(["cat", "dog"])

        def reward(self, prompt, completion):
            return float(completion == prompt[:-1][::-1])

        def evaluate(self, model, n):
            return {"exact_match": 1.0}

    assert isinstance(Reverse(), Task)  # three methods, no base class
