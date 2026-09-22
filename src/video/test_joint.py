import numpy as np
import pytest
import torch

from joint import Joint, take_starts
from reverse import Reverse
from task import windows


@pytest.fixture
def rev(tok):
    return Reverse(tok, n_train=600, n_val=200, lmin=3, lmax=3)


@pytest.fixture
def rev2(tok):
    # a second stream, distinguishable by length alone
    return Reverse(tok, n_train=600, n_val=200, lmin=5, lmax=5, seed=99)


def test_taking_starts_keeps_the_tokens_and_drops_only_the_openings(build):
    packed = build(["cat", "dog", "bird"])
    fewer = take_starts(packed, 2, np.random.default_rng(0))
    assert len(fewer.starts) == 2
    assert np.array_equal(fewer.ids, packed.ids)  # the stream is untouched
    assert np.array_equal(fewer.mask, packed.mask)


def test_start_zero_always_survives(build):
    packed = build(["cat", "dog", "bird"])
    for seed in range(8):
        assert take_starts(packed, 2, np.random.default_rng(seed)).starts[0] == 0


def test_asking_for_more_starts_than_exist_is_refused(build):
    packed = build(["cat", "dog"])
    with pytest.raises(AssertionError, match="want 5 of 2 starts"):
        take_starts(packed, 5, np.random.default_rng(0))


def test_the_mix_is_a_share_of_starts(rev, rev2):
    task = Joint((rev, rev2), (0.67, 0.33))
    packed = task.build("train")
    # rev examples are 8 tokens, rev2's are 12, so length identifies the half
    lens = np.diff([*packed.starts, len(packed)])
    assert len(packed.starts) == 895  # 600 / 0.67, capped by the smaller share
    assert round((lens == 8).sum() / len(packed.starts), 2) == 0.67


def test_fracs_that_do_not_sum_to_one_are_refused(rev, rev2):
    with pytest.raises(AssertionError, match="sum to"):
        Joint((rev, rev2), (0.5, 0.3))


def test_one_frac_per_task(rev, rev2):
    with pytest.raises(AssertionError):
        Joint((rev, rev2), (1.0,))


def test_both_halves_are_reachable_as_windows(rev, rev2):
    task = Joint((rev, rev2), (0.5, 0.5))
    packed = task.build("train")
    x, _, _ = windows(packed, 64, 16, np.random.default_rng(0))
    # every window opens on an example, so column 0 is always a first letter
    assert x.shape == (64, 16)
    assert len(set(x[:, 0].tolist())) > 1


def test_the_concatenated_stream_stays_sorted(rev, rev2):
    packed = Joint((rev, rev2), (0.67, 0.33)).build("train")
    assert np.all(np.diff(packed.starts) > 0)  # windows() binary-searches these
    assert packed.starts[0] == 0
    assert packed.starts[-1] < len(packed.ids)


def test_train_and_val_draw_different_starts(rev, rev2):
    task = Joint((rev, rev2), (0.67, 0.33))
    assert not np.array_equal(task.build("train").ids, task.build("val").ids)


def test_the_mix_is_reproducible(rev, rev2):
    a = Joint((rev, rev2), (0.67, 0.33), seed=7).build("train")
    b = Joint((rev, rev2), (0.67, 0.33), seed=7).build("train")
    c = Joint((rev, rev2), (0.67, 0.33), seed=8).build("train")
    assert np.array_equal(a.starts, b.starts)
    assert not np.array_equal(a.starts, c.starts)


def test_the_scoreboard_prefixes_every_member(rev, rev2, tok):
    from dataclasses import asdict

    from gpt import GPT
    from gpt_config import GPTConfig

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        n_embed=32,
        n_head=2,
        n_layer=1,
        attention="sdpa",
        position="rope",
    )
    torch.manual_seed(0)
    scores = Joint((rev, rev2), (0.67, 0.33)).evaluate(GPT(**asdict(cfg)), n=2)
    # both halves are Reverse, so unprefixed keys would collide to one set
    assert set(scores) == {
        "reverse_exact_match",
        "reverse_reward",
        "reverse_stop_rate",
    }


def test_a_joint_stream_refuses_to_pretend_it_has_one_reward(rev, rev2):
    task = Joint((rev, rev2), (0.67, 0.33))
    with pytest.raises(NotImplementedError, match="no single reward"):
        task.reward("cat>", "tac")
