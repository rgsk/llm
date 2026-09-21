import numpy as np
import pytest
import torch

from gpt import GPT
from reverse import Reverse
from task import Task, windows
from tokenizer import ENDOFTEXT


class Oracle:
    """A model that already solves the task: its argmax is the next token of
    the reversal, then eot (or one more letter, with stop=False)."""

    block_size = 64
    training = False

    def __init__(self, task: Reverse, stop: bool = True):
        self.task, self.stop = task, stop

    def parameters(self):
        return iter([torch.zeros(1)])

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*idx.shape, self.task.tok.vocab_size)
        for b, row in enumerate(idx.tolist()):
            sep = row.index(self.task.sep_id)
            word, written = row[:sep], row[sep + 1 :]
            if len(written) < len(word):
                nxt = word[::-1][len(written)]
            else:
                nxt = self.task.eot if self.stop else word[0]
            logits[b, -1, nxt] = 1.0
        return logits


@pytest.fixture
def task(tok):
    return Reverse(tok, n_train=3, n_val=2, lmin=3, lmax=5)


@pytest.fixture
def task3(tok):
    return Reverse(tok, n_train=2, lmin=3, lmax=3)


@pytest.fixture
def gpt(tok):
    return GPT(
        vocab_size=tok.vocab_size, block_size=16, n_embed=32, n_head=4, n_layer=2
    )


def test_an_example_is_a_word_then_its_reversal(task, chars):
    p = task.build("train")
    assert chars(p.ids) == [
        *"kyhwm>mwhyk", "<eot>",
        *"dbf>fbd", "<eot>",
        *"eihmo>omhie", "<eot>",
    ]  # fmt: skip
    assert p.starts.tolist() == [0, 12, 20]


def test_the_prompt_is_out_of_the_loss(task):
    p = task.build("train")
    assert (
        p.mask[:12].tolist() == [False] * 6 + [True] * 6
    )  # "kyhwm>" then "mwhyk<eot>"


def test_every_example_reverses_its_own_word(tok):
    p = Reverse(tok, n_train=500, lmin=3, lmax=8).build("train")
    for s, e in zip(p.starts, [*p.starts[1:], len(p)]):
        word, answer = tok.decode(p.ids[s:e].tolist()).split(">", 1)
        assert answer == word[::-1] + ENDOFTEXT


def test_lengths_stay_inside_the_range(tok):
    p = Reverse(tok, n_train=500, lmin=3, lmax=8).build("train")
    lens = (np.diff([*p.starts, len(p)]) - 2) // 2
    assert sorted(set(lens)) == [3, 4, 5, 6, 7, 8]
    assert int(p.mask.sum()) == len(p) // 2  # prompt and completion are equal length


def test_weights_pick_the_lengths(tok, chars):
    p = Reverse(tok, n_train=4, lmin=3, lmax=5, weights=[0, 0, 1]).build("train")
    assert chars(p.ids)[:12] == [*"hwmdb>bdmwh", "<eot>"]
    assert p.starts.tolist() == [0, 12, 24, 36]  # every example is 5 + 1 + 5 + 1


def test_val_is_a_different_stream_of_the_same_task(task, chars):
    assert chars(task.build("val").ids)[:12] == [*"yazis>sizay", "<eot>"]
    assert chars(task.build("train").ids) == chars(task.build("train").ids)


def test_a_window_trains_only_on_the_reversal(task3, chars):
    p = task3.build("train")  # 2 examples of 8 tokens, so one legal window at 0
    x, _, keep = windows(p, 1, 8, np.random.default_rng(0))
    assert chars(x[0]) == [*"oxk>kxo", "<eot>"]
    assert keep[0].tolist() == [False] * 3 + [True] * 4 + [False]


def test_reward_is_one_for_the_exact_reversal(task):
    assert task.reward("cat>", "tac") == 1.0
    assert task.reward("cat>", "tac" + ENDOFTEXT) == 1.0


def test_reward_is_graded_not_exact_match(task):
    assert task.reward("cat>", "tao") == 2 / 3
    assert task.reward("cat>", "cat") == 1 / 3  # only the middle letter lands
    assert task.reward("cat>", "zzz") == 0.0


def test_reward_pays_for_stopping_on_time(task):
    assert task.reward("cat>", "ta") == 2 / 3  # stopped early
    assert task.reward("cat>", "tacc") == 3 / 4  # never stopped


def test_evaluate_scores_a_model_that_knows_the_task(task3):
    assert task3.evaluate(Oracle(task3), 4) == {
        "exact_match": 1.0,
        "reward": 1.0,
        "stop_rate": 1.0,
    }


def test_evaluate_counts_a_correct_reversal_that_never_stops_as_wrong(task3):
    assert task3.evaluate(Oracle(task3, stop=False), 4) == {
        "exact_match": 0.0,
        "reward": 0.75,  # 3 of 4 characters, the 4th is junk
        "stop_rate": 0.0,
    }


def test_an_untrained_model_scores_zero(task3, gpt):
    assert task3.evaluate(gpt, 4)["exact_match"] == 0.0


def test_evaluate_leaves_the_model_in_training_mode(task3, gpt):
    gpt.train()
    task3.evaluate(gpt, 2)
    assert gpt.training


def test_reverse_is_a_task(task):
    assert isinstance(task, Task)
