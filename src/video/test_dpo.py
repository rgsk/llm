import math
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from dpo import Pair, batch_logps, cut, dpo, dpo_loss, pick
from generate import length_groups
from gpt import GPT
from gpt_config import GPTConfig
from train_config import TrainConfig


@pytest.fixture(autouse=True)
def _logs_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr("logger.LOG_DIR", tmp_path)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))


@pytest.fixture
def ids(tok):
    return lambda s: [tok.encode(c)[0] for c in s]  # one id per character


@pytest.fixture
def eot(tok):
    return tok.specials["<|endoftext|>"]


@pytest.fixture
def gpt_cfg(tok):
    return GPTConfig(
        vocab_size=tok.vocab_size, block_size=32, n_embed=64, n_head=4, n_layer=2,
        attention="sdpa", norm="rms", ffn="gated", position="rope",
    )  # fmt: skip


@pytest.fixture
def model(gpt_cfg):
    return GPT(**asdict(gpt_cfg))


def test_a_pair_is_the_best_sample_against_the_worst():
    assert pick([0.2, 0.9, 0.5, 0.4]) == (1, 0)


def test_a_group_that_all_scored_the_same_makes_no_pair():
    assert pick([0.5, 0.5, 0.5]) is None


def test_a_sample_is_cut_after_its_first_eot(ids, eot):
    assert cut(ids("tac") + [eot] + ids("xx"), eot) == ids("tac") + [eot]
    assert cut(ids("tacxx"), eot) == ids("tacxx")  # never stopped: kept whole


def test_equal_length_prompts_share_a_generate_call():
    lengths = [len(w) for w in ["cat>", "dog>", "bird>", "yak>", "fish>", "owl>"]]
    assert length_groups(lengths, cap=8) == [[0, 1, 3, 5], [2, 4]]
    assert length_groups(lengths, cap=3) == [[0, 1, 3], [5], [2, 4]]


def test_loss_is_log2_when_the_policy_is_the_reference():
    same = torch.tensor([-12.0, -30.0])
    loss, z = dpo_loss(same, same - 5, same, same - 5, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert z.tolist() == [0.0, 0.0]


def test_loss_falls_as_the_policy_prefers_chosen_more_than_the_reference():
    zero = torch.zeros(1)
    losses = [
        dpo_loss(torch.tensor([gap]), zero, zero, zero, beta=1.0)[0].item()
        for gap in (-2.0, 0.0, 2.0)
    ]
    assert losses == pytest.approx([2.1269, 0.6931, 0.1269], abs=1e-4)


def test_only_the_change_from_the_reference_counts():
    # chosen is already 20 nats likelier under ref; that earns nothing by itself
    loss, _ = dpo_loss(
        torch.tensor([-10.0]), torch.tensor([-30.0]),
        torch.tensor([-10.0]), torch.tensor([-30.0]), beta=0.1,
    )  # fmt: skip
    assert loss.item() == pytest.approx(math.log(2))


def test_the_gradient_raises_chosen_and_lowers_rejected():
    pi_c = torch.tensor([-10.0], requires_grad=True)
    pi_r = torch.tensor([-12.0], requires_grad=True)
    loss, _ = dpo_loss(pi_c, pi_r, torch.tensor([-10.0]), torch.tensor([-12.0]), 0.1)
    loss.backward()
    assert pi_c.grad.item() == pytest.approx(-0.05)  # beta * sigmoid(-0) = 0.05
    assert pi_r.grad.item() == pytest.approx(+0.05)


def test_loss_matches_torch_logsigmoid():
    g = torch.Generator().manual_seed(0)
    pi_c, pi_r, ref_c, ref_r = (torch.randn(64, generator=g) * 20 for _ in range(4))
    loss, _ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    oracle = -F.logsigmoid(0.1 * ((pi_c - ref_c) - (pi_r - ref_r))).mean()
    assert loss.item() == pytest.approx(oracle.item(), rel=1e-6)


def test_logps_sum_the_completion_tokens_only(model, ids, eot):
    prompt, comp = ids("cat>"), ids("tac") + [eot]
    x = torch.tensor([prompt + comp])
    lp = model(x[:, :-1]).log_softmax(-1)[0]
    # position i predicts token i + 1: "t" is predicted at ">", index 3
    by_hand = sum(lp[3 + i, t] for i, t in enumerate(comp))
    assert batch_logps(model, [(prompt, comp)]).item() == pytest.approx(
        by_hand.item(), rel=1e-5
    )


def test_right_padding_does_not_change_a_row(model, ids, eot):
    short = (ids("cat>"), ids("tac") + [eot])
    long = (ids("bird>"), ids("drib") + [eot])
    alone = batch_logps(model, [short])
    padded = batch_logps(model, [short, long, long])[0]
    assert padded.item() == pytest.approx(alone.item(), abs=1e-5)


class Stub:
    name = "stub"

    def evaluate(self, model, n, **kw):
        return {}


@pytest.mark.slow
def test_dpo_learns_to_prefer_the_reversal(model, gpt_cfg, ids, eot):
    words = ["cat", "dog", "bird", "fish", "frog", "moth", "owl", "yak"]
    pairs = [
        Pair(ids(w + ">"), ids(w[::-1]) + [eot], ids(w) + [eot], 1.0, 0.0)
        for w in words
    ]
    cfg = TrainConfig(
        batch_size=4, max_steps=40, lr=1e-3, min_lr=1e-4, warmup_steps=4,
        eval_interval=40, eval_iters=1, use_compile=False, amp=False,
    )  # fmt: skip
    h = dpo(model, pairs, pairs, Stub(), cfg, gpt_cfg, beta=0.1, eval_n=0)
    assert h[0]["val_loss"] == pytest.approx(math.log(2))  # policy == ref
    assert h[0]["val_acc"] == 0.0
    assert h[-1]["val_acc"] == 1.0
    assert h[-1]["val_loss"] < 0.3
    assert h[-1]["d_chosen"] > h[-1]["d_rejected"]
