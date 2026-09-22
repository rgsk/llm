import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from gpt import GPT
from gpt_config import GPTConfig
from reverse import Reverse
from sft import sft, split_losses
from train_config import TrainConfig


@pytest.fixture(autouse=True)
def _logs_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr("logger.LOG_DIR", tmp_path)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))  # Run always inits, offline


@pytest.fixture
def task(tok):
    return Reverse(tok, n_train=2_000, n_val=200, lmin=3, lmax=4)


@pytest.fixture
def gpt_cfg(tok):
    return GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=64,
        n_embed=64,
        n_head=4,
        n_layer=2,
        attention="sdpa",
        norm="rms",
        ffn="gated",
        position="rope",
    )


@pytest.fixture
def model(gpt_cfg):
    torch.manual_seed(0)
    return GPT(**asdict(gpt_cfg))


@pytest.fixture
def cfg():
    return TrainConfig(
        batch_size=8,
        max_steps=40,
        lr=3e-3,
        min_lr=3e-4,
        warmup_steps=5,
        eval_interval=40,
        eval_iters=3,
        name="test_sft",
        use_compile=False,
        use_wandb=False,
    )


def test_an_untrained_model_scores_both_halves_at_ln_vocab(task, model, tok):
    comp, prompt = split_losses(
        model, task.build("val"), 8, 32, 3, np.random.default_rng(0), doc_mask=False
    )
    assert abs(comp - math.log(tok.vocab_size)) < 0.1
    assert abs(prompt - math.log(tok.vocab_size)) < 0.1


def test_the_two_halves_are_scored_on_different_tokens(task, model):
    packed = task.build("val")
    a = split_losses(model, packed, 8, 32, 3, np.random.default_rng(0), doc_mask=False)
    b = split_losses(model, packed, 8, 32, 3, np.random.default_rng(0), doc_mask=False)
    assert a == b  # same rng, same windows
    assert a[0] != a[1]


def test_a_window_wider_than_the_model_is_refused(task, model, cfg, gpt_cfg):
    with pytest.raises(AssertionError, match="block_size"):
        sft(model, task, cfg, gpt_cfg, block_size=gpt_cfg.block_size + 1)


def test_a_narrower_window_is_fine(task, model, cfg, gpt_cfg):
    # rope is computed per call, so the training window is not the model's
    history = sft(model, task, cfg, gpt_cfg, block_size=16)
    assert [r["step"] for r in history] == [0, cfg.max_steps]


def test_a_vocab_mismatch_is_caught_before_training(task, cfg):
    small = GPTConfig(
        vocab_size=4096,
        block_size=64,
        n_embed=64,
        n_head=4,
        n_layer=2,
        attention="sdpa",
        position="rope",
    )
    with pytest.raises(AssertionError, match="4096 rows"):
        sft(GPT(**asdict(small)), task, cfg, small)


def test_the_checkpoint_records_the_scoreboard(task, model, cfg, gpt_cfg, tmp_path):
    ckpt = tmp_path / "sft.pt"
    sft(model, task, cfg, gpt_cfg, block_size=16, ckpt_path=ckpt, eval_n=8)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["task"] == "reverse"
    assert set(saved) >= {"exact_match", "reward", "stop_rate", "prompt_loss"}


@pytest.mark.slow
def test_masking_opens_a_gap_between_the_two_halves(task, model, cfg, gpt_cfg):
    history = sft(model, task, cfg, gpt_cfg, block_size=32, eval_n=16)
    first, last = history[0], history[-1]
    assert abs(first["comp"] - first["prompt"]) < 0.05  # untrained, both at ln(V)
    assert last["comp"] < first["comp"] - 3.0
    assert last["prompt"] - last["comp"] > 0.5  # only one half is in the loss


def test_selecting_on_comp_keeps_the_lowest_loss_step(
    task, model, cfg, gpt_cfg, tmp_path
):
    ckpt = tmp_path / "comp.pt"
    history = sft(
        model, task, cfg, gpt_cfg, block_size=16, ckpt_path=ckpt, eval_n=8,
        select="comp",
    )  # fmt: skip
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["val_loss"] == min(r["comp"] for r in history)
    assert saved["step"] == min(history, key=lambda r: r["comp"])["step"]


def test_the_best_checkpoint_follows_the_task_not_a_metric_name(
    task, model, cfg, gpt_cfg, tmp_path
):
    class Renamed(Reverse):
        name = "renamed"

        def evaluate(self, model, n=200, **kw):
            scores = super().evaluate(model, n, **kw)
            return {"solved": scores["exact_match"], **scores}

    ckpt = tmp_path / "renamed.pt"
    sft(model, Renamed(task.tok), cfg, gpt_cfg, block_size=16, ckpt_path=ckpt, eval_n=8)
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["task"] == "renamed"
