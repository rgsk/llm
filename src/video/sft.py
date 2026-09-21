"""Supervised finetuning: train.py's loop, fed by a Task instead of a corpus.

Three things change. Batches come from windows() so every row starts at an
example; the loss is masked to the completion; and attention gets a document
mask, because a 1024-wide window over 18-token examples packs ~70 of them.

It never names a dataset -- swapping reverse for instruct is one argument.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from adamw import AdamW, decay_groups
from amp import autocast
from checkpoint import save_checkpoint
from clip_grad_norm import clip_grad_norm
from cross_entropy import cross_entropy
from gpt import GPT
from gpt_config import GPTConfig
from logger import Run
from lr_schedule import get_lr
from mask import doc_block_mask, doc_ids
from task import Packed, Task, check_vocab, masked_targets, windows
from train_config import TrainConfig


def batch(
    packed: Packed,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str,
    doc_mask: bool,
):
    """One training batch: windows plus the mask that keeps its examples apart."""
    x, y, keep = windows(packed, batch_size, block_size, rng, device)
    bm = doc_block_mask(doc_ids(x, packed.eot_id)) if doc_mask else None
    return x, y, keep, bm


@torch.no_grad()
def split_losses(
    model: GPT,
    packed: Packed,
    batch_size: int,
    block_size: int,
    iters: int,
    rng: np.random.Generator,
    device: str = "cpu",
    amp: bool = True,
    doc_mask: bool = True,
) -> tuple[float, float]:
    """(completion, prompt) loss. Masking is supposed to move them in opposite
    directions -- completion down, prompt up -- so reporting one number hides
    half of what SFT does."""
    was_training = model.training
    model.eval()
    comp, prompt = [], []
    for _ in range(iters):
        x, y, keep = windows(packed, batch_size, block_size, rng, device)
        # no_grad, so flex runs here even on cpu -- only its backward is missing
        bm = doc_block_mask(doc_ids(x, packed.eot_id)) if doc_mask else None
        with autocast(device, amp):
            logits = model(x, block_mask=bm).reshape(-1, model.lm_head.weight.size(0))
        flat, k = y.reshape(-1), keep.reshape(-1)
        comp.append(cross_entropy(logits[k], flat[k]).item())
        prompt.append(cross_entropy(logits[~k], flat[~k]).item())
    if was_training:
        model.train()
    return float(np.mean(comp)), float(np.mean(prompt))


def sft(
    model: GPT,
    task: Task,
    cfg: TrainConfig,
    gpt_cfg: GPTConfig,
    block_size: int | None = None,
    device: str = "cpu",
    ckpt_path: Path | None = None,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_n: int = 200,
) -> list[dict]:
    """block_size is the training window, not the model's: rope is computed on
    demand, so a 1024-block model finetunes fine on 256-wide windows."""
    T = gpt_cfg.block_size if block_size is None else block_size
    assert T <= gpt_cfg.block_size, f"window {T} > model block_size"
    V = gpt_cfg.vocab_size

    train_packed, val_packed = task.build("train"), task.build("val")
    check_vocab(train_packed, model)
    # flex has no cpu backward, so cpu trains without document isolation
    doc_mask = device.startswith("cuda") and gpt_cfg.attention == "flex"

    run = Run(
        cfg.name,
        # window, because gpt_cfg.block_size is the model's, not the batch's
        asdict(cfg) | asdict(gpt_cfg) | {"task": task.name, "window": T},
        online=cfg.use_wandb,
    )
    torch.manual_seed(cfg.seed)
    train_rng = np.random.default_rng(cfg.seed)
    opt = AdamW(decay_groups(model, weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
    model.train()

    best = (-1.0, -float("inf"))  # (exact_match, -completion loss)
    history: list[dict] = []
    t0 = time.perf_counter()
    gnorm = torch.tensor(float("nan"))

    def evaluate(it: int, lr: float) -> None:
        nonlocal best
        comp, prompt = split_losses(
            model, val_packed, cfg.batch_size, T, cfg.eval_iters,
            np.random.default_rng(cfg.seed + 1),  # same val windows every eval
            device, cfg.amp, doc_mask,
        )  # fmt: skip
        scores = task.evaluate(model, eval_n)
        row = {
            "step": it,
            "comp": comp,
            "prompt": prompt,
            **scores,
            "lr": lr,
            "gnorm": gnorm.item(),
            "secs": time.perf_counter() - t0,
        }
        history.append(row)
        print(
            f"step {it:>5}  comp {comp:7.4f}  prompt {prompt:7.4f}  "
            + "  ".join(f"{k} {v:.3f}" for k, v in scores.items())
            + f"  lr {lr:.2e}  |g| {gnorm:5.2f}  {row['secs']:6.1f}s"
        )
        run.log(row, step=it)
        # the scoreboard is the objective; completion loss only breaks ties
        score = (scores["exact_match"], -comp)
        if score > best and ckpt_path is not None:
            save_checkpoint(
                ckpt_path, model, gpt_cfg, step=it, val_loss=comp,
                prompt_loss=prompt, **scores, task=task.name,
            )  # fmt: skip
        best = max(best, score)

    fwd = torch.compile(model) if cfg.use_compile else model

    for it in range(cfg.max_steps):
        lr = get_lr(
            it,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            max_lr=cfg.lr,
            min_lr=cfg.min_lr,
        )
        opt.lr = lr
        if it % cfg.eval_interval == 0:
            evaluate(it, lr)

        opt.zero_grad()
        for _ in range(cfg.grad_accum_steps):
            x, y, keep, bm = batch(
                train_packed, cfg.batch_size, T, train_rng, device, doc_mask
            )
            with autocast(device, cfg.amp):
                logits = fwd(x, block_mask=bm)
                loss = cross_entropy(
                    logits.reshape(-1, V),
                    masked_targets(y, keep).reshape(-1),
                    ignore_index=-100,
                )
            (loss / cfg.grad_accum_steps).backward()
        gnorm = clip_grad_norm(model.parameters(), grad_clip)
        opt.step()

    evaluate(cfg.max_steps, opt.lr)
    run.summary(best_exact_match=best[0], total_time_s=time.perf_counter() - t0)
    run.finish()
    return history


if __name__ == "__main__":
    # assertions are in test_sft.py; this trains a small model on reverse
    from reverse import Reverse

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    task = Reverse(n_train=20_000, n_val=2_000, lmin=3, lmax=6)
    gpt_cfg = GPTConfig(
        vocab_size=task.tok.vocab_size,
        block_size=128,
        n_embed=192,
        n_head=6,
        n_layer=4,
        attention="flex" if device == "cuda" else "sdpa",
        n_kv_head=2 if device == "cuda" else None,
        norm="rms",
        ffn="gated",
        position="rope",
    )
    cfg = TrainConfig(
        batch_size=32,
        max_steps=400,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=40,
        eval_interval=100,
        eval_iters=20,
        name="sft_smoke",
        use_compile=False,
        use_wandb=False,
    )
    model = GPT(**asdict(gpt_cfg)).to(device)
    print(f"{device}  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    history = sft(model, task, cfg, gpt_cfg, device=device, eval_n=100)
    print(
        f"\nexact match {history[0]['exact_match']:.2f} -> {history[-1]['exact_match']:.2f}"
    )
