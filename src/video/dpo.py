"""DPO: push the better of two samples up, relative to a frozen reference.

    loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))

Each term is a completion's summed log-prob under the policy (pi) or the SFT
model it started from (ref). Pairs cost no labels: sample K per prompt from the
SFT model, score them with task.reward, keep best against worst.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from adamw import AdamW, decay_groups
from amp import autocast
from checkpoint import save_checkpoint
from clip_grad_norm import clip_grad_norm
from generate import generate, length_groups
from gpt import GPT
from gpt_config import GPTConfig
from logger import Run
from lr_schedule import get_lr
from train_config import TrainConfig


@dataclass
class Pair:
    prompt: list[int]
    chosen: list[int]  # completion ids, eot included when it stopped
    rejected: list[int]
    r_chosen: float
    r_rejected: float


def pick(rewards: list[float]) -> tuple[int, int] | None:
    """-> (best, worst) index, or None when every sample scored the same."""
    best, worst = int(np.argmax(rewards)), int(np.argmin(rewards))
    return None if rewards[best] == rewards[worst] else (best, worst)


def cut(ids: list[int], eot: int) -> list[int]:
    """A sampled row up to and including its first eot."""
    return ids[: ids.index(eot) + 1] if eot in ids else ids


@torch.no_grad()
def make_pairs(
    model: GPT,
    task,
    prompts: list[str],
    k: int = 8,
    max_new_tokens: int = 384,
    temperature: float = 1.0,
    seed: int = 0,
    group: int = 16,
) -> list[Pair]:
    """k samples for each of up to `group` equal-length prompts per generate
    call. A decode step costs about the same at 8 rows as at 128, so rows are
    nearly free and calls are what cost."""
    device = next(model.parameters()).device
    model.eval()
    enc = [task.tok.encode(p) for p in prompts]
    found: dict[int, Pair] = {}
    for gi in length_groups([len(e) for e in enc], group):
        n = len(enc[gi[0]])
        x = torch.tensor([enc[i] for i in gi for _ in range(k)], device=device)
        g = torch.Generator(device=device).manual_seed(seed + gi[0])
        room = model.block_size + 1 - n  # the cache holds block_size tokens
        out = generate(
            model, x, min(max_new_tokens, room), temperature=temperature,
            use_cache=True, generator=g, stop=task.eot,
        )  # fmt: skip
        for j, i in enumerate(gi):
            comps = [cut(r[n:].tolist(), task.eot) for r in out[j * k : (j + 1) * k]]
            rewards = [task.reward(prompts[i], task.tok.decode(c)) for c in comps]
            if bw := pick(rewards):
                b, w = bw
                found[i] = Pair(enc[i], comps[b], comps[w], rewards[b], rewards[w])
    return [found[i] for i in sorted(found)]


def batch_logps(model: GPT, rows: list[tuple[list[int], list[int]]]) -> Tensor:
    """(prompt, completion) rows -> summed log-prob of each completion, [N].
    Right-padded: under a causal mask a pad can only follow what it pads."""
    device = next(model.parameters()).device
    T = max(len(p) + len(c) for p, c in rows)
    x = torch.zeros(len(rows), T, dtype=torch.long, device=device)
    score = torch.zeros(len(rows), T, dtype=torch.bool, device=device)
    for i, (p, c) in enumerate(rows):
        x[i, : len(p) + len(c)] = torch.tensor(p + c, device=device)
        score[i, len(p) : len(p) + len(c)] = True
    logits = model(x[:, :-1]).float()
    lp = logits.log_softmax(-1).gather(-1, x[:, 1:, None]).squeeze(-1)
    return (lp * score[:, 1:]).sum(-1)


@torch.no_grad()
def pair_logps(
    model: GPT, pairs: list[Pair], batch_size: int = 16, amp: bool = True
) -> tuple[Tensor, Tensor]:
    """(chosen, rejected) log-probs for every pair -- the reference, computed once."""
    device = str(next(model.parameters()).device)
    model.eval()
    c, r = [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        with autocast(device, amp):
            c.append(batch_logps(model, [(p.prompt, p.chosen) for p in chunk]))
            r.append(batch_logps(model, [(p.prompt, p.rejected) for p in chunk]))
    return torch.cat(c), torch.cat(r)


def dpo_loss(
    pi_c: Tensor, pi_r: Tensor, ref_c: Tensor, ref_r: Tensor, beta: float
) -> tuple[Tensor, Tensor]:
    """-> (loss, z). z is the implicit reward margin; -log sigmoid(z) = log(1 + e^-z)."""
    z = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    return torch.logaddexp(torch.zeros_like(z), -z).mean(), z


def dpo(
    model: GPT,
    pairs: list[Pair],
    val_pairs: list[Pair],
    task,
    cfg: TrainConfig,
    gpt_cfg: GPTConfig,
    beta: float = 0.1,
    device: str = "cpu",
    ckpt_path: Path | None = None,
    grad_clip: float = 1.0,
    eval_n: int = 100,
    max_new_tokens: int = 384,
) -> list[dict]:
    """model starts as the SFT checkpoint, which is also the reference: its
    log-probs are taken before the first step and never recomputed."""
    ref_c, ref_r = pair_logps(model, pairs, amp=cfg.amp)
    vref_c, vref_r = pair_logps(model, val_pairs, amp=cfg.amp)

    run = Run(
        cfg.name,
        asdict(cfg) | asdict(gpt_cfg) | {"task": task.name, "beta": beta},
        online=cfg.use_wandb,
    )
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = AdamW(decay_groups(model, 0.0), lr=cfg.lr, betas=(0.9, 0.95))
    history: list[dict] = []
    t0 = time.perf_counter()
    gnorm = torch.tensor(float("nan"))

    def evaluate(it: int, lr: float) -> None:
        pi_c, pi_r = pair_logps(model, val_pairs, amp=cfg.amp)
        loss, z = dpo_loss(pi_c, pi_r, vref_c, vref_r, beta)
        # sampled, not greedy: the pairs were drawn at temperature 1.0
        scores = task.evaluate(
            model, eval_n, max_new_tokens=max_new_tokens, temperature=1.0
        )
        row = {
            "step": it,
            "val_loss": loss.item(),
            "val_acc": (z > 0).float().mean().item(),
            # both can fall while the gap grows -- watch these, not just the loss
            "d_chosen": (pi_c - vref_c).mean().item(),
            "d_rejected": (pi_r - vref_r).mean().item(),
            **scores,
            "lr": lr,
            "gnorm": gnorm.item(),
            "secs": time.perf_counter() - t0,
        }
        history.append(row)
        print(
            f"step {it:>4}  loss {row['val_loss']:.4f}  acc {row['val_acc']:.3f}  "
            f"dc {row['d_chosen']:+7.2f}  dr {row['d_rejected']:+7.2f}  "
            + "  ".join(f"{k} {v:.3f}" for k, v in scores.items())
            + f"  |g| {gnorm:5.2f}  {row['secs']:6.1f}s"
        )
        run.log(row, step=it)
        model.train()

    B = cfg.batch_size
    model.train()
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
            idx = rng.choice(len(pairs), B, replace=False)
            chunk = [pairs[i] for i in idx]
            rows = [(p.prompt, p.chosen) for p in chunk]
            rows += [(p.prompt, p.rejected) for p in chunk]
            with autocast(device, cfg.amp):
                lp = batch_logps(model, rows)
            j = torch.as_tensor(idx, device=ref_c.device)
            loss, _ = dpo_loss(lp[:B], lp[B:], ref_c[j], ref_r[j], beta)
            (loss / cfg.grad_accum_steps).backward()
        gnorm = clip_grad_norm(model.parameters(), grad_clip)
        opt.step()

    evaluate(cfg.max_steps, opt.lr)
    if ckpt_path is not None:
        last = history[-1]
        save_checkpoint(
            ckpt_path, model, gpt_cfg, task=task.name, beta=beta,
            **{k: v for k, v in last.items() if k not in ("lr", "gnorm", "secs")},
        )  # fmt: skip
    run.finish()
    return history


if __name__ == "__main__":
    # assertions are in test_dpo.py; this prints the loss at the three points
    # that define it
    zero = torch.zeros(1)
    for gap in (-2.0, 0.0, 2.0):
        loss, z = dpo_loss(torch.tensor([gap]), zero, zero, zero, beta=1.0)
        print(f"policy prefers chosen by {gap:+.0f} nats -> loss {loss.item():.4f}")
