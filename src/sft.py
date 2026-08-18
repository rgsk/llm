from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

import wandb
from main import (
    CKPT_DIR,
    GPT,
    GPTConfig,
    device,
    get_lr,
    latest_ckpt,
    timestamp,
    tok,
)
from sft_data import build_or_load, iter_records, render, source_path


@dataclass(frozen=True, kw_only=True)
class SFTConfig:
    """Finetuning hyperparameters. The model's own config comes from the base
    checkpoint and must not be changed here."""

    base_ckpt: str
    name: str = "sft"
    # Pretraining ran at 3e-4 with batch 64; this is 2/3 of that at half the
    # batch, roughly sqrt-scaling. 682M fresh instruct tokens is closer to
    # continued pretraining than to a light format adaptation, so the old 5e-5 --
    # sized for 6.5M tokens seen twice, where the risk was dissolving what the
    # base model knew -- is far too timid when nothing repeats.
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup_steps: int = 400  # 2% of max_steps, the convention big_cfg uses
    # 682M train tokens = 41,611 steps per epoch at 32 x 512, so this is 48% of
    # one pass. The old 1600 was ~2 epochs of the 25k-record valid file; here
    # nothing repeats, so the number stopped meaning "to convergence" and started
    # meaning "budget". Raise it until val_comp stops falling.
    max_steps: int = 20_000
    # 32 is where this GPU tops out: 4.07 GB peak, and 64 needs ~7.7 GB of 8.19.
    # Going 16 -> 32 buys only ~5% throughput (105.7 -> 201.2 ms/step for twice
    # the work), so the reason to take it is halved gradient noise, not speed.
    batch_size: int = 32
    # Above the ~1.25 gnorm settles at, so normal steps pass through untouched
    # and this is an outlier guard again rather than a constant 0.8x rescale of
    # every update -- which is what a 1.0 threshold silently was.
    grad_clip: float = 2.0
    eval_interval: int = 500
    eval_iters: int = 100
    seed: int = 1337
    use_wandb: bool = True


def load_ckpt(name: str | Path) -> GPT:
    """Load a pretraining or SFT checkpoint -- both store the same keys, and an
    SFT one additionally carries the SFTConfig and the three-way metrics."""
    path = name if isinstance(name, Path) else CKPT_DIR / name
    saved = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    print(f"loaded {path.name}: step {saved['step']}, val_loss {saved['val_loss']:.4f}")
    if "metrics" in saved:
        m = saved["metrics"]
        print(
            f"  sft metrics: val_comp {m['val_comp']:.3f}  "
            f"val_prompt {m['val_prompt']:.3f}  val_all {m['val_all']:.3f}"
        )
    return model


def _windows(
    packed: dict[str, np.ndarray],
    split: Literal["train", "val"],
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """(x, y, keep): a batch of windows plus a per-token completion flag.

    Windows open at example starts, not at uniform offsets. A uniform offset
    opens mid-story 99% of the time, and 37% of kept targets then have their
    prompt outside the window -- training "continue this story" rather than
    "follow this instruction". Aligning also puts prompts at position 0, which
    is where they sit at inference.

    `keep` is read at i+1 alongside y, not at i: position t predicts token t+1,
    so a target is kept iff the token *being predicted* is a completion token.
    That is the prompt_len-1 offset, expressed so it survives packing.

    y comes back unmasked so eval can score masked and unmasked losses on the
    *same* windows in one forward pass -- the comparison this pipeline exists to
    make. train() masks it immediately.
    """
    ids, mask, starts = (packed[f"{split}_{k}"] for k in ("ids", "mask", "starts"))
    # starts is sorted, so the cutoff is a binary search. The old boolean filter
    # scanned and reallocated all 2.5M starts on every single batch.
    n_ok = np.searchsorted(starts, len(ids) - block_size - 1, side="right")
    ix = starts[rng.integers(n_ok, size=batch_size)]
    x = np.stack([ids[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([ids[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    keep = np.stack([mask[i + 1 : i + 1 + block_size] for i in ix])
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(keep).to(device),
    )


@torch.no_grad()
def estimate_loss(
    model: GPT,
    packed: dict[str, np.ndarray],
    cfg: SFTConfig,
    block_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Three numbers per split from one forward pass: loss on completions only
    (the objective), on the prompt region only, and over everything.

    The prompt number is the argument for masking -- it stays high because the
    base model never saw instruction formatting, and unmasked training would let
    it dominate the gradient.
    """
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        acc: dict[str, list[Tensor]] = {"comp": [], "prompt": [], "all": []}
        for _ in range(cfg.eval_iters):
            x, y, keep = _windows(packed, split, cfg.batch_size, block_size, rng)  # type: ignore[arg-type]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            flat = rearrange(logits, "b t v -> (b t) v").float()
            for key, tgt in (
                ("comp", y.masked_fill(~keep, -100)),
                ("prompt", y.masked_fill(keep, -100)),
                ("all", y),
            ):
                acc[key].append(
                    F.cross_entropy(
                        flat, rearrange(tgt, "b t -> (b t)"), ignore_index=-100
                    )
                )
        for key, vals in acc.items():
            out[f"{split}_{key}"] = torch.stack(vals).mean().item()
    if was_training:
        model.train()
    return out


def train(model: GPT, packed: dict[str, np.ndarray], cfg: SFTConfig, ckpt_path: Path):
    block_size = model.cfg.block_size
    train_rng = np.random.default_rng(cfg.seed)
    eval_rng = np.random.default_rng(cfg.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best_val = float("inf")
    torch.cuda.reset_peak_memory_stats()
    # Run name is the checkpoint stem, so a run and the weights it produced are
    # never ambiguous about which belongs to which.
    wandb.init(
        project="llm",
        name=ckpt_path.stem,
        config=asdict(cfg)
        | {
            "n_params": sum(p.numel() for p in model.parameters()),
            "block_size": block_size,
            "tokens_per_step": cfg.batch_size * block_size,
        },
        settings=wandb.Settings(silent=True),
        mode=None if cfg.use_wandb else "disabled",
    )
    t0 = time.perf_counter()

    for it in range(cfg.max_steps):
        x, y, keep = _windows(packed, "train", cfg.batch_size, block_size, train_rng)
        y = y.masked_fill(~keep, -100)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # finetuning starts from a converged model, so a single bad batch can do
        # real damage; clipping costs nothing and bounds that.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        lr = get_lr(
            it,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            max_lr=cfg.lr,
            min_lr=cfg.min_lr,
        )
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()

        # Logged every step, not just at eval: the per-step loss and gnorm traces
        # are what actually distinguish "LR too hot" from "needs more steps", and
        # eval-interval sampling is too coarse to show it. The .item() syncs cost
        # well under 1% of a 200 ms step.
        log = {"train_batch_loss": loss.item(), "lr": lr, "gnorm": gnorm.item()}

        if it % cfg.eval_interval == 0 or it == cfg.max_steps - 1:
            m = estimate_loss(model, packed, cfg, block_size, eval_rng)
            if m["val_comp"] < best_val:
                best_val = m["val_comp"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "config": asdict(model.cfg),
                        "sft_config": asdict(cfg),
                        "step": it,
                        "val_loss": m["val_comp"],
                        "metrics": m,
                    },
                    ckpt_path,
                )
            log |= m
            print(
                f"step {it:>4} : train_comp {m['train_comp']:.3f}  "
                f"val_comp {m['val_comp']:.3f}  val_prompt {m['val_prompt']:.3f}  "
                f"val_all {m['val_all']:.3f}  lr {lr:.2e}  gnorm {log['gnorm']:.2f}  "
                f"{time.perf_counter() - t0:.0f}s"
            )

        wandb.log(log, step=it)

    _finish_run(cfg, block_size, best_val, t0)


def _finish_run(cfg: SFTConfig, block_size: int, best_val: float, t0: float) -> None:
    wandb.summary["best_val_comp"] = best_val
    wandb.summary["train_time_s"] = time.perf_counter() - t0
    wandb.summary["tokens_seen"] = cfg.max_steps * cfg.batch_size * block_size
    wandb.summary["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    wandb.summary["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    print(
        f"best val_comp {best_val:.4f}  "
        f"{wandb.summary['train_time_s']:.0f}s  "
        f"{wandb.summary['tokens_seen'] / 1e6:.0f}M tokens  "  # type: ignore
        f"peak {wandb.summary['max_memory_allocated_gb']:.2f} GB"
    )
    wandb.finish()


def held_out(n: int = 64) -> list[dict[str, str]]:
    """The first n records of the val source file, as records.

    These are genuinely unseen: val is now the dataset's own valid file, so
    nothing here appears anywhere in the packed train stream. Streaming means
    only n records are parsed, not the whole file.
    """
    return list(islice(iter_records(source_path("valid")), n))


@torch.no_grad()
def generate_sample(
    model: GPT,
    prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> tuple[str, bool]:
    """-> (completion text, whether it emitted <|endoftext|>).

    GPT.generate has no stop condition, so we over-generate and cut. Whether the
    terminator appears at all is the interesting bit: it is the one behaviour
    that packing-with-truncation can fail to teach, since every clipped example
    trains "keep going" and never "stop here".
    """
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt)
    room = model.cfg.block_size - len(ids)
    out = model.generate(
        torch.tensor([ids], device=device),
        max_new_tokens=max(1, min(max_new_tokens, room)),
        temperature=temperature,
        top_p=top_p,
    )
    # Detect the terminator by token id, not by substring: generation can stop
    # mid-"<|endoftext|>" at the max_new_tokens edge, and a text match would
    # then read as "did not terminate".
    gen = out[0, len(ids) :].tolist()
    eot = tok.encode("<|")[0]
    term = eot in gen
    text = tok.decode(gen[: gen.index(eot)] if term else gen)
    if was_training:
        model.train()
    return text, term


def report_samples(
    model: GPT,
    records: list[dict[str, str]],
    n: int = 20,
) -> None:
    """Termination rate and length over held-out prompts, then one full sample."""
    stopped, lengths = 0, []
    n_hits, n_words = 0, 0
    for f in records[:n]:
        prompt = render(f)[0]
        text, term = generate_sample(model, prompt)
        stopped += term
        lengths.append(len(tok.encode(text)))
        words = [w.strip().lower() for w in f.get("Words", "").split(",") if w.strip()]
        hit = sum(bool(re.search(rf"\b{re.escape(w)}", text.lower())) for w in words)
        n_hits += hit
        n_words += len(words)
    print(
        f"  terminated {stopped}/{n}   "
        f"completion tokens: mean {np.mean(lengths):.0f}, max {max(lengths)}"
        f"   word hit rate: {n_hits / n_words * 100:.2f} %"
    )

    prompt, _ = render(records[0])
    print("-" * 72)
    print(prompt + generate_sample(model, prompt)[0])
    print("-" * 72)


def print_sample(model: GPT, records: list[dict[str, str]], idx=0):
    assert idx < len(records)
    prompt, _ = render(records[idx])
    print("-" * 72)
    print(prompt + generate_sample(model, prompt)[0])
    print("-" * 72)


if __name__ == "__main__":
    run_training = 1
    run_report_samples = 0
    # 1 -> short A/B against the logged baseline: 2000 steps at lr 5e-5 with
    # grad_clip 1.0 ended at train_comp 1.093 / val_comp 1.078, so the only
    # question is whether 1.5e-4 with clip 2.0 beats that at the same step count.
    # 0 -> the full 20k budget.
    sanity = 1
    cfg = SFTConfig(base_ckpt="big_2026-08-16_06-45-06.pt", seed=90)
    if sanity:
        cfg = replace(cfg, max_steps=2000, warmup_steps=200, eval_interval=200)
    sft_ckpt = None  # None -> newest sft_*.pt; or a filename to pin one
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    samples = held_out()
    if run_training:
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        report_samples(model, samples)
        packed = build_or_load()

        ckpt_path = CKPT_DIR / f"{cfg.name}_{timestamp()}.pt"
        print(
            f"\n=== finetuning {cfg.max_steps} steps @ lr {cfg.lr:.1e}, "
            f"clip {cfg.grad_clip} -> {ckpt_path.name} ==="
        )
        train(model, packed, cfg, ckpt_path)

        print("\n=== after finetuning ===")
        # report_samples(model, samples)
        saved_ckpt = latest_ckpt(cfg.name)
        ckpt_model = load_ckpt(saved_ckpt)
        report_samples(ckpt_model, samples)
    elif run_report_samples:
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        report_samples(model, samples)
        sft_model = load_ckpt(sft_ckpt or latest_ckpt(cfg.name))
        print("\n=== finetuned model ===")
        report_samples(sft_model, samples)
    else:
        idx = 5
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        print_sample(model, samples, idx=idx)
        sft_model = load_ckpt(sft_ckpt or latest_ckpt(cfg.name))
        print("\n=== finetuned model ===")
        print_sample(sft_model, samples, idx=idx)
