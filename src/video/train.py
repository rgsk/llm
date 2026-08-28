import time
from dataclasses import asdict
from pathlib import Path

import torch
from adamw import AdamW, decay_groups
from checkpoint import save_checkpoint
from clip_grad_norm import clip_grad_norm
from cross_entropy import cross_entropy
from evaluate import bits_per_char, estimate_loss, full_loss
from gpt import GPT
from gpt_config import GPTConfig
from lr_schedule import get_lr
from train_config import TrainConfig

from data import BinDataset, get_batch, meta


def train(
    model: GPT,
    cfg: TrainConfig,
    gpt_cfg: GPTConfig,
    train_ds: BinDataset,
    val_ds: BinDataset,
    device: str = "cpu",
    ckpt_path: Path | None = None,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
) -> list[dict]:
    torch.manual_seed(cfg.seed)
    train_gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 1)

    opt = AdamW(decay_groups(model, weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
    model.train()

    T = gpt_cfg.block_size
    V = gpt_cfg.vocab_size
    best_val = float("inf")
    history: list[dict] = []
    t0 = time.perf_counter()
    gnorm = torch.tensor(float("nan"))  # no gradient exists before the first step

    def evaluate(it: int, lr: float) -> None:
        nonlocal best_val
        tr = estimate_loss(
            model, train_ds, cfg.batch_size, T, cfg.eval_iters, eval_gen, device
        )
        va = full_loss(model, val_ds, cfg.batch_size, T, device, max_windows=4000)
        bpc = bits_per_char(va)
        history.append(
            {
                "step": it,
                "train": tr,
                "val": va,
                "bpc": bpc,
                "lr": lr,
                "gnorm": gnorm.item(),
                "secs": time.perf_counter() - t0,
            }
        )
        print(
            f"step {it:>5}  train {tr:.4f}  val {va:.4f}  bpc {bpc:.3f}  "
            f"lr {lr:.2e}  |g| {gnorm:6.2f}  {time.perf_counter() - t0:6.1f}s"
        )
        if va < best_val and ckpt_path is not None:
            best_val = va
            save_checkpoint(ckpt_path, model, gpt_cfg, step=it, val_loss=va, bpc=bpc)

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
            evaluate(it, lr)  # model state after exactly `it` updates

        x, y = get_batch(train_ds, cfg.batch_size, T, train_gen)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        gnorm = clip_grad_norm(model.parameters(), grad_clip)
        opt.step()

    evaluate(cfg.max_steps, opt.lr)  # final model, after every update

    return history


if __name__ == "__main__":
    import math
    import tempfile

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpt_cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=128,
        n_embed=192,
        n_head=6,
        n_layer=4,
    )
    cfg = TrainConfig(
        batch_size=32,
        max_steps=150,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=20,
        eval_interval=50,
        eval_iters=20,
        name="smoke",
    )

    torch.manual_seed(cfg.seed)
    model = GPT(**asdict(gpt_cfg)).to(device)
    print(
        f"device {device}   params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    train_ds, val_ds = BinDataset("train"), BinDataset("val")
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "smoke.pt"
        history = train(
            model, cfg, gpt_cfg, train_ds, val_ds, device=device, ckpt_path=ckpt
        )

        # 1. step 0 is the UNTRAINED model, so it sits at ln(V), plus the small
        #    logit-spread term (~sigma^2/2, sigma = 0.02*sqrt(n_embed) = 0.28 here)
        assert abs(history[0]["val"] - math.log(gpt_cfg.vocab_size)) < 0.05
        assert history[0]["step"] == 0
        assert math.isnan(history[0]["gnorm"])  # no gradient has been taken yet
        assert history[-1]["step"] == cfg.max_steps  # final row is the final model
        assert history[-1]["val"] < history[0]["val"] - 3.0, history[-1]["val"]

        # 2. val improves monotonically over this short run
        vals = [h["val"] for h in history]
        assert vals == sorted(vals, reverse=True), vals

        # 3. the best checkpoint was written and reloads into a fresh model
        saved = torch.load(ckpt, map_location=device)
        assert saved["val_loss"] == min(vals)
        assert GPTConfig(**saved["config"]) == gpt_cfg
        fresh = GPT(**saved["config"]).to(device)
        fresh.load_state_dict(saved["model"])
        assert (
            full_loss(
                fresh,
                val_ds,
                cfg.batch_size,
                gpt_cfg.block_size,
                device,
                max_windows=4000,
            )
            == saved["val_loss"]
        )

    # 4. the model is left in train mode, not eval
    assert model.training

    # 5. the lr schedule was followed
    assert history[0]["lr"] == cfg.lr / cfg.warmup_steps  # one warmup unit
    # the last step is max_steps - 1, and the cosine only hits min_lr at
    # max_steps exactly -- so we land just above it, never on it
    assert cfg.min_lr < history[-1]["lr"] < cfg.min_lr * 1.01

    train_ds.close()
    val_ds.close()
    print("ok")
