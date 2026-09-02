import time
from dataclasses import asdict
from pathlib import Path

import torch
from adamw import AdamW, decay_groups
from amp import autocast
from checkpoint import save_checkpoint
from clip_grad_norm import clip_grad_norm
from cross_entropy import cross_entropy
from dataset import BinDataset, get_batch, meta
from evaluate import bits_per_char, estimate_loss, full_loss
from gpt import GPT
from gpt_config import GPTConfig
from logger import Run
from lr_schedule import get_lr
from train_config import TrainConfig


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
    resume: dict | None = None,
) -> list[dict]:
    torch.manual_seed(cfg.seed)
    run = Run(cfg.name, asdict(cfg) | asdict(gpt_cfg), enabled=cfg.use_wandb)
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

    start_step = 0
    if resume is not None:
        # load_checkpoint's metadata: the weights are already in `model`, this is
        # everything else the loop carries
        need = {"opt", "train_gen", "eval_gen", "step", "val_loss"}
        assert need <= resume.keys(), f"not resumable: no {need - resume.keys()}"
        opt.load_state_dict(resume["opt"])
        train_gen.set_state(resume["train_gen"].cpu())  # map_location put them on gpu
        eval_gen.set_state(resume["eval_gen"].cpu())
        best_val = resume["val_loss"]
        # not step + 1: evaluate() saves before the update at that step
        start_step = resume["step"]
        print(f"resuming at step {start_step}, val {best_val:.4f}")

    def evaluate(it: int, lr: float) -> None:
        nonlocal best_val
        eval_state = eval_gen.get_state()  # before the draws, so a resume replays them
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
        run.log(history[-1], step=it)
        if va < best_val and ckpt_path is not None:
            best_val = va
            save_checkpoint(
                ckpt_path,
                model,
                gpt_cfg,
                step=it,
                val_loss=va,
                bpc=bpc,
                opt=opt.state_dict(),
                train_gen=train_gen.get_state(),
                eval_gen=eval_state,
            )

    # compile the forward, but keep `model` for everything else: zero_grad,
    # clipping and state_dict all want the real module, and the wrapper shares
    # its parameters anyway
    fwd = torch.compile(model) if cfg.use_compile else model

    for it in range(start_step, cfg.max_steps):
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

        opt.zero_grad()
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(train_ds, cfg.batch_size, T, train_gen)
            x, y = x.to(device), y.to(device)
            with autocast(device, cfg.amp):
                logits = fwd(x)
                loss = cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss = loss / cfg.grad_accum_steps
            loss.backward()
        gnorm = clip_grad_norm(model.parameters(), grad_clip)
        opt.step()

    evaluate(cfg.max_steps, opt.lr)  # final model, after every update

    run.summary(
        best_val=best_val,
        total_time_s=time.perf_counter() - t0,
        peak_memory_gb=(
            torch.cuda.max_memory_allocated() / 1e9
            if device.startswith("cuda")
            else 0.0
        ),
    )
    run.finish()
    return history


if __name__ == "__main__":
    import math
    import tempfile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    gpt_cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=128,
        n_embed=192,
        n_head=6,
        n_layer=4,
        attention="sdpa",
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
        use_compile=True,
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
        assert abs(history[0]["val"] - math.log(gpt_cfg.vocab_size)) < 0.1
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

    # 6. a run that dies mid-step resumes into exactly the run it would have been
    from dataclasses import replace

    from checkpoint import load_checkpoint

    r_gpt_cfg = replace(gpt_cfg, block_size=64, n_embed=96, n_head=4, n_layer=2)
    r_cfg = replace(
        cfg,
        batch_size=16,
        max_steps=60,
        eval_interval=20,
        eval_iters=5,
        name="resume",
        use_compile=False,
    )

    def fresh_model():
        torch.manual_seed(0)
        return GPT(**asdict(r_gpt_cfg)).to(device)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = fresh_model()
        hist_a = train(
            a, r_cfg, r_gpt_cfg, train_ds, val_ds, device=device, ckpt_path=d / "a.pt"
        )

        # the same run, killed at the start of step 40 -- just after evaluate(40)
        # wrote its checkpoint, which is therefore all that survives
        real_get_batch, calls = get_batch, [0]

        def flaky(*args, **kw):
            calls[0] += 1
            if calls[0] > 40:
                raise RuntimeError("simulated crash")
            return real_get_batch(*args, **kw)

        get_batch = flaky
        b = fresh_model()
        try:
            train(
                b,
                r_cfg,
                r_gpt_cfg,
                train_ds,
                val_ds,
                device=device,
                ckpt_path=d / "b.pt",
            )
            raise SystemExit("should have crashed")
        except RuntimeError:
            pass
        get_batch = real_get_batch

        # pick the wreckage back up -- one read gives the model and its state
        c, _, m = load_checkpoint(d / "b.pt", device=device)
        assert m["step"] == 40
        hist_c = train(
            c,
            r_cfg,
            r_gpt_cfg,
            train_ds,
            val_ds,
            device=device,
            ckpt_path=d / "c.pt",
            resume=m,
        )

        # the resumed leg reproduces the uninterrupted run's tail row for row,
        # starting by re-running eval 40 on the very same windows
        tail = [h for h in hist_a if h["step"] >= 40]
        assert [h["step"] for h in tail] == [h["step"] for h in hist_c]
        for ra, rc in zip(tail, hist_c):
            for k in ("train", "val", "bpc", "lr"):
                assert ra[k] == rc[k], (rc["step"], k, ra[k], rc[k])

        # ... and lands on the same weights, bit for bit
        assert all(
            (p - q).abs().max() == 0 for p, q in zip(a.parameters(), c.parameters())
        )

        # the control: weights alone, no moments and no RNG, ends up somewhere else
        n, _, _ = load_checkpoint(d / "b.pt", device=device)
        train(
            n, r_cfg, r_gpt_cfg, train_ds, val_ds, device=device, ckpt_path=d / "n.pt"
        )
        assert (
            max((p - q).abs().max() for p, q in zip(a.parameters(), n.parameters()))
            > 1e-4
        )

    train_ds.close()
    val_ds.close()
    print("ok")
