from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from gpt import GPT
from gpt_config import GPTConfig
from paths import CKPT_DIR


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_ckpt_path(name: str, ckpt_dir: Path = CKPT_DIR) -> Path:
    """A fresh path for a new run. "scratch" is the exception: it overwrites itself,
    so throwaway runs do not litter the folder."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if name == "scratch":
        return ckpt_dir / "scratch.pt"
    return ckpt_dir / f"{name}_{timestamp()}.pt"


def latest_ckpt(
    prefix: str, ckpt_dir: Path = CKPT_DIR, max_val_loss: float | None = None
) -> Path:
    """The newest checkpoint starting with prefix. %Y-%m-%d_%H-%M-%S is fixed-width
    and big-endian, so lexicographic order == chronological order.

    max_val_loss walks newest -> oldest and takes the first checkpoint at or
    below it, so an aborted run sitting at the init loss does not win just by
    being newest. Anything that will not open, or that never recorded a
    val_loss, is skipped rather than raising -- a half-written file is exactly
    the case this is here to step over.

    Reading val_loss means opening the file, but mmap=True leaves the weights on
    disk and pages in only the pickled metadata: ~2 ms against ~32 ms for a real
    load of a 100 MB checkpoint.
    """
    matches = sorted(ckpt_dir.glob(f"{prefix}*.pt"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint matching {prefix}*.pt in {ckpt_dir}")
    if max_val_loss is None:
        return matches[-1]
    for path in reversed(matches):
        try:
            val_loss = torch.load(path, map_location="cpu", mmap=True).get("val_loss")
        except (RuntimeError, OSError):  # empty, truncated, not a checkpoint
            continue
        if val_loss is not None and val_loss <= max_val_loss:
            return path
    raise FileNotFoundError(
        f"no {prefix}*.pt in {ckpt_dir} with val_loss <= {max_val_loss}"
    )


def save_checkpoint(path: Path, model: GPT, cfg: GPTConfig, **meta) -> None:
    """Weights, plus the config needed to rebuild the architecture that holds them.
    On its own a state_dict is an unlabelled bag of tensors."""
    torch.save({"model": model.state_dict(), "config": asdict(cfg), **meta}, path)


def load_checkpoint(path: Path, device: str = "cpu", **overrides) -> tuple[GPT, dict]:
    """Rebuild the exact architecture from the saved config, then fill in the weights.
    The caller does not need to know the shape in advance -- the file says it.

    overrides patch the config before the model is built. main.py chose its
    attention from a module-level flag and never wrote it down, so its
    checkpoints need attention="fused" supplied by hand."""
    saved = torch.load(path, map_location=device)
    cfg = replace(GPTConfig.from_dict(saved["config"]), **overrides)
    model = GPT(**asdict(cfg)).to(device)
    model.load_state_dict(saved["model"])
    return model, {k: v for k, v in saved.items() if k != "model"}


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_embed=32, n_head=4, n_layer=2)
    model = GPT(**asdict(cfg))
    x = torch.randint(0, cfg.vocab_size, (2, 8))

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # 1. a checkpoint reloads into a model that computes exactly the same thing
        path = generate_ckpt_path("demo", d)
        save_checkpoint(path, model, cfg, step=42, val_loss=1.5, bpc=0.9)
        loaded, meta = load_checkpoint(path)
        model.eval(), loaded.eval()
        assert (model(x) - loaded(x)).abs().max() == 0

        # 2. the caller never states the shape -- the file carries it
        assert loaded.block_size == 16 and loaded.n_layer == 2
        assert sum(p.numel() for p in loaded.parameters()) == sum(
            p.numel() for p in model.parameters()
        )

        # 3. weight tying survives the round trip
        assert loaded.lm_head.weight is loaded.token_embedding_table.weight

        # 4. metadata comes back, without the weights riding along
        assert meta["step"] == 42 and meta["val_loss"] == 1.5 and meta["bpc"] == 0.9
        assert "model" not in meta
        assert GPTConfig.from_dict(meta["config"]) == cfg

        # 5. named runs get a unique file, "scratch" deliberately reuses one
        assert generate_ckpt_path("scratch", d) == generate_ckpt_path("scratch", d)
        assert generate_ckpt_path("big", d).name.startswith("big_2")

        # 6. latest_ckpt picks the newest by name, and respects the prefix
        for stamp in [
            "2026-01-01_00-00-00",
            "2026-08-29_12-00-00",
            "2026-03-05_09-00-00",
        ]:
            (d / f"big_{stamp}.pt").touch()
        (d / "small_2099-01-01_00-00-00.pt").touch()
        assert latest_ckpt("big", d).name == "big_2026-08-29_12-00-00.pt"
        assert latest_ckpt("small", d).name == "small_2099-01-01_00-00-00.pt"

        # 7. a missing checkpoint says so, instead of returning None
        try:
            latest_ckpt("nothing", d)
            raise SystemExit("should have failed")
        except FileNotFoundError as e:
            assert "nothing*.pt" in str(e)

        # 8. max_val_loss: newest wins only among checkpoints that trained.
        #    "run" writes three -- a good one, then a better one, then a fresh
        #    start that aborted at the init loss
        for stamp, vl in [
            ("2026-05-01_00-00-00", 4.0),
            ("2026-05-02_00-00-00", 3.1),
            ("2026-05-03_00-00-00", 8.3),  # aborted, and the newest
        ]:
            save_checkpoint(d / f"run_{stamp}.pt", model, cfg, step=1, val_loss=vl)
        assert latest_ckpt("run", d).name == "run_2026-05-03_00-00-00.pt"  # newest
        assert latest_ckpt("run", d, 6.0).name == "run_2026-05-02_00-00-00.pt"
        assert latest_ckpt("run", d, 3.5).name == "run_2026-05-02_00-00-00.pt"
        assert latest_ckpt("run", d, 4.0).name == "run_2026-05-02_00-00-00.pt"

        #    the threshold can exclude everything, and that is not silent
        try:
            latest_ckpt("run", d, 1.0)
            raise SystemExit("should have failed")
        except FileNotFoundError as e:
            assert "val_loss <= 1.0" in str(e)

        #    files that will not open are stepped over, not fatal: test 6 left
        #    three empty big_*.pt behind, so a real one has to be found past them
        save_checkpoint(d / "big_2026-01-02_00-00-00.pt", model, cfg, val_loss=2.0)
        assert latest_ckpt("big", d, 5.0).name == "big_2026-01-02_00-00-00.pt"

        #    and so is a checkpoint that never recorded one
        save_checkpoint(d / "noloss_2026-01-01_00-00-00.pt", model, cfg, step=1)
        try:
            latest_ckpt("noloss", d, 5.0)
            raise SystemExit("should have failed")
        except FileNotFoundError as e:
            assert "val_loss" in str(e)

        # 9. the resume payload rides along in **meta: optimizer moments and the
        #    position of the batch stream, back out of load_checkpoint untouched
        from adamw import AdamW, decay_groups

        opt = AdamW(decay_groups(model, 0.1), lr=1e-3)
        model(x).sum().backward()
        opt.step()
        gen = torch.Generator().manual_seed(1337)
        torch.randint(0, 10, (5,), generator=gen)

        rpath = generate_ckpt_path("resume", d)
        save_checkpoint(
            rpath, model, cfg, step=7, opt=opt.state_dict(), train_gen=gen.get_state()
        )
        loaded, meta = load_checkpoint(rpath)

        opt2 = AdamW(decay_groups(loaded, 0.1), lr=1e-3)
        opt2.load_state_dict(meta["opt"])
        assert opt2.t == opt.t == 1
        assert len(opt2.state) == len(opt.state)

        # the generator picks up mid-stream, rather than restarting at seed 1337
        g2 = torch.Generator()
        g2.set_state(meta["train_gen"])
        assert torch.equal(
            torch.randint(0, 10, (5,), generator=g2),
            torch.randint(0, 10, (5,), generator=gen),
        )

    print("ok")
