from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Int, jaxtyped
from torch import Tensor

from main import (
    CKPT_DIR,
    GPT,
    ROOT,
    GPTConfig,
    device,
    get_lr,
    latest_ckpt,
    timestamp,
    tok,
)

FIELDS = ("Random sentence:", "Features:", "Words:", "Summary:", "Story:")
DATASET = "roneneldan/TinyStoriesInstruct"
VALID_FILE = "TinyStories-Instruct-valid.txt"


def parse(rec: str) -> dict[str, str]:
    """One record's text -> {field name: value}, in source order."""
    fields: dict[str, list[str]] = {}
    cur = None
    for line in rec.split("\n"):
        hit = next((f for f in FIELDS if line.startswith(f)), None)
        if hit:
            cur = hit[:-1]
            fields[cur] = [line[len(hit) :].strip()]
        elif cur:
            fields[cur].append(line.strip())
    return {k: "\n".join(v).strip() for k, v in fields.items()}


def load_records(split: str = "valid") -> list[dict[str, str]]:
    path = hf_hub_download(DATASET, VALID_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    recs = (parse(r.strip()) for r in text.split("<|endoftext|>") if r.strip())
    # need a story to learn, and at least one field to condition on
    return [f for f in recs if f.get("Story") and any(f[k] for k in f if k != "Story")]


def render(f: dict[str, str]) -> tuple[str, str]:
    """Fields -> (prompt, completion). Story is forced last; other fields keep
    their source order, which varies per record on purpose."""
    prompt = "".join(f"{k}: {f[k]}\n" for k in f if k != "Story") + "Story.\n\n"
    return prompt, f["Story"] + "\n<|endoftext|>\n"


def encode_example(f: dict[str, str]) -> tuple[list[int], list[bool]]:
    """-> (token ids, is_completion flag per token). Tokenized separately so the
    prompt/completion boundary is exact, not inferred."""
    p, c = render(f)
    ip, ic = tok.encode(p), tok.encode(c)
    return ip + ic, [False] * len(ip) + [True] * len(ic)


def _pack(
    records: list[dict[str, str]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Records -> one contiguous token stream + its per-token completion mask.

    Shuffled at the record level so packed neighbours aren't corpus-adjacent.
    Examples run end to end with no separator: each already ends with
    <|endoftext|>, which is exactly how documents abut in the pretraining stream.
    """
    ids: list[int] = []
    is_c: list[bool] = []
    for i in rng.permutation(len(records)):
        a, b = encode_example(records[i])
        ids += a
        is_c += b
    return np.array(ids, dtype=np.uint16), np.array(is_c, dtype=bool)


def build_or_load(val_frac: float = 0.05, seed: int = 1337) -> dict[str, np.ndarray]:
    """Tokenize + pack once, cache to disk, memmap thereafter.

    The val slice is taken before packing, so no window straddles the split and
    a val story never appears as context for a train one.
    """
    out = ROOT / "artifacts" / "data" / "tinystories_instruct"
    meta_path = out / "meta.json"
    names = {
        f"{s}_{k}": out / f"{s}.{k}.bin"
        for s in ("train", "val")
        for k in ("ids", "mask")
    }

    if not (meta_path.exists() and all(p.exists() for p in names.values())):
        out.mkdir(parents=True, exist_ok=True)
        records = load_records()
        n_val = int(len(records) * val_frac)
        splits = {"train": records[:-n_val], "val": records[-n_val:]}
        meta: dict = {"seed": seed, "val_frac": val_frac}
        for split, recs in splits.items():
            ids, mask = _pack(recs, np.random.default_rng(seed))
            ids.tofile(names[f"{split}_ids"])
            mask.tofile(names[f"{split}_mask"])
            meta[split] = {
                "n_records": len(recs),
                "n_tokens": int(ids.size),
                "completion_frac": float(mask.mean()),
            }
            print(split, meta[split])
        meta_path.write_text(json.dumps(meta, indent=2))

    packed: dict[str, np.ndarray] = {
        k: np.memmap(p, dtype=np.uint16 if k.endswith("ids") else bool, mode="r")
        for k, p in names.items()
    }
    for split in ("train", "val"):
        # an example begins wherever the mask goes True -> False: a completion
        # ended and the next prompt started. Derived at load rather than cached;
        # it is one pass over the mask and keeps the on-disk format to two arrays.
        m = packed[f"{split}_mask"]
        packed[f"{split}_starts"] = np.concatenate(
            [[0], np.flatnonzero((~m[1:]) & m[:-1]) + 1]
        )
    return packed


@jaxtyped(typechecker=beartype)
def get_batch(
    packed: dict[str, np.ndarray],
    split: Literal["train", "val"],
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[Int[Tensor, "b t"], Int[Tensor, "b t"]]:
    """Same shape as main.get_batch, with prompt positions set to -100.

    The mask is indexed at i+1 alongside y, not at i: position t predicts token
    t+1, so a target is kept iff the token being predicted is a completion token.
    That is the prompt_len-1 offset, expressed so it survives packing.
    """
    ids, mask, starts = (packed[f"{split}_{k}"] for k in ("ids", "mask", "starts"))
    # Sample example starts, not uniform offsets. A uniform offset opens
    # mid-story 99% of the time, and 37% of kept targets then have their prompt
    # outside the window -- training "continue this story" rather than "follow
    # this instruction". Aligning also puts prompts at position 0, which is
    # where they sit at inference.
    ok = starts[starts <= len(ids) - block_size - 1]
    ix = ok[rng.integers(len(ok), size=batch_size)]
    x = np.stack([ids[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([ids[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    keep = np.stack([mask[i + 1 : i + 1 + block_size] for i in ix])
    y[~keep] = -100
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@dataclass(frozen=True, kw_only=True)
class SFTConfig:
    """Finetuning hyperparameters. The model's own config comes from the base
    checkpoint and must not be changed here."""

    base_ckpt: str
    name: str = "sft"
    # 20k pretraining steps already happened; this LR is ~15x smaller so the run
    # adapts the format without dissolving what the base model knows.
    lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 100
    max_steps: int = 1600  # ~2 epochs at 6.5M train tokens, batch 16 x 512
    batch_size: int = 16
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 100
    seed: int = 1337


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
    """(x, y, keep) with y unmasked -- the raw form get_batch masks.

    Kept separate so eval can score masked and unmasked losses on the *same*
    windows in one forward pass, which is the comparison this whole pipeline
    exists to make.
    """
    ids, mask, starts = (packed[f"{split}_{k}"] for k in ("ids", "mask", "starts"))
    ok = starts[starts <= len(ids) - block_size - 1]
    ix = ok[rng.integers(len(ok), size=batch_size)]
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
            print(
                f"step {it:>4} : train_comp {m['train_comp']:.3f}  "
                f"val_comp {m['val_comp']:.3f}  val_prompt {m['val_prompt']:.3f}  "
                f"val_all {m['val_all']:.3f}  lr {lr:.2e}  gnorm {gnorm.item():.2f}  "
                f"{time.perf_counter() - t0:.0f}s"
            )


def held_out(val_frac: float = 0.05) -> list[dict[str, str]]:
    """The same tail slice build_or_load packs as "val", as records.

    Re-parses the source file (~1s) rather than caching, because it is only
    wanted for eyeballing generations.
    """
    records = load_records()
    return records[-int(len(records) * val_frac) :]


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
    run_report_samples = 1
    cfg = SFTConfig(base_ckpt="big_2026-08-16_06-45-06.pt", seed=90)
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
        print(f"\n=== finetuning -> {ckpt_path.name} ===")
        train(model, packed, cfg, ckpt_path)

        print("\n=== after finetuning ===")
        report_samples(model, samples)
    elif run_report_samples:
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        report_samples(model, samples)
        sft_model = load_ckpt(sft_ckpt or latest_ckpt(cfg.name))
        print("\n=== finetuned model ===")
        report_samples(sft_model, samples)
    else:
        idx = 3
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        print_sample(model, samples, idx=idx)
        sft_model = load_ckpt(sft_ckpt or latest_ckpt(cfg.name))
        print("\n=== finetuned model ===")
        print_sample(sft_model, samples, idx=idx)
