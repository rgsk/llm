"""Full finetune of a HF chat model on accepted Codeforces C++ solutions.

Prompt is the eval's own chat prompt; the target is the solution in a ```cpp
block, then <|im_end|>. Loss counts on the target only.
"""

from __future__ import annotations

import glob
import os
import random
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from codeforces import LANG_IDS, prompt
from lr_schedule import get_lr

IGNORE = -100


def train_problems(max_rating=1000) -> list[dict]:
    """Codeforces rows from the 39 train shards (hub cache), statement + C++ only."""
    hub = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pat = (
        f"{hub}/hub/datasets--deepmind--code_contests/snapshots/*/data/train-*.parquet"
    )
    shards = sorted(glob.glob(pat))
    assert len(shards) == 39, f"expected 39 train shards under {pat}, got {len(shards)}"
    cols = ["name", "source", "cf_rating", "description", "solutions"]
    out = []
    for f in shards:
        for r in pq.read_table(f, columns=cols).to_pylist():
            if r["source"] != 2 or not 0 < r["cf_rating"] <= max_rating:
                continue
            s = r["solutions"]
            cpp = [
                x for x, i in zip(s["solution"], s["language"]) if i == LANG_IDS["cpp"]
            ]
            if cpp:
                out.append(
                    {"name": r["name"], "statement": r["description"], "cpp": cpp}
                )
    return out


def encode(
    tok, statement: str, solution: str, fence=True
) -> tuple[list[int], list[int]]:
    """(ids, labels): labels are IGNORE over the prompt. fence=False keeps a
    model's own full answer (prose and all) as the target."""
    p = tok(prompt(tok, statement, "cpp"), add_special_tokens=False).input_ids
    text = f"```cpp\n{solution.strip()}\n```" if fence else solution.strip()
    a = tok(text, add_special_tokens=False).input_ids
    a = a + [tok.convert_tokens_to_ids("<|im_end|>")]
    return p + a, [IGNORE] * len(p) + a


def examples(tok, problems, k: int, max_len: int, seed: int) -> list[tuple]:
    """Up to k random solutions per problem; drops any over max_len tokens."""
    rng = random.Random(seed)
    out = []
    for p in problems:
        for sol in rng.sample(p["cpp"], min(k, len(p["cpp"]))):
            ids, labels = encode(tok, p["statement"], sol)
            if len(ids) <= max_len:
                out.append((ids, labels))
    return out


def collate(
    rows, pad_id: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad to the longest row; padding is masked out of attention and loss."""
    T = max(len(ids) for ids, _ in rows)
    x = torch.full((len(rows), T), pad_id)
    y = torch.full((len(rows), T), IGNORE)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    for b, (ids, labels) in enumerate(rows):
        x[b, : len(ids)] = torch.tensor(ids)
        y[b, : len(ids)] = torch.tensor(labels)
        att[b, : len(ids)] = 1
    return x.to(device), y.to(device), att.to(device)


def loss_fn(model, x, y, att) -> torch.Tensor:
    """lm_head only where the loss counts: [B, T, 151k] logits over the prompt
    would OOM a 32 GB card at micro-batch 4."""
    h = model.model(input_ids=x, attention_mask=att).last_hidden_state[:, :-1]
    target = y[:, 1:]
    keep = target != IGNORE
    logits = model.lm_head(h[keep])
    return F.cross_entropy(logits.float(), target[keep])


@torch.no_grad()
def val_loss(model, rows, pad_id, micro) -> float:
    model.eval()
    tot = n = 0.0
    for i in range(0, len(rows), micro):
        x, y, att = collate(rows[i : i + micro], pad_id, model.device)
        with torch.autocast("cuda", torch.bfloat16):
            k = (y[:, 1:] != IGNORE).sum().item()
            tot += loss_fn(model, x, y, att).item() * k
        n += k
    model.train()
    return tot / n


def sft(
    model,
    tok,
    train_rows,
    val_rows,
    *,
    lr=1e-5,
    micro=4,
    accum=4,
    warmup=30,
    eval_every=100,
    seed=0,
    log=print,
):
    """One epoch over train_rows, fp32 master weights under bf16 autocast."""
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)
    rows = train_rows[:]
    random.Random(seed).shuffle(rows)
    steps = len(rows) // (micro * accum)
    pad = tok.pad_token_id
    log(f"sft: {len(rows)} examples, {steps} steps of {micro}x{accum}")
    log(f"step 0 val {val_loss(model, val_rows, pad, micro):.4f}")
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = get_lr(
                step, warmup_steps=warmup, max_steps=steps, max_lr=lr, min_lr=lr / 10
            )
        chunk = rows[step * micro * accum : (step + 1) * micro * accum]
        tot = 0.0
        for i in range(accum):
            x, y, att = collate(chunk[i * micro : (i + 1) * micro], pad, model.device)
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_fn(model, x, y, att) / accum
            loss.backward()
            tot += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            dt = (time.time() - t0) / (step + 1)
            log(f"step {step}/{steps} loss {tot:.4f} norm {norm:.2f} {dt:.2f}s/step")
        if (step + 1) % eval_every == 0 or step == steps - 1:
            log(f"step {step + 1} val {val_loss(model, val_rows, pad, micro):.4f}")
    model.config.use_cache = True
    model.eval()
    return model
