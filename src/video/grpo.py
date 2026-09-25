"""GRPO on a HF chat model, reward = the program passes every test.

Per step: sample a group of completions per problem, grade them, and score
each against its own group: advantage = (r - group mean) / group std. A group
that all passes or all fails has advantage 0 and teaches nothing, so it is
dropped. One gradient step per batch, so the policy ratio is 1 and the loss is
plain REINFORCE with the group as the baseline. No KL term.
"""

from __future__ import annotations

import random
import time

import torch
from torch import Tensor

from cf_sft import IGNORE, collate
from codeforces import Problem, judge_all, prompt


def advantages(rewards: Tensor, group: int, eps: float = 1e-4) -> Tensor:
    """Each reward minus its group's mean, over the group's std."""
    r = rewards.view(-1, group)
    a = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)
    return a.flatten()


def trim(ids: list[int], stop: set[int]) -> list[int]:
    """Cut a completion after its first stop token (kept: it ends the turn)."""
    for i, t in enumerate(ids):
        if t in stop:
            return ids[: i + 1]
    return ids


@torch.no_grad()
def rollout(
    model, tok, statements, group, *, max_new=1024, temperature=0.8, batch=32
) -> list[tuple[list[int], list[int]]]:
    """`group` samples per statement, as (prompt ids, completion ids)."""
    tok.padding_side = "left"
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    texts = [prompt(tok, s, "cpp") for s in statements for _ in range(group)]
    out = []
    model.eval()
    model.config.use_cache = True
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], return_tensors="pt", padding=True)
        enc = enc.to(model.device)
        with torch.autocast("cuda", torch.bfloat16):
            ids = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id,
            )
        P = enc.input_ids.shape[1]
        for row, keep, full in zip(enc.input_ids, enc.attention_mask, ids):
            out.append((row[keep.bool()].tolist(), trim(full[P:].tolist(), stop)))
    return out


def pg_loss(model, x, y, att, adv, n_tokens: int, temperature: float) -> Tensor:
    """-sum(A * log p(token)) over completion tokens, / n_tokens in the batch.
    Logits are divided by the sampling temperature so log p is the policy's."""
    h = model.model(input_ids=x, attention_mask=att).last_hidden_state[:, :-1]
    target = y[:, 1:]
    keep = target != IGNORE
    logits = model.lm_head(h[keep]).float() / temperature
    logp = logits.log_softmax(-1).gather(1, target[keep][:, None]).squeeze(1)
    w = adv[:, None].expand_as(target)[keep]
    return -(w * logp).sum() / n_tokens


def grpo(
    model,
    tok,
    problems: list[Problem],
    *,
    steps: int,
    per_step=8,
    group=8,
    lr=1e-6,
    micro=2,
    max_new=1024,
    temperature=0.8,
    gen_batch=32,
    seed=0,
    log=print,
):
    """fp32 master weights; rollouts and the backward run under bf16 autocast."""
    rng = random.Random(seed)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)
    pad = tok.pad_token_id
    history = []
    for step in range(steps):
        t0 = time.time()
        ps = rng.sample(problems, per_step)
        pairs = rollout(
            model,
            tok,
            [p.statement for p in ps],
            group,
            max_new=max_new,
            temperature=temperature,
            batch=gen_batch,
        )
        t1 = time.time()
        texts = [tok.decode(c, skip_special_tokens=True) for _, c in pairs]
        graded = judge_all([p for p in ps for _ in range(group)], texts, "cpp")
        r = torch.tensor([float(g["status"] == "accepted") for g in graded])
        adv = advantages(r, group)
        t2 = time.time()

        rows = [
            ((p + c, [IGNORE] * len(p) + c), a)
            for (p, c), a in zip(pairs, adv.tolist())
            if a != 0 and c
        ]
        mixed = int((r.view(-1, group).std(1) > 0).sum())
        stats = {
            "step": step,
            "reward": r.mean().item(),
            "mixed": mixed,
            "rows": len(rows),
            "gen_s": t1 - t0,
            "grade_s": t2 - t1,
        }
        if rows:
            model.train()
            model.config.use_cache = False
            n_tokens = sum(len(ids) - lab.count(IGNORE) for (ids, lab), _ in rows)
            total = 0.0
            for i in range(0, len(rows), micro):
                chunk = rows[i : i + micro]
                x, y, att = collate([row for row, _ in chunk], pad, model.device)
                a = torch.tensor([a for _, a in chunk], device=model.device)
                with torch.autocast("cuda", torch.bfloat16):
                    loss = pg_loss(model, x, y, att, a, n_tokens, temperature)
                loss.backward()
                total += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            stats |= {"loss": total, "norm": norm.item()}
        stats["train_s"] = time.time() - t2
        history.append(stats)
        log(
            f"step {step} reward {stats['reward']:.3f} mixed {mixed}/{per_step} "
            f"rows {len(rows)} loss {stats.get('loss', 0):.4f} "
            f"norm {stats.get('norm', 0):.2f} | gen {stats['gen_s']:.0f}s "
            f"grade {stats['grade_s']:.0f}s train {stats['train_s']:.0f}s"
        )
    model.eval()
    model.config.use_cache = True
    return history
