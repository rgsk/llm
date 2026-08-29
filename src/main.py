import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, assert_never, cast

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn

import wandb
from tokenizer import BPETokenizer
from utils import repo_root

use_cuda = True
use_flash = True

ROOT = repo_root()
CKPT_DIR = ROOT / "artifacts" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


BIN_DIR = ROOT / "artifacts" / "data" / "tinystories"
meta = json.loads((BIN_DIR / "meta.json").read_text())
chars_per_token = meta["val"]["chars_per_token"]
tok = BPETokenizer.load(str(ROOT / "artifacts" / "tokenizer" / "bpe_ts_4096.json"))

train_rng = np.random.default_rng(1337)
eval_rng = np.random.default_rng(1337)


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    # model
    n_embed: int  # E
    n_head: int  # nh
    n_layer: int
    block_size: int  # T
    dropout: float
    vocab_size: int  # V
    # training
    batch_size: int  # B
    max_steps: int
    lr: float
    min_lr: float  # lr / 10
    warmup_steps: int  # (2% of max_steps)
    eval_interval: int
    eval_iters: int
    use_compile: bool  # needs JAXTYPING_DISABLE=1
    name: str
    grad_accum_steps: int
    seed: int = 1337
    use_wandb: bool = True
    norm: Literal["layer", "rms"] = "layer"
    ffn: Literal["dense", "gated"] = "dense"
    activation: Literal["relu", "gelu", "silu"] = "relu"


small_cfg = GPTConfig(
    vocab_size=tok.vocab_size,
    n_embed=32,
    n_head=4,
    n_layer=3,
    block_size=8,
    batch_size=32,
    dropout=0.0,
    lr=1e-2,
    min_lr=1e-3,
    max_steps=200,
    warmup_steps=20,
    eval_interval=100,
    eval_iters=100,
    use_compile=False,
    name="scratch",
    grad_accum_steps=1,
    use_wandb=False,
    norm="rms",
    ffn="gated",
    activation="silu",
)

scaled_cfg = GPTConfig(
    vocab_size=tok.vocab_size,
    n_embed=384,
    n_head=6,
    n_layer=6,
    block_size=512,
    batch_size=64,
    dropout=0.0,
    lr=3e-4,
    min_lr=3e-5,
    max_steps=5000,
    warmup_steps=100,
    eval_interval=500,
    eval_iters=100,
    use_compile=True,
    name="scaled",
    grad_accum_steps=1,
    use_wandb=True,
    norm="rms",
    ffn="gated",
    activation="silu",
)

big_cfg = replace(
    scaled_cfg,
    n_embed=512,
    n_head=8,
    n_layer=8,
    dropout=0.0,
    batch_size=64,
    grad_accum_steps=1,
    max_steps=20000,
    warmup_steps=400,
    eval_interval=1000,
    name="big",
)

temp_cfg = replace(
    big_cfg,
    max_steps=200,
    eval_interval=100,
    name="temp",
    use_wandb=False,
)


@jaxtyped(typechecker=beartype)
def get_batch(
    split: Literal["train", "val"],
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[Int[Tensor, "b t"], Int[Tensor, "b t"]]:
    d = np.memmap(BIN_DIR / f"{split}.bin", dtype=np.uint16, mode="r")
    ix = rng.integers(len(d) - block_size, size=batch_size)
    x = np.stack([d[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def make_norm(cfg: GPTConfig) -> nn.LayerNorm | nn.RMSNorm:
    match cfg.norm:
        case "layer":
            return nn.LayerNorm(cfg.n_embed, eps=1e-5)
        case "rms":
            return nn.RMSNorm(cfg.n_embed, eps=1e-5)
        case _:
            assert_never(cfg.norm)


type KVCacheIn = tuple[Float[Tensor, "b nh t_past hs"], Float[Tensor, "b nh t_past hs"]]
type KVCacheOut = tuple[Float[Tensor, "b nh t_kv hs"], Float[Tensor, "b nh t_kv hs"]]


class ResidualProj(nn.Linear):
    """Linear whose output is added to the residual stream; gets 1/√(2·n_layer) init."""


class CausalSelfAttention(nn.Module):
    tril: Tensor

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embed % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embed // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embed, 3 * cfg.n_embed, bias=False)
        self.proj = ResidualProj(cfg.n_embed, cfg.n_embed)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
            persistent=False,
        )
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "b t e"], kv_cache: KVCacheIn | None = None
    ) -> tuple[Float[Tensor, "b t e"], KVCacheOut]:
        _, T, _ = x.shape
        nh, hs = self.n_head, self.head_size
        qkv = self.qkv(x)  # [B, T, 3E]
        q, k, v = rearrange(
            qkv, "b t (three nh hs) -> three b nh t hs", three=3, nh=nh
        )  # [B, nh, T, hs]
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat(
                [past_k, k], dim=2
            )  # [B, nh, T_past + T, hs] = [B, nh, T_kv, hs]
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)
        scores = (
            q @ rearrange(k, "b nh t_kv hs -> b nh hs t_kv") * hs**-0.5
        )  # [B, nh, T, T_kv]
        T_kv = k.size(2)
        assert T_kv <= self.tril.size(0)
        T_past = T_kv - T
        causal = self.tril[T_past:T_kv, :T_kv]  # [T, T_kv]
        scores = scores.masked_fill(causal == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.attn_dropout(w)
        out = w @ v  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.resid_dropout(self.proj(out))
        return out, new_cache


class FlashAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embed % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout_p = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embed, 3 * cfg.n_embed, bias=False)
        self.proj = ResidualProj(cfg.n_embed, cfg.n_embed)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "b t e"], kv_cache: KVCacheIn | None = None
    ) -> tuple[Float[Tensor, "b t e"], KVCacheOut]:
        nh = self.n_head
        qkv = self.qkv(x)  # [B, T, 3E]
        q, k, v = rearrange(
            qkv, "b t (three nh hs) -> three b nh t hs", three=3, nh=nh
        )  # [B, nh, T, hs]
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat(
                [past_k, k], dim=2
            )  # [B, nh, T_past + T, hs] = [B, nh, T_kv, hs]
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)
        if kv_cache is None:
            is_causal = True
        else:
            assert q.size(2) == 1, (
                "cached path assumes one new token at a time; T>1 with a non-empty "
                "cache needs an explicit attn_mask, since is_causal aligns top-left"
            )
            is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.resid_dropout(self.proj(out))
        return out, new_cache


def make_activation(cfg: GPTConfig) -> nn.Module:
    match cfg.activation:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU(approximate="tanh")
        case "silu":
            return nn.SiLU()
        case _:
            assert_never(cfg.activation)


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embed, 4 * cfg.n_embed),
            make_activation(cfg),
            ResidualProj(4 * cfg.n_embed, cfg.n_embed),
            nn.Dropout(cfg.dropout),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        return self.net(x)  # [B, T, E]


class GatedFeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        # param-matched to the ReLU FFN: that has 2*E*4E weights, SwiGLU has 3*E*h,
        # so h = 8E/3. Rounded to a multiple of 64 for GEMM-friendly shapes.
        hidden = round(8 * cfg.n_embed / 3 / 64) * 64  # E=512 -> 1344
        self.gate_up = nn.Linear(cfg.n_embed, 2 * hidden, bias=False)
        self.activation = make_activation(cfg)
        self.down = ResidualProj(hidden, cfg.n_embed, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        gate, up = rearrange(self.gate_up(x), "b t (two h) -> two b t h", two=2)
        return self.dropout(self.down(self.activation(gate) * up))  # [B, T, E]


def make_ffwd(cfg: GPTConfig) -> nn.Module:
    match cfg.ffn:
        case "dense":
            return FeedForward(cfg)
        case "gated":
            return GatedFeedForward(cfg)
        case _:
            assert_never(cfg.ffn)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.ln2 = make_norm(cfg)
        self.attn = FlashAttention(cfg) if use_flash else CausalSelfAttention(cfg)
        self.ffwd = make_ffwd(cfg)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "b t e"], kv_cache: KVCacheIn | None = None
    ) -> tuple[Float[Tensor, "b t e"], KVCacheOut]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return (
            x,  # [B, T, E]
            new_cache,
        )


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            # layers that write back into the residual stream are scaled down by
            # 1/sqrt(2*n_layer): each block adds to the stream twice (attn + ffwd),
            # so over n_layer blocks the variance would grow ~linearly with the
            # number of adds. Shrinking each contributing projection keeps it flat.
            if isinstance(module, ResidualProj):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        idx: Int[Tensor, "b t"],
        targets: Int[Tensor, "b t"] | None = None,
        *,
        use_cache: bool = False,
        kv_caches: list[KVCacheIn] | None = None,
    ) -> (
        tuple[Float[Tensor, "b t v"], Float[Tensor, ""] | None]
        | tuple[Float[Tensor, "b t v"], Float[Tensor, ""] | None, list[KVCacheOut]]
    ):
        _, T = idx.shape
        tok_embed = self.token_embedding_table(idx)  # [B, T, E]
        if use_cache:
            T_past = 0 if kv_caches is None else kv_caches[0][0].size(2)
            assert T_past + T <= self.cfg.block_size
            pos = torch.arange(T_past, T_past + T, device=idx.device)
        else:
            pos = torch.arange(T, device=idx.device)
        pos_embed = self.position_embedding_table(pos)  # [T, E]
        x = tok_embed + pos_embed  # [B, T, E]

        new_caches: list[KVCacheOut] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, layer_cache)
            if use_cache:
                new_caches.append(new_cache)

        x = self.ln_f(x)  # [B, T, E]
        logits = self.lm_head(x)  # [B, T, V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                rearrange(logits, "b t v -> (b t) v"),
                rearrange(targets, "b t -> (b t)"),
            )
        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        idx: Int[Tensor, "b t"],
        max_new_tokens: int,
        use_cache: bool = False,
        temperature: float = 1.0,  # 1.0 no-op | 0.0 greedy | 0.8 recommended
        top_k: int | None = None,  # None no-op | 1 greedy   | prefer top_p
        top_p: float | None = None,  # 1.0 no-op | 0.0 greedy  | 0.95 recommended
        # alternative to top_p
        min_p: float | None = None,  # 0.0 no-op | 1.0 greedy  | 0.05-0.1 recommended
    ) -> Int[Tensor, "b t_out"]:
        assert temperature >= 0.0
        assert top_k is None or top_k > 0
        assert top_p is None or 0.0 <= top_p <= 1.0
        assert min_p is None or 0.0 <= min_p <= 1.0
        if use_cache:
            fed = idx.size(1) + max_new_tokens - 1
            assert fed <= self.cfg.block_size, (
                f"use_cache needs prompt+max_new_tokens-1 ({idx.size(1)}+"
                f"{max_new_tokens}-1={fed}) <= block_size ({self.cfg.block_size}); "
                f"positions come from the position embedding table, which has only "
                f"block_size rows. Use use_cache=False to crop+recompute."
            )

        kv_caches = None
        for _ in range(max_new_tokens):
            if use_cache:
                idx_cond = idx if kv_caches is None else idx[:, -1:]
                logits, _, kv_caches = self(
                    idx_cond,
                    use_cache=True,
                    kv_caches=kv_caches,
                )
            else:
                idx_cond = idx[:, -self.cfg.block_size :]
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # [B, V]
            if temperature == 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    v, _ = torch.topk(logits, k, dim=-1)  # [B, k]
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)  # [B, V]
                if top_p is not None:
                    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
                    cumprobs = sorted_probs.cumsum(dim=-1)
                    sorted_remove = (cumprobs - sorted_probs) > top_p
                    remove = torch.zeros_like(sorted_remove).scatter(
                        -1, sorted_idx, sorted_remove
                    )
                    probs = probs.masked_fill(remove, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                if min_p is not None:
                    keep = probs >= min_p * probs.max(dim=-1, keepdim=True).values
                    probs = probs.masked_fill(~keep, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                nxt = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, nxt], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(
    model: GPT,
    iters=200,
    splits=("train", "val"),
):
    was_training = model.training
    model.eval()
    cfg = model.cfg
    out = {}
    for split in splits:
        losses = []
        for _ in range(iters):
            xb, yb = get_batch(
                split,
                cfg.batch_size,
                cfg.block_size,
                eval_rng,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(xb, yb)
            losses.append(loss)
        out[split] = torch.stack(losses).mean().item()
    if was_training:
        model.train()
    return out


def get_lr(
    step: int, *, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float
) -> float:
    """Learning rate at `step`: linear warmup to max_lr, then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(model: GPT, ckpt_path: Path):
    t_start = time.perf_counter()
    cfg = model.cfg
    wandb.init(
        project="llm",
        name=f"{cfg.name}_{timestamp()}",
        config=asdict(cfg),
        settings=wandb.Settings(silent=True),
        mode=None if cfg.use_wandb else "disabled",
    )
    fwd = cast(GPT, torch.compile(model)) if cfg.use_compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best_val = float("inf")
    TIMING_WARMUP = 10
    t0 = last_it = last_t = None
    for it in range(cfg.max_steps):
        if it == TIMING_WARMUP:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            last_t, last_it = t0, it
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad()
        for _ in range(cfg.grad_accum_steps):
            xb, yb = get_batch("train", cfg.batch_size, cfg.block_size, train_rng)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = fwd(xb, yb)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
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
            ms_step = float("nan")
            if last_it is not None and it > last_it and last_t is not None:
                torch.cuda.synchronize()
                now = time.perf_counter()
                ms_step = (now - last_t) / (it - last_it) * 1000

            torch.cuda.synchronize()
            t_e = time.perf_counter()
            out = estimate_loss(fwd, cfg.eval_iters, splits=("train",))
            torch.cuda.synchronize()
            est_s = time.perf_counter() - t_e

            t_v = time.perf_counter()
            val_loss = full_val_loss(fwd)
            torch.cuda.synchronize()
            val_s = time.perf_counter() - t_v

            train_loss = out["train"]
            # bits per char
            bpc = val_loss / math.log(2) / chars_per_token
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "config": asdict(model.cfg),
                        "step": it,
                        "val_loss": val_loss,
                        "bpc": bpc,
                    },
                    ckpt_path,
                )
            print(
                f"step {it:>4} : train {train_loss:.3f}   val {val_loss:.3f}   bpc {bpc:.3f}"
                f"   {ms_step:.1f} ms/step   est_s {est_s:.1f} s   val_s {val_s:.1f} s"
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "bpc": bpc,
                    "lr": lr,
                    "ms_step": ms_step,
                    "est_s": est_s,
                    "val_s": val_s,
                },
                step=it,
            )
            torch.cuda.synchronize()
            last_t, last_it = time.perf_counter(), it

    torch.cuda.synchronize()
    wandb.summary["total_time_s"] = time.perf_counter() - t_start
    assert t0 is not None
    wandb.summary["train_time_s"] = time.perf_counter() - t0
    wandb.summary["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    wandb.summary["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9

    print(f"total time elapsed: {wandb.summary['total_time_s']:.1f} s")
    print(f"time elapsed (post-warmup): {wandb.summary['train_time_s']:.1f} s")
    print(f"max_memory_allocated: ~{wandb.summary['max_memory_allocated_gb']:.1f} GB")
    print(f"max_memory_reserved: ~{wandb.summary['max_memory_reserved_gb']:.1f} GB")
    wandb.finish()


def generate_sample(model: GPT):
    prompt = torch.tensor([tok.encode("\n")], device=device)
    sample = tok.decode(
        model.generate(
            prompt,
            max_new_tokens=300,
            use_cache=False,
            temperature=0.8,
            top_p=0.95,
        )[0].tolist()
    )
    print(sample)


@torch.no_grad()
def full_val_loss(model: GPT) -> float:
    was_training = model.training
    model.eval()
    cfg = model.cfg
    B = cfg.batch_size
    T = cfg.block_size
    d = np.memmap(BIN_DIR / "val.bin", dtype=np.uint16, mode="r")
    nwin = (len(d) - 1) // T  # how many full windows fit
    x = torch.from_numpy(d[: nwin * T].astype(np.int64)).view(
        nwin, T
    )  # [nwin, T] inputs
    y = torch.from_numpy(d[1 : nwin * T + 1].astype(np.int64)).view(
        nwin, T
    )  # targets, shifted +1
    total = count = 0
    for i in range(0, nwin, B):  # batch the windows through the model
        xb, yb = x[i : i + B].to(device), y[i : i + B].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        # weight by no. of tokens so the mean is correct over uneven chunks
        total += loss.item() * yb.numel()
        count += yb.numel()
    if was_training:
        model.train()
    return total / count


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005


def generate_ckpt_path(name: str):
    if name == "scratch":
        return CKPT_DIR / "scratch.pt"
    return CKPT_DIR / f"{name}_{timestamp()}.pt"


def latest_ckpt(prefix: str) -> Path:
    matches = sorted(CKPT_DIR.glob(f"{prefix}*.pt"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint matching {prefix}*.pt in {CKPT_DIR}")
    return matches[-1]


def set_seed(seed: int):
    global train_rng, eval_rng
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1)


def test(model: GPT):
    pass


if __name__ == "__main__":
    run_training = 0
    print(f"use_cuda: {use_cuda}")
    print(f"use_flash: {use_flash}")
    print(f"JAXTYPING_DISABLE: {os.getenv('JAXTYPING_DISABLE')}")
    print(f"run_training: {run_training}")
    cfg = big_cfg
    # ckpt = CKPT_DIR / "big_2026-08-16_06-45-06.pt"
    ckpt = None
    if run_training:
        cfg = cfg
        set_seed(cfg.seed)
        model = GPT(cfg)
        print(f"model.cfg.name: {model.cfg.name}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{num_params / 1e6:.1f}M params ({num_params})")
        model.to(device)
        ckpt_path = generate_ckpt_path(model.cfg.name)
        train(model, ckpt_path)
        ckpt = ckpt_path
    else:
        ckpt = ckpt or latest_ckpt(cfg.name)
    print(f"loading model - {ckpt.name}")
    saved = torch.load(ckpt, map_location=device)
    # rebuild the exact architecture from the saved config, then load the weights
    reloaded = GPT(GPTConfig(**saved["config"])).to(device)
    reloaded.load_state_dict(saved["model"])
    print(
        f"best checkpoint model at step: {saved['step']}, "
        f"saved val_loss: {saved['val_loss']:.3f}, saved bpc: {saved['bpc']:.3f}"
    )
    generate_sample(reloaded)
    test(reloaded)
