from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn

from utils import repo_root

ROOT = repo_root()
DATA = ROOT / "data" / "input.txt"
text = DATA.read_text(encoding="utf-8")
itoc = sorted(set(text))
ctoi = {c: i for i, c in enumerate(itoc)}
device = "cuda" if torch.cuda.is_available() else "cpu"


def encode(s: str):
    return [ctoi[c] for c in s]


def decode(ids: list[int]):
    return "".join(itoc[i] for i in ids)


data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    # model
    n_embed: int = 32  # E
    n_head: int = 4  # nh
    n_layer: int = 3
    block_size: int = 8  # T
    dropout: float = 0.2
    vocab_size: int  # V
    # training
    batch_size: int = 32  # B
    lr: float = 1e-2
    max_steps: int = 1000
    eval_interval: int = 100
    eval_iters: int = 200


small_cfg = GPTConfig(vocab_size=len(itoc))

scaled_cfg = GPTConfig(
    vocab_size=len(itoc),
    n_embed=384,
    n_head=6,
    n_layer=6,
    block_size=256,
    batch_size=64,
    lr=3e-4,
    max_steps=5000,
    eval_interval=500,
    eval_iters=100,
)


@jaxtyped(typechecker=beartype)
def get_batch(
    split: Literal["train", "val"],
    batch_size: int,
    block_size: int,
) -> tuple[Int[Tensor, "b t"], Int[Tensor, "b t"]]:
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    tril: Tensor

    def __init__(self, block_size: int, n_embed: int, head_size: int, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t hs"]:
        _, T, _ = x.shape
        q = self.query(x)  # [B, T, hs]
        k = self.key(x)  # [B, T, hs]
        scores = (
            q @ rearrange(k, "b t hs -> b hs t") * self.head_size**-0.5
        )  # [B, T, T]

        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)  # [B, T, T]
        w = self.dropout(w)
        v = self.value(x)  # [B, T, hs]
        out = w @ v  # [B, T, hs]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size: int, n_embed: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        head_size = n_embed // n_head
        self.heads = nn.ModuleList(
            [Head(block_size, n_embed, head_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, E]
        return self.dropout(self.proj(out))  # [B, T, E]


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        return self.net(x)  # [B, T, E]


class Block(nn.Module):
    def __init__(self, block_size: int, n_embed: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(block_size, n_embed, n_head, dropout)
        self.ffwd = FeedForward(n_embed, dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x  # [B, T, E]


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(cfg.block_size, cfg.n_embed, cfg.n_head, cfg.dropout)
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        idx: Int[Tensor, "b t"],
        targets: Int[Tensor, "b t"] | None = None,
    ) -> tuple[Float[Tensor, "b t v"], Float[Tensor, ""] | None]:
        _, T = idx.shape
        tok_embed = self.token_embedding_table(idx)  # [B, T, E]
        pos = torch.arange(T, device=idx.device)
        pos_embed = self.position_embedding_table(pos)  # [T, E]
        x = tok_embed + pos_embed  # [B, T, E]
        x = self.blocks(x)  # [B, T, E]
        x = self.ln_f(x)  # [B, T, E]
        logits = self.lm_head(x)  # [B, T, V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                rearrange(logits, "b t v -> (b t) v"),
                rearrange(targets, "b t -> (b t)"),
            )
        return logits, loss

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def generate(
        self, idx: Int[Tensor, "b t"], max_new_tokens: int
    ) -> Int[Tensor, "b t_out"]:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            probs = F.softmax(logits[:, -1, :], dim=-1)  # [B, V]
            nxt = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, nxt], dim=1)
        return idx


model = GPT(scaled_cfg)
model.to(device)


@torch.no_grad()
def estimate_loss(iters=200):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(iters):
            xb, yb = get_batch(
                split,
                model.cfg.batch_size,
                model.cfg.block_size,
            )
            _, loss = model(xb, yb)
            losses.append(loss)
        out[split] = torch.stack(losses).mean().item()
    model.train()
    return out


def train():
    cfg = model.cfg
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for it in range(cfg.max_steps):
        xb, yb = get_batch(
            "train",
            cfg.batch_size,
            cfg.block_size,
        )
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % cfg.eval_interval == 0 or it == cfg.max_steps - 1:
            out = estimate_loss(cfg.eval_iters)
            train_loss, val_loss = out["train"], out["val"]
            print(f"step {it:>4} : train {train_loss:.3f}   val {val_loss:.3f}")


def generate_sample():
    start = torch.tensor([encode("\n")], device=device)
    sample = decode(model.generate(start, max_new_tokens=500)[0].tolist())
    print(sample)


if __name__ == "__main__":
    train()
    generate_sample()
