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

block_size = 8  # T
batch_size = 32  # B


@jaxtyped(typechecker=beartype)
def get_batch(
    split: Literal["train", "val"],
) -> tuple[Int[Tensor, "b t"], Int[Tensor, "b t"]]:
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


vocab_size = len(itoc)  # V
n_embed = 32  # E
n_head = 4
n_layer = 3


class Head(nn.Module):
    tril: Tensor

    def __init__(self, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

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
        v = self.value(x)  # [B, T, hs]
        out = w @ v  # [B, T, hs]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, E]
        return self.proj(out)  # [B, T, E]


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        return self.net(x)  # [B, T, E]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention()
        self.ffwd = FeedForward()

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x  # [B, T, E]


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

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
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)  # [B, V]
            nxt = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, nxt], dim=1)
        return idx


model = GPT()
model.to(device)


@torch.no_grad()
def estimate_loss(eval_iters=200):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss)
        out[split] = torch.stack(losses).mean().item()
    model.train()
    return out


lr = 1e-2
opt = torch.optim.AdamW(model.parameters(), lr=lr)
max_steps = 1000
for it in range(max_steps):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 100 == 0 or it == max_steps - 1:
        out = estimate_loss()
        train_loss, val_loss = out["train"], out["val"]
        print(f"step {it:>4} : train {train_loss:.3f}   val {val_loss:.3f}")

start = torch.tensor([encode("\n")], device=device)
sample = decode(model.generate(start, max_new_tokens=500)[0].tolist())
print(sample)
