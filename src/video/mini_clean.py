from pathlib import Path
from types import SimpleNamespace as namespace
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def repo_root() -> Path:
    here = Path(__file__)
    for d in here.parents:
        if (d / "pyproject.toml").exists():
            return d
    raise RuntimeError("no pyproject.toml found above " + str(here))


text = (repo_root() / "data" / "input.txt").read_text()
device = "cuda" if torch.cuda.is_available() else "cpu"


class CharTokenizer:
    def __init__(self, text: str):
        self.itoc = sorted(set(text))
        self.ctoi = {c: i for i, c in enumerate(self.itoc)}
        self.vocab_size = len(self.itoc)

    def encode(self, s: str) -> list[int]:
        return [self.ctoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itoc[i] for i in ids)


tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(d: Tensor, *, batch_size: int, block_size: int):
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


class ResidualProj(nn.Linear):
    """Linear whose output is added to the residual stream; gets 1/sqrt(2*n_layer) init."""


class FlashAttention(nn.Module):
    def __init__(self, *, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)

    def forward(self, x: Tensor) -> Tensor:
        qkv: Tensor = self.qkv(x)  # [B, T, 3E]
        qkv = rearrange(qkv, "b t (three e) -> three b t e", three=3)
        q, k, v = [
            rearrange(t, "b t (nh hs) -> b nh t hs", nh=self.n_head) for t in qkv
        ]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, *, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            ResidualProj(4 * n_embed, n_embed),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # [B, T, E]


class Block(nn.Module):
    def __init__(self, *, n_embed: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = FlashAttention(n_embed=n_embed, n_head=n_head)
        self.ffwd = FeedForward(n_embed=n_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# a namespace with these attributes fits; extra attributes are fine
class GPTConfig(Protocol):
    vocab_size: int
    block_size: int
    n_embed: int
    n_head: int
    n_layer: int


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_layer = cfg.n_layer
        self.block_size = cfg.block_size
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed=cfg.n_embed, n_head=cfg.n_head) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module: nn.Module) -> None:
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                std = 0.02
                if isinstance(module, ResidualProj):
                    std *= (2 * self.n_layer) ** -0.5
                module.weight.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(mean=0.0, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        _, T = idx.shape
        x: Tensor = self.token_embedding_table(idx)
        x = x + self.position_embedding_table(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))  # [B, T, V]

    # nn.Module types model(x) as Any; checker-only, keep in step with forward
    if TYPE_CHECKING:

        def __call__(self, idx: Tensor) -> Tensor: ...


def batch_loss(model: GPT, x: Tensor, y: Tensor) -> Tensor:
    B, T = x.shape
    logits = model(x)  # [B, T, V]
    return F.cross_entropy(logits.reshape(B * T, -1), y.reshape(B * T))


@torch.no_grad()
def full_val_loss(model: GPT, *, batch_size: int) -> float:
    was_training = model.training
    model.eval()
    T = model.block_size
    nwin = (len(val_data) - 1) // T  # how many full windows fit
    xs = val_data[: nwin * T].view(nwin, T)
    ys = val_data[1 : nwin * T + 1].view(nwin, T)  # targets, shifted +1
    # summed on the GPU: no sync per batch
    total, count = torch.tensor(0.0, device=device), 0
    for i in range(0, nwin, batch_size):
        x, y = xs[i : i + batch_size].to(device), ys[i : i + batch_size].to(device)
        # weight by #tokens so the mean is correct over the uneven last chunk
        total += batch_loss(model, x, y) * y.numel()
        count += y.numel()
    model.train(was_training)
    return (total / count).item()


@torch.no_grad()
def train_loss_est(model: GPT, *, batch_size: int, iters: int) -> float:
    """Cheap estimate from random batches; the train split is too big for a full pass."""
    was_training = model.training
    model.eval()
    losses: list[Tensor] = []
    for _ in range(iters):
        x, y = get_batch(train_data, batch_size=batch_size, block_size=model.block_size)
        losses.append(batch_loss(model, x, y))
    model.train(was_training)
    return torch.stack(losses).mean().item()  # one sync, not one per batch


class TrainConfig(Protocol):
    batch_size: int
    max_steps: int
    lr: float
    eval_interval: int
    eval_iters: int


def train(model: GPT, cfg: TrainConfig):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def evaluate(step: int):
        tr = train_loss_est(model, batch_size=cfg.batch_size, iters=cfg.eval_iters)
        va = full_val_loss(model, batch_size=cfg.batch_size)
        print(f"  step {step:>4} : train {tr:.3f}   val {va:.3f}")

    for step in range(cfg.max_steps):
        if step % cfg.eval_interval == 0:
            evaluate(step)
        x, y = get_batch(
            train_data, batch_size=cfg.batch_size, block_size=model.block_size
        )
        loss = batch_loss(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    evaluate(cfg.max_steps)


@torch.no_grad()
def generate(model: GPT, idx: Tensor, *, max_new_tokens: int) -> Tensor:
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size :])[:, -1, :]  # [B, V]
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)  # [B, 1]
        idx = torch.cat([idx, nxt], dim=1)
    model.train(was_training)
    return idx


# defaults hold only size-independent fields; spread into each config with **vars(...)
gpt_defaults = namespace(vocab_size=tok.vocab_size)
train_defaults = namespace(eval_interval=100, eval_iters=100)

gpt_cfg: GPTConfig = namespace(
    **vars(gpt_defaults),
    block_size=8,
    n_embed=64,
    n_head=4,
    n_layer=3,
)
train_cfg: TrainConfig = namespace(
    **vars(train_defaults),
    batch_size=32,
    max_steps=500,
    lr=1e-2,
)
print(gpt_cfg)
print(train_cfg)

x, y = get_batch(
    train_data,
    batch_size=train_cfg.batch_size,
    block_size=gpt_cfg.block_size,
)
print(x.shape, y.shape)

model = GPT(gpt_cfg).to(device)
deep_cfg: GPTConfig = namespace(**vars(gpt_cfg) | dict(n_layer=6))
for cfg in (gpt_cfg, deep_cfg):
    n_params = sum(p.numel() for p in GPT(cfg).parameters())
    print(f"n_layer={cfg.n_layer}: {n_params:,} params")
print(model(x).shape)  # [B, T, V]
# untrained, so close to uniform over the vocab: loss ≈ ln(65) = 4.17
print(batch_loss(model, x, y).item())

train(model, train_cfg)

prompt = torch.tensor([tok.encode("\n")], device=device)
out = generate(model, prompt, max_new_tokens=200)
print(tok.decode(out[0].tolist()))
