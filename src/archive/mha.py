import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


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
