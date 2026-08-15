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


class CausalSelfAttention(nn.Module):
    tril: Tensor

    def __init__(self, block_size: int, n_embed: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        _, T, _ = x.shape
        nh, hs = self.n_head, self.head_size
        qkv = self.qkv(x)  # [B, T, 3E]
        q, k, v = rearrange(
            qkv, "b t (three nh hs) -> three b nh t hs", three=3, nh=nh
        )  # [B, nh, T, hs]
        scores = q @ rearrange(k, "b nh t hs -> b nh hs t") * hs**-0.5  # [B, nh, T, T]
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.attn_dropout(w)
        out = w @ v  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        return self.resid_dropout(self.proj(out))


class FlashAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.dropout_p = dropout
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.resid_dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t e"]) -> Float[Tensor, "b t e"]:
        nh = self.n_head
        qkv = self.qkv(x)  # [B, T, 3E]
        q, k, v = rearrange(
            qkv, "b t (three nh hs) -> three b nh t hs", three=3, nh=nh
        )  # [B, nh, T, hs]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.resid_dropout(self.proj(out))
        return out


type KVCacheIn = tuple[Float[Tensor, "b nh t_past hs"], Float[Tensor, "b nh t_past hs"]]
type KVCacheOut = tuple[Float[Tensor, "b nh t_kv hs"], Float[Tensor, "b nh t_kv hs"]]


class CausalSelfAttentionKV(nn.Module):
    tril: Tensor

    def __init__(self, block_size: int, n_embed: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

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


class FlashAttentionKV(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.dropout_p = dropout
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.resid_dropout = nn.Dropout(dropout)

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
