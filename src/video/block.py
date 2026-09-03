from typing import Literal

from common import KVCache
from feed_forward import FeedForward
from fused_qkv_attention import FusedQKVAttention
from gated_feed_forward import GatedFeedForward
from gqa_attention import GQAttention
from layer_norm import LayerNorm
from module import Module
from multi_head_attention import MultiHeadAttention
from rms_norm import RMSNorm
from sdpa_attention import SDPAttention
from torch import Tensor

Attention = Literal["mha", "fused", "sdpa", "gqa"]
Norm = Literal["layer", "rms"]
FFN = Literal["dense", "gated"]


def make_attention(
    kind: Attention,
    n_embed: int,
    n_head: int,
    block_size: int,
    dropout: float,
    n_kv_head: int | None = None,
) -> Module:
    """All four write the same residual stream; only "mha" has different keys.

    n_kv_head is "gqa" only -- the others project one k and v per query head, so
    there is nothing to set. "gqa" with n_kv_head=None is MHA arithmetic in the
    gqa layout, which is how the two get compared.
    """
    match kind:
        case "mha":
            return MultiHeadAttention(n_embed, n_head, block_size, dropout)
        case "fused":
            return FusedQKVAttention(n_embed, n_head, block_size, dropout)
        case "sdpa":
            return SDPAttention(n_embed, n_head, dropout)  # builds its own mask
        case "gqa":
            # block_size is passed for the cache buffer, not for a mask: this is
            # the only attention that preallocates instead of growing by copying
            return GQAttention(n_embed, n_head, n_kv_head, dropout, block_size)
        case _:
            raise ValueError(f"unknown attention: {kind}")


def make_norm(kind: Norm, n_embed: int) -> Module:
    match kind:
        case "layer":
            return LayerNorm(n_embed, eps=1e-5)
        case "rms":
            return RMSNorm(n_embed, eps=1e-5)
        case _:
            raise ValueError(f"unknown norm: {kind}")


def make_ffwd(kind: FFN, n_embed: int, dropout: float) -> Module:
    """dense is ReLU, gated is SwiGLU -- the activation is not a separate knob
    here, since a gated FFN with ReLU is not a thing anyone ships."""
    match kind:
        case "dense":
            return FeedForward(n_embed, dropout)
        case "gated":
            return GatedFeedForward(n_embed, dropout)
        case _:
            raise ValueError(f"unknown ffn: {kind}")


class Block(Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        attention: Attention = "mha",
        norm: Norm = "layer",
        ffn: FFN = "dense",
        n_kv_head: int | None = None,
    ):
        self.ln1 = make_norm(norm, n_embed)
        self.ln2 = make_norm(norm, n_embed)
        self.attn = make_attention(
            attention, n_embed, n_head, block_size, dropout, n_kv_head
        )
        self.ffwd = make_ffwd(ffn, n_embed, dropout)

    def forward(
        self,
        x: Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        """The cache belongs to attn -- ln1, ffwd and the two residual adds are
        per-position, so they neither read it nor need one of their own. A block
        just carries it in and hands the new one back.

        use_cache needs attention="fused", "sdpa" or "gqa"; "mha" takes x only.
        """
        if use_cache:
            attn_out, new_cache = self.attn(self.ln1(x), kv_cache, use_cache=True)
        else:
            attn_out = self.attn(self.ln1(x))
        x = x + attn_out  # communicate
        x = x + self.ffwd(self.ln2(x))  # compute
        return (x, new_cache) if use_cache else x  # [B, T, E]


if __name__ == "__main__":
    import torch

    B, T, E, NH = 2, 8, 32, 4
    x = torch.randn(B, T, E)
    blk = Block(E, NH, T)

    # 1. shape preserved -- blocks are stackable
    assert blk(x).shape == (B, T, E)

    # 2. key layout matches main.py's Block
    keys = list(blk.state_dict())
    assert keys[:4] == ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"]
    assert any(k.startswith("attn.") for k in keys)
    assert keys[-4:] == [
        "ffwd.net.0.weight",
        "ffwd.net.0.bias",
        "ffwd.net.2.weight",
        "ffwd.net.2.bias",
    ]

    # 3. the residual path is an identity: zero both output projections
    #    and the block passes its input through untouched
    with torch.no_grad():
        blk.attn.proj.weight.zero_()
        blk.attn.proj.bias.zero_()
        blk.ffwd.net[2].weight.zero_()
        blk.ffwd.net[2].bias.zero_()
    assert (blk(x) - x).abs().max() == 0
    print("zeroed sublayers -> block is exactly identity")

    # 4. causality survives the block
    blk = Block(E, NH, T)
    out = blk(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        assert (blk(x2)[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"

    # 5. the cache survives the ffwd and the residual adds: decoding one token
    #    at a time through a fused block reproduces its full forward exactly
    dec = Block(E, NH, T, attention="fused")
    full = dec(x)
    cache, steps = None, []
    for t in range(T):
        out_t, cache = dec(x[:, t : t + 1], cache, use_cache=True)
        steps.append(out_t)
    assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-6
    assert cache[0].shape == (B, NH, T, E // NH)
    print("block decodes incrementally")

    # 6. the same holds for sdpa, whose cache takes a different route to it
    dec = Block(E, NH, T, attention="sdpa")
    full = dec(x)
    cache, steps = None, []
    for t in range(T):
        out_t, cache = dec(x[:, t : t + 1], cache, use_cache=True)
        steps.append(out_t)
    assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-6

    # 7. use_cache is opt-in, so mha -- which has no cache -- still works
    assert Block(E, NH, T, attention="mha")(x).shape == (B, T, E)

    print("ok")
