"""Attention for packed finetuning: GQA + rope, flex while training, SDPA at decode.

The rungs beside this file (mha/fused/sdpa/gqa, window, ring, sinks) are left as
they are; this is the slice SFT and DPO run on.

The flex path needs CUDA to train: torch has no CPU backward for it, so on CPU
a forward that builds a graph raises. Passing no block_mask falls back to SDPA,
which is how the CPU tests still exercise this layer.

Two paths because they want opposite things. Training sees a packed window whose
documents must not read each other, and a BlockMask skips those blocks whole --
faster than no mask at all. Decode sees one query against a filled cache: no
blocks to skip, no boundaries to respect, so plain SDPA is both simpler and
quicker.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from dropout import Dropout
from kv_cache import KVCache
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from rope import apply_rope

# compiled once at import: eager create_block_mask/flex_attention materialize
# what they exist to avoid -- 3.05ms vs 0.16ms to build one mask on a 4060
flex = torch.compile(flex_attention)


class FlexAttention(Module):
    """GQA with rope. cos/sin are built once per batch by GPT and required here:
    a layer computing its own would use the default base and scale, so a
    PI/NTK/YaRN run would come out silently un-extended. Angles not a table for
    the same reason -- extension varies both per call, and runs past block_size."""

    takes_rope = True  # Block reads this to decide whether to hand over cos/sin

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        n_kv_head: int | None = None,
    ):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        n_kv_head = n_head if n_kv_head is None else n_kv_head
        assert n_head % n_kv_head == 0, "n_kv_head must divide n_head"
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.head_size = n_embed // n_head
        kv_dim = n_kv_head * self.head_size
        self.qkv = Linear(n_embed, n_embed + 2 * kv_dim, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        self.resid_dropout = Dropout(dropout)
        self.dropout_p = dropout
        self.block_size = block_size  # only to size a preallocated cache

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        block_mask: BlockMask | None = None,
    ) -> Tensor | tuple[Tensor, KVCache]:
        B, T, E = x.shape
        nh, nkv, hs = self.n_head, self.n_kv_head, self.head_size
        q, k, v = self.qkv(x).split([E, nkv * hs, nkv * hs], dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nkv, hs).transpose(1, 2)
        v = v.view(B, T, nkv, hs).transpose(1, 2)

        # rotate before the append, so the cache holds keys already rotated
        assert cos.size(0) == T, f"cos covers {cos.size(0)} positions, not {T}"
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if use_cache:
            if kv_cache is None:
                # grow=True: block_size sizes the first buffer, it does not cap
                # the run -- rope has no ceiling now, so the cache must not either
                kv_cache = KVCache((B, nkv, self.block_size, hs), grow=True)
            k, v = kv_cache.append(k, v)

        p = self.dropout_p if self.training else 0.0
        if T == 1:
            # decode: the whole cached row is visible, so no mask at all
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=nh != nkv)
        elif block_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=p, enable_gqa=nh != nkv
            )
        else:
            assert x.is_cuda or not torch.is_grad_enabled(), (
                "flex_attention has no CPU backward; use block_mask=None on cpu"
            )
            out = flex(q, k, v, block_mask=block_mask, enable_gqa=nh != nkv)

        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.resid_dropout(self.proj(out))
        return (out, kv_cache) if use_cache else out


if __name__ == "__main__":
    # assertions are in test_flex_attention.py; this prints what the two paths cost
    import time

    from mask import doc_block_mask, doc_ids
    from rope import rope_angles

    if not torch.cuda.is_available():
        print("no gpu, skipping")
        raise SystemExit

    B, T, E, NH, NKV = 4, 512, 384, 6, 2
    layer = FlexAttention(E, NH, 1024, n_kv_head=NKV).cuda()  # room to decode past T
    hs = E // NH
    x = torch.randn(B, T, E, device="cuda")
    ids = torch.ones(B, T, dtype=torch.long, device="cuda")
    ids[:, ::128] = 0  # a document every 128 tokens
    bm = doc_block_mask(doc_ids(ids, 0))

    def ms(fn, n=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    print(f"B={B} T={T} E={E} n_head={NH} n_kv_head={NKV}")
    cos, sin = (t.cuda() for t in rope_angles(0, T, hs))
    print(f"  causal, no doc mask  {ms(lambda: layer(x, cos, sin)):6.2f} ms")
    print(
        f"  flex + doc mask      {ms(lambda: layer(x, cos, sin, block_mask=bm)):6.2f} ms"
    )

    # in-place cache writes are invisible to autograd, so decode is no_grad
    with torch.no_grad():
        step = torch.randn(B, 1, E, device="cuda")
        _, cache = layer(x, cos, sin, use_cache=True)
        c1, s1 = (t.cuda() for t in rope_angles(T, 1, hs))
        dt = ms(lambda: layer(step, c1, s1, cache, use_cache=True))
    print(f"  one decode step      {dt:6.2f} ms")
