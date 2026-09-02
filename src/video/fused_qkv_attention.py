import torch
from common import KVCache
from dropout import Dropout
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from softmax import softmax
from torch import Tensor


class FusedQKVAttention(Module):
    """The same attention MultiHeadAttention computes, with the per-head Linears
    fused into one. This is main.py's CausalSelfAttention, and the weight layout
    its checkpoints were written with.

    n_head separate query projections, each [E -> hs], stacked ARE a single
    [E -> E] projection: they all read the same x and write into disjoint output
    slices. Same for key, same for value. So 3 * n_head small matmuls become one
    [E -> 3E]. Same arithmetic, one weight matrix, no Python loop over heads.

    The 3E axis is laid out as (3, n_head, head_size) -- q for every head, then
    k for every head, then v. That ordering is pure convention, and it is the
    one the checkpoints on disk assume.

    This is also the first layer with a KV cache. Generation feeds the model one
    token at a time, and row t of the score matrix only ever needs k and v for
    positions <= t. Those are a function of x alone -- past keys and values do
    not change when a new token arrives -- so recomputing them every step is
    pure waste. Hand the layer the previous (k, v) and it projects only the new
    token, concatenates, and attends: one row of scores instead of T.

    Cost per token goes from O(T^2 E) to O(T E), in exchange for holding
    2 * n_layer * B * T * E cache values in memory. That trade is why decoding
    is memory-bandwidth-bound and training is compute-bound.
    """

    def __init__(
        self, n_embed: int, n_head: int, block_size: int, dropout: float = 0.0
    ):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        self.register_buffer(
            "tril",
            torch.ones(block_size, block_size, dtype=torch.bool).tril(),
            persistent=False,
        )
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        """x is only the NEW tokens. kv_cache carries everything before them.

        use_cache picks the return type: a plain Tensor (what Block adds to the
        residual stream) or (out, new_cache). Keeping it a flag is what lets
        this file gain a cache without every caller learning about one.
        """
        B, T, E = x.shape
        nh, hs = self.n_head, self.head_size
        # one matmul, then split into q, k, v each [B, T, E]
        qkv = self.qkv(x)
        q, k, v = qkv.split(E, dim=-1)

        # split E into (nh, hs) and move nh to the batch position -> [B, nh, T, hs]
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            # [B, nh, T_past, hs] ++ [B, nh, T, hs] -> [B, nh, T_kv, hs]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        T_kv = k.size(2)
        T_past = T_kv - T
        assert T_kv <= self.tril.size(0), "sequence outgrew block_size"

        scores = q @ k.transpose(-2, -1) * hs**-0.5  # [B, nh, T, T_kv]
        # q rows are positions T_past .. T_kv-1, so the mask is that row band of
        # tril, not its top-left corner. With T == 1 the band is a single row of
        # all True: the newest token may look at every cached position.
        causal = self.tril[T_past:T_kv, :T_kv]  # [T, T_kv]
        scores = scores.masked_fill(~causal, float("-inf"))
        w = softmax(scores, dim=-1)
        w = self.attn_dropout(w)
        out = w @ v  # [B, nh, T, hs]
        out = out.transpose(1, 2).reshape(B, T, E)  # heads back side by side
        out = self.resid_dropout(self.proj(out))
        return (out, new_cache) if use_cache else out


if __name__ == "__main__":
    import time

    import torch.nn.functional as F
    from multi_head_attention import MultiHeadAttention
    from torch import nn

    torch.manual_seed(0)
    B, T, E, NH = 2, 8, 32, 4
    HS = E // NH
    x = torch.randn(B, T, E)
    fused = FusedQKVAttention(E, NH, T)

    # 1. same shape and same parameter count as the unfused version -- fusing
    #    rearranges weights, it does not add or remove any
    mha = MultiHeadAttention(E, NH, T)
    assert fused(x).shape == (B, T, E)
    n_fused = sum(p.numel() for p in fused.parameters())
    n_mha = sum(p.numel() for p in mha.parameters())
    assert n_fused == n_mha == 3 * E * E + E * E + E, (n_fused, n_mha)

    # 2. the keys the checkpoints on disk actually use
    assert [n for n, _ in fused.named_parameters()] == [
        "qkv.weight",
        "proj.weight",
        "proj.bias",
    ]
    assert fused.qkv.weight.shape == (3 * E, E)

    # 3. THE test: transplanting MHA's per-head weights into one fused matrix
    #    reproduces MHA exactly. Row block i of the q half IS head i's query.
    with torch.no_grad():
        # h.query.weight [hs, E] (weights are stored like)
        q_all = torch.cat([h.query.weight for h in mha.heads], dim=0)  # [E, E]
        k_all = torch.cat([h.key.weight for h in mha.heads], dim=0)
        v_all = torch.cat([h.value.weight for h in mha.heads], dim=0)
        fused.qkv.weight.copy_(torch.cat([q_all, k_all, v_all], dim=0))  # [3E, E]
        fused.proj.weight.copy_(mha.proj.weight)
        fused.proj.bias.copy_(mha.proj.bias)
    print("fused vs unfused:", (fused(x) - mha(x)).abs().max().item())
    assert (fused(x) - mha(x)).abs().max() < 1e-6

    # 3. the causal mask is state, not a parameter -- and not a checkpoint key
    assert [n for n, _ in fused.named_buffers()] == ["tril"]
    assert [n for n, _ in fused.named_parameters()] == [
        "qkv.weight",
        "proj.weight",
        "proj.bias",
    ]
    assert "tril" not in fused.state_dict()  # persistent=False
    assert fused.tril.dtype == torch.bool and not fused.tril.requires_grad
    assert fused.tril.shape == (T, T)
    # one mask serves every length up to block_size -- forward just slices it
    assert fused(x[:, :3]).shape == (B, 3, E)
    # and .to() moves it without casting it, so masked_fill still works
    assert FusedQKVAttention(E, NH, T).to(torch.float64)(x.double()).shape == (B, T, E)

    # 4. ... and the gradients agree, so it is a drop-in replacement
    xf = x.clone().requires_grad_(True)
    xu = x.clone().requires_grad_(True)
    fused(xf).square().sum().backward()
    mha(xu).square().sum().backward()
    assert (xf.grad - xu.grad).abs().max() < 1e-5
    assert (
        fused.qkv.weight.grad[0:HS] - mha.heads[0].query.weight.grad
    ).abs().max() < 1e-5

    # we can perform split like below alternatively

    # 5. the split really is (3, nh, hs): check against SDPA on the same q, k, v
    qkv = fused.qkv(x).view(B, T, 3, NH, HS).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref_out = ref_out.transpose(1, 2).reshape(B, T, E)
    assert (fused(x) - fused.proj(ref_out)).abs().max() < 1e-6

    # 7. causality: perturbing token t cannot change any output before t
    out = fused(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        out2 = fused(x2)
        assert (out2[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"
        assert (out2[:, t] - out[:, t]).abs().max() > 1e-3
    print("causality holds for all t")

    # 8. heads still do not mix before proj: zeroing head 0's value rows blanks
    #    exactly the first hs columns of the concatenated output, nothing else
    probe = FusedQKVAttention(E, NH, T)
    with torch.no_grad():
        probe.proj.weight.copy_(torch.eye(E))
        probe.proj.bias.zero_()
        before = probe(x).clone()
        probe.qkv.weight[2 * E : 2 * E + HS].zero_()  # head 0's value rows
    after = probe(x)
    assert (after[..., :HS] == 0).all()
    assert (after[..., HS:] - before[..., HS:]).abs().max() < 1e-6

    # 9. against a torch mirror, forward and backward
    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(E, 3 * E, bias=False)
            self.proj = nn.Linear(E, E)

        def forward(self, x):
            B, T, E_ = x.shape
            qkv = self.qkv(x).view(B, T, 3, NH, HS).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
            return self.proj(o.transpose(1, 2).reshape(B, T, E_))

    ref = Ref()
    assert fused.state_dict().keys() == ref.state_dict().keys()
    fused.load_state_dict(ref.state_dict())
    fused.zero_grad()  # test 4 left gradients on it; they would accumulate
    xm = x.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    om, orf = fused(xm), ref(xr)
    assert (om - orf).abs().max() < 1e-6
    om.square().sum().backward()
    orf.square().sum().backward()
    assert (xm.grad - xr.grad).abs().max() < 1e-5
    assert all(
        (a.grad - b.grad).abs().max() < 1e-4
        for a, b in zip(fused.parameters(), ref.parameters())
    )

    # 10. fusing is NOT automatically faster, and it is worth seeing that.
    #     One GEMM replaces 3*nh, but the permuted q/k/v are non-contiguous and
    #     the output needs a transpose+copy that the unfused path gets for free
    #     from torch.cat. The win here is one weight matrix and one convention --
    #     the speed arrives with SDPA, in the next file.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Eb, NHb, Tb = 512, 8, 256
    xb = torch.randn(8, Tb, Eb, device=dev)
    big = {
        "unfused": MultiHeadAttention(Eb, NHb, Tb).to(dev),
        "fused": FusedQKVAttention(Eb, NHb, Tb).to(dev),
    }
    t = {}
    for name, f in big.items():
        for _ in range(3):
            f(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            f(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
        t[name] = (time.perf_counter() - t0) / 10 * 1e3
    print(
        f"{dev}  [8,{Tb},{Eb}] nh={NHb} -- unfused {t['unfused']:.2f} ms   "
        f"fused {t['fused']:.2f} ms   ratio {t['unfused'] / t['fused']:.2f}x"
    )

    # ---------------------------------------------------------------- KV cache

    dec = FusedQKVAttention(E, NH, T)
    full = dec(x)  # [B, T, E], every position at once

    # 11. THE test: decoding one token at a time through the cache reproduces
    #     the full forward exactly. Same weights, same arithmetic, fewer matmuls
    cache = None
    steps = []
    for t_ in range(T):
        out_t, cache = dec(x[:, t_ : t_ + 1], cache, use_cache=True)
        steps.append(out_t)
    assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-6
    print("incremental decode matches full forward")

    # 12. and any prefill/decode split does too -- prefill T-1 tokens in one
    #     call, then step the last one. This is what generate() actually does
    prefix, new_cache = dec(x[:, :-1], use_cache=True)
    last, new_cache = dec(x[:, -1:], new_cache, use_cache=True)
    assert (torch.cat([prefix, last], dim=1) - full).abs().max() < 1e-6

    # 13. the cache is exactly the k and v a full forward would have computed --
    #     nothing is approximated, past keys simply never change
    k_full, v_full = dec.qkv(x).split(E, dim=-1)[1:]
    k_full = k_full.view(B, T, NH, HS).transpose(1, 2)
    v_full = v_full.view(B, T, NH, HS).transpose(1, 2)
    ck, cv = cache
    assert ck.shape == cv.shape == (B, NH, T, HS)
    assert (ck - k_full).abs().max() < 1e-6 and (cv - v_full).abs().max() < 1e-6

    # 14. use_cache is the only thing that changes the return type -- Block still
    #     gets a bare Tensor, and the cached path is differentiable like any other
    assert isinstance(dec(x), Tensor)
    assert isinstance(dec(x, use_cache=True), tuple)
    xc = x.clone().requires_grad_(True)
    o, _ = dec(xc[:, :4], use_cache=True)
    o.square().sum().backward()
    assert xc.grad is not None and (xc.grad[:, 4:] == 0).all()

    # 15. what it buys: decoding T tokens by recomputing the whole prefix every
    #     step, against feeding one token into the cache. The gap is the point
    Td = 256
    xd = torch.randn(1, Td, Eb, device=dev)
    gen = FusedQKVAttention(Eb, NHb, Td).to(dev)

    def decode(use_cache: bool) -> float:
        with torch.no_grad():
            for _ in range(2):
                gen(xd[:, :8])
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            cache = None
            for t_ in range(Td):
                if use_cache:
                    _, cache = gen(xd[:, t_ : t_ + 1], cache, use_cache=True)
                else:
                    gen(xd[:, : t_ + 1])  # recompute the whole prefix
            if dev == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    recompute, cached = decode(False), decode(True)
    print(
        f"{dev}  decode {Td} tokens, E={Eb} -- recompute {recompute:.1f} ms   "
        f"cached {cached:.1f} ms   ratio {recompute / cached:.2f}x"
    )

    print("ok")
