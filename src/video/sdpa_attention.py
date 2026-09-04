import torch
import torch.nn.functional as F
from torch import Tensor

from common import KVCache
from dropout import Dropout
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from rope import apply_rope, rope_tables
from sliding_window import sliding_window_mask


class SDPAttention(Module):
    """FusedQKVAttention with the mask, the softmax and both matmuls handed to
    torch's fused kernel.

    This is the one file in the series that does not rebuild what it calls, and
    the reason is worth being precise about. SDPA's advantage is not arithmetic
    -- it computes exactly what the previous file computes. It is that it never
    materialises the [B, nh, T, T] score matrix: it walks the sequence in tiles
    that stay in on-chip SRAM, so memory grows with T instead of T^2. That is a
    memory-access pattern, not a formula, and eager tensor ops cannot express it.

    People call this layer "FlashAttention". The name overpromises. SDPA is a
    dispatcher: it picks a backend that fits the inputs, and the flash kernel
    needs fp16/bf16. Ask for it in fp32 and you get the memory-efficient backend
    instead, silently.

    The KV cache costs one thing here that it did not cost the hand-written
    version: is_causal stops being usable the moment there is a past. The flag
    aligns its triangle to the top-left corner, which is only the causal mask
    when the queries start at position 0. With a cache they do not, so the layer
    has to say which of the two situations it is in -- see forward.

    use_rope=True turns on rotary positions (rope.py). This is the only place
    positions can enter once they are rotary rather than additive: GPT stops
    adding a position vector to the residual stream and this layer rotates q and
    k instead. The rotation happens BEFORE the cache append, so the cache holds
    keys that are already rotated and no step re-rotates them.

    window=W masks each query to the W keys ending at it (sliding_window.py).
    It changes what this layer COMPUTES and not what it stores: the cache still
    holds every position and the kernel still walks every one of them. What it
    buys is a model whose reach grows with depth -- L*(W-1)+1 -- instead of a
    score matrix that grows with T^2. And it is only sound because rope already
    made scores a function of i - j, so a key 500 positions back scores exactly
    as it should whether or not position 499 is still in the mask.
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float = 0.0,
        block_size: int | None = None,
        use_rope: bool = False,
        window: int | None = None,
    ):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        assert window is None or window >= 1, "a window has to include the query itself"
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        self.dropout_p = dropout
        self.resid_dropout = Dropout(dropout)
        self.window = window
        self.use_rope = use_rope
        if use_rope:
            assert block_size is not None, "rope needs block_size to size its tables"
            cos, sin = rope_tables(block_size, self.head_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        B, T, E = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(E, dim=-1)

        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        T_past = 0 if kv_cache is None else kv_cache[0].size(2)
        if self.use_rope:
            # the new tokens are at T_past .. T_past+T-1, not at 0 .. T-1 --
            # the same off-by-cache GPT has to get right for an additive
            # position. and this runs BEFORE the append: the past keys in the
            # cache were rotated when they were written, so rotating the
            # concatenated k would rotate them a second time
            cos = self.rope_cos[T_past : T_past + T]
            sin = self.rope_sin[T_past : T_past + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)  # [B, nh, T_kv, hs]
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        T_kv = k.size(2)
        # a window nothing has fallen out of yet is not a window, which keeps
        # both mask-free fast paths alive for the start of every generation
        unwindowed = self.window is None or T_kv <= self.window
        if unwindowed and kv_cache is None:
            # no past: the triangle the kernel builds is exactly ours
            is_causal, attn_mask = True, None
        elif unwindowed and T == 1:
            # one new token, and it may attend to every cached position. the
            # whole row is visible, so there is nothing to mask at all
            is_causal, attn_mask = False, None
        else:
            # everything else has to be spelled out. T > 1 on top of a cache is
            # the case is_causal is quietly WRONG for: the kernel aligns its
            # triangle top-left, assuming query i sits at position i, and ours
            # sit at T_past + i. a window the flag cannot express at all.
            # sliding_window_mask covers both -- it reads the query offset off
            # T_kv - T, and trims from below when window is set
            is_causal = False
            attn_mask = sliding_window_mask(T, T_kv, self.window, x.device)

        # dropout_p must be zeroed by hand at eval: the kernel is a function,
        # it cannot see that the Module around it is in eval mode
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, E)
        out = self.resid_dropout(self.proj(out))
        return (out, new_cache) if use_cache else out


if __name__ == "__main__":
    import time

    from fused_qkv_attention import FusedQKVAttention
    from rope import RopeAttention

    torch.manual_seed(0)
    B, T, E, NH = 2, 8, 32, 4
    x = torch.randn(B, T, E)
    sdpa, fused = SDPAttention(E, NH), FusedQKVAttention(E, NH, T)

    # 1. a drop-in replacement: same parameters, same names, same shapes
    assert sdpa.state_dict().keys() == fused.state_dict().keys()
    assert sum(p.numel() for p in sdpa.parameters()) == sum(
        p.numel() for p in fused.parameters()
    )

    # 2. same weights in, same answer out -- the kernel is not an approximation
    sdpa.load_state_dict(fused.state_dict())
    print("sdpa vs hand-written:", (sdpa(x) - fused(x)).abs().max().item())
    assert (sdpa(x) - fused(x)).abs().max() < 1e-6

    # 3. ... and the same gradients
    xs = x.clone().requires_grad_(True)
    xf = x.clone().requires_grad_(True)
    sdpa(xs).square().sum().backward()
    fused(xf).square().sum().backward()
    assert (xs.grad - xf.grad).abs().max() < 1e-5
    assert all(
        (a.grad - b.grad).abs().max() < 1e-5
        for a, b in zip(sdpa.parameters(), fused.parameters())
    )

    # 4. is_causal=True really is our tril: nothing leaks backwards
    out = sdpa(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        out2 = sdpa(x2)
        assert (out2[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"
        assert (out2[:, t] - out[:, t]).abs().max() > 1e-3
    print("causality holds for all t")

    # dropout: the kernel cannot see self.training, so the guard is load-bearing
    d = SDPAttention(E, NH, dropout=0.5)
    d.train()
    assert not torch.equal(d(x), d(x))
    d.eval()
    assert torch.equal(d(x), d(x))

    # 5. the KV cache, and it must agree with the hand-written one step for step
    full_s, full_f = sdpa(x), fused(x)
    cache_s = cache_f = None
    steps_s, steps_f = [], []
    for t in range(T):
        out_s, cache_s = sdpa(x[:, t : t + 1], cache_s, use_cache=True)
        out_f, cache_f = fused(x[:, t : t + 1], cache_f, use_cache=True)
        steps_s.append(out_s)
        steps_f.append(out_f)
    assert (torch.cat(steps_s, dim=1) - full_s).abs().max() < 1e-6
    assert (torch.cat(steps_s, dim=1) - torch.cat(steps_f, dim=1)).abs().max() < 1e-6
    print("sdpa decodes incrementally, and agrees with the hand-written cache")

    # 6. the case is_causal cannot express: several new tokens on top of a
    #    cache. prefill 3, then hand it 5 more at once
    pre_s, cs = sdpa(x[:, :3], use_cache=True)
    rest_s, cs = sdpa(x[:, 3:], cs, use_cache=True)
    assert (torch.cat([pre_s, rest_s], dim=1) - full_s).abs().max() < 1e-6

    #    and this is what is_causal=True would have given instead -- the kernel
    #    puts its triangle top-left, so query 0 of the continuation is allowed
    #    to see only cached position 0, when it should see all 4 before it
    q_, k_, v_ = (
        t_.view(B, T, NH, E // NH).transpose(1, 2)
        for t_ in sdpa.qkv(x).split(E, dim=-1)
    )
    T_past, T_kv = 3, T

    def attend(**kw) -> Tensor:
        o = F.scaled_dot_product_attention(q_[:, :, T_past:], k_, v_, **kw)
        return sdpa.proj(o.transpose(1, 2).reshape(B, T - T_past, E))

    wrong = attend(is_causal=True)
    assert (wrong - rest_s).abs().max() > 1e-2
    print("is_causal on a cached step is wrong by", (wrong - rest_s).abs().max().item())

    #    and this is the proof that top-left is exactly what it does: build that
    #    mask by hand -- rows 0..T-1 instead of T_past..T_kv-1 -- and the kernel
    #    reproduces its own wrong answer to the bit. the flag is not doing
    #    something subtle, it is indexing our queries from the wrong origin
    j = torch.arange(T_kv)
    top_left = attend(attn_mask=j <= torch.arange(0, T_kv - T_past).unsqueeze(1))
    shifted = attend(attn_mask=j <= torch.arange(T_past, T_kv).unsqueeze(1))
    assert (top_left - wrong).abs().max() == 0  # is_causal IS the top-left band
    assert (shifted - rest_s).abs().max() < 1e-6  # the band forward builds
    print("is_causal == hand-built top-left mask, exactly")

    # 7. rope, wired in. the reference is rope.py's RopeAttention -- same
    #    parameter names, same rotation, no cache -- so with the same weights
    #    the two must agree exactly. that pins the wiring rather than the maths,
    #    which rope.py already tested
    HS = E // NH
    rp = SDPAttention(E, NH, 0.0, T, use_rope=True)
    ref_rope = RopeAttention(E, NH, T)
    assert rp.state_dict().keys() == ref_rope.state_dict().keys()
    rp.load_state_dict(ref_rope.state_dict())
    full_r = rp(x)
    assert (full_r - ref_rope(x)).abs().max() < 1e-6
    #    and it is doing something: same weights without the flag differ
    plain = SDPAttention(E, NH)
    plain.load_state_dict(ref_rope.state_dict())
    assert (full_r - plain(x)).abs().max() > 1e-3
    #    the tables cost no parameters and no checkpoint keys
    assert sum(p.numel() for p in rp.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )
    assert [n for n, _ in rp.named_buffers()] == ["rope_cos", "rope_sin"]

    # 8. rope and the cache, which is the one thing this file can get wrong that
    #    rope.py cannot. the invariant: what goes INTO the cache is already
    #    rotated, by the position the token actually had
    pre_r, cr = rp(x[:, :3], use_cache=True)
    kp = rp.qkv(x[:, :3]).split(E, dim=-1)[1].view(B, 3, NH, HS).transpose(1, 2)
    rotated = apply_rope(kp, rp.rope_cos[:3], rp.rope_sin[:3])
    assert (cr[0] - rotated).abs().max() < 1e-6  # cached k IS the rotated k
    assert (cr[0] - kp).abs().max() > 1e-3  # not the raw projection
    #    so the continuation rotates only its own rows. rotate the concatenated
    #    k instead and the three cached keys pick up a second rotation -- the
    #    forward still runs, still returns plausible logits, and is wrong
    rest_r, _ = rp(x[:, 3:], cr, use_cache=True)
    assert (torch.cat([pre_r, rest_r], dim=1) - full_r).abs().max() < 1e-6
    twice = apply_rope(cr[0], rp.rope_cos[:3], rp.rope_sin[:3])
    assert (twice - cr[0]).abs().max() > 1e-3
    #    and forgetting T_past is the other half of the same bug: rotating the
    #    new rows as if they started at 0 is what an offset-free table gives
    at_zero = apply_rope(kp, rp.rope_cos[:3], rp.rope_sin[:3])  # correct only here
    assert (at_zero - rotated).abs().max() == 0  # T_past = 0 for the prefill
    k5 = rp.qkv(x[:, 3:]).split(E, dim=-1)[1].view(B, T - 3, NH, HS).transpose(1, 2)
    right = apply_rope(k5, rp.rope_cos[3:T], rp.rope_sin[3:T])
    wrong_off = apply_rope(k5, rp.rope_cos[: T - 3], rp.rope_sin[: T - 3])
    assert (right - wrong_off).abs().max() > 1e-3
    print("rope: rotated keys go into the cache, and only the new rows rotate")

    # 9. the sliding window. Everything above kept the mask causal; this makes
    #    it a band. The two ends pin what the flag means, and both are exact:
    #    a window at least as wide as the sequence is the causal mask back, and
    #    a window of 1 lets each position attend to itself alone -- which makes
    #    attention the identity, so the layer collapses to proj(v)
    W = 3
    win = SDPAttention(E, NH, window=W)
    win.load_state_dict(sdpa.state_dict())
    wide = SDPAttention(E, NH, window=T)
    wide.load_state_dict(sdpa.state_dict())
    assert (wide(x) - sdpa(x)).abs().max() == 0
    one = SDPAttention(E, NH, window=1)
    one.load_state_dict(sdpa.state_dict())
    v_only = one.qkv(x).split(E, dim=-1)[2]
    assert (one(x) - one.proj(v_only)).abs().max() < 1e-6
    #    ...and in between it is a genuinely different function, not a rounding
    #    difference. this rung changes the model's answers, which is why it
    #    needs an oracle rather than a "nothing broke" check
    assert (win(x) - sdpa(x)).abs().max() > 1e-2

    #    the oracle: the same attention built by hand out of eager ops, with
    #    the band applied to the scores before the softmax. no SDPA, no flags,
    #    nothing shared with the code under test except the weights
    def band_oracle(layer: SDPAttention, xs: Tensor, window: int | None) -> Tensor:
        Bo, To, Eo = xs.shape
        nh_, hs_ = layer.n_head, layer.head_size
        qo, ko, vo = (
            t_.view(Bo, To, nh_, hs_).transpose(1, 2)
            for t_ in layer.qkv(xs).split(Eo, dim=-1)
        )
        if layer.use_rope:  # v is never rotated -- position lives in the scores
            cos_, sin_ = layer.rope_cos[:To], layer.rope_sin[:To]
            qo, ko = apply_rope(qo, cos_, sin_), apply_rope(ko, cos_, sin_)
        m_ = sliding_window_mask(To, To, window)
        w_ = (qo @ ko.transpose(-2, -1) * hs_**-0.5).masked_fill(~m_, -torch.inf)
        o_ = (w_.softmax(dim=-1) @ vo).transpose(1, 2).reshape(Bo, To, Eo)
        return layer.proj(o_)

    assert (win(x) - band_oracle(win, x, W)).abs().max() < 1e-6
    assert (sdpa(x) - band_oracle(sdpa, x, None)).abs().max() < 1e-6
    print(f"window W={W} matches a hand-built band mask, and W>=T is plain causal")

    # 10. the window and the cache, which is where the fast paths can lie. the
    #     layer skips the mask entirely while T_kv <= window, and hands SDPA a
    #     mask after that -- so a decode crosses a branch boundary mid-run and
    #     both sides have to land on the same number. one token at a time, and
    #     then the prefill-plus-continuation split, against the parallel forward
    full_w = win(x)
    cache, steps = None, []
    for t in range(T):
        step, cache = win(x[:, t : t + 1], cache, use_cache=True)
        steps.append(step)
    assert (torch.cat(steps, dim=1) - full_w).abs().max() < 1e-6
    for split in (1, W, W + 1, T - 1):  # either side of the boundary, and on it
        pre, c = win(x[:, :split], use_cache=True)
        rest, c = win(x[:, split:], c, use_cache=True)
        assert (torch.cat([pre, rest], dim=1) - full_w).abs().max() < 1e-6, split
    #     and the cache is NOT smaller: rung 1 is a mask, so every position the
    #     window stopped looking at is still sitting in memory. that is the
    #     whole gap the ring buffer exists to close
    assert c[0].shape == (B, NH, T, E // NH)
    print("windowed decode agrees with the parallel forward across the fast paths")

    # 11. the window narrows what a position sees, and does not weaken what it
    #     may NOT see. perturb position t: nothing before t moves (causality),
    #     and nothing more than W-1 positions after t moves either -- the band
    #     is closed from both sides, which is the property depth then rebuilds
    out = win(x)
    for t in range(T):
        x2 = x.clone()
        x2[:, t] += 10.0
        d = (win(x2) - out).abs().amax(dim=(0, 2))
        assert (d[:t] == 0).all(), f"leak backwards at t={t}"
        assert d[t] > 1e-3
        assert (d[t + W :] == 0).all(), f"leak past the window at t={t}"
    print(f"one layer of W={W} moves exactly {W} positions")

    # 12. rope and the window together, which is the pair that makes rung 1
    #     legitimate. rotations are baked in at write time by ABSOLUTE position,
    #     so dropping columns from the mask cannot desynchronise the ones left:
    #     the score for a key W-1 back is R(-(W-1)) whether it is key 3 or key
    #     3000. decode has to reproduce the parallel forward here too
    rw = SDPAttention(E, NH, 0.0, T, use_rope=True, window=W)
    rw.load_state_dict(ref_rope.state_dict())
    full_rw = rw(x)
    assert (full_rw - band_oracle(rw, x, W)).abs().max() < 1e-6
    cache, steps = None, []
    for t in range(T):
        step, cache = rw(x[:, t : t + 1], cache, use_cache=True)
        steps.append(step)
    assert (torch.cat(steps, dim=1) - full_rw).abs().max() < 1e-6
    assert (full_rw - rp(x)).abs().max() > 1e-2  # the window is doing something
    print("rope + window decodes incrementally: baked rotations survive eviction")

    if not torch.cuda.is_available():
        print("ok (cpu -- skipped the memory and backend tests)")
        raise SystemExit

    # 13. the point of the whole file: memory. Ours allocates [B, nh, T, T] and
    #    several copies of it; SDPA allocates none, so it scales with T not T^2
    print(
        f"\npeak memory for one forward, B=4 nh=8 E=512  ({torch.cuda.get_device_name(0)})"
    )
    print(f"  {'T':>6}  {'hand-written':>14}  {'sdpa':>10}  {'B*nh*T*T fp32':>14}")
    for Tb in (512, 1024, 2048):
        xb = torch.randn(4, Tb, 512, device="cuda")
        peak = {}
        for name, m in (
            ("fused", FusedQKVAttention(512, 8, Tb).to("cuda")),
            ("sdpa", SDPAttention(512, 8).to("cuda")),
        ):
            m(xb)  # warm up allocator
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            m(xb)
            peak[name] = (torch.cuda.max_memory_allocated() - before) / 2**20
        scores_mb = 4 * 8 * Tb * Tb * 4 / 2**20
        print(
            f"  {Tb:>6}  {peak['fused']:>11.0f} MB  {peak['sdpa']:>7.0f} MB  {scores_mb:>11.0f} MB"
        )
        assert peak["sdpa"] < peak["fused"]
    """
    Test 13 is the table to hold on screen. Doubling T roughly quadruples the hand-written column (133 → 458 → 1689 MB) and exactly doubles SDPA's (24 → 48 → 96 MB). That's T² versus T, measured, not asserted. The last column shows why: one copy of the score matrix at T=2048 is 512 MB, and we hold several at once — scores, the masked version, the softmax output. At T=2048 this layer alone is 1.7 GB on an 8 GB card, which is the actual reason long-context models were impossible before this kernel.
    """

    # 14. and it is faster, which is a consequence of the same thing -- fewer
    #    round trips to HBM, not fewer floating point operations
    xb = torch.randn(4, 1024, 512, device="cuda")
    t = {}
    for name, m in (
        ("fused", FusedQKVAttention(512, 8, Tb).to("cuda")),
        ("sdpa", SDPAttention(512, 8).to("cuda")),
    ):
        for _ in range(3):
            m(xb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            m(xb)
        torch.cuda.synchronize()
        t[name] = (time.perf_counter() - t0) / 20 * 1e3
    print(
        f"\nforward at T=1024 -- hand-written {t['fused']:.2f} ms   "
        f"sdpa {t['sdpa']:.2f} ms   {t['fused'] / t['sdpa']:.1f}x"
    )
    assert t["sdpa"] < t["fused"]

    """
    Test 14's 4.4x is worth framing correctly: SDPA does the same number of FLOPs. It wins by not shipping 512 MB out to HBM and back between each step. That's the whole of FlashAttention's idea — it's an IO story, not an arithmetic one.
    """

    # 15. "flash" is a backend, not this layer. In fp32 it is not even available
    import warnings

    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(4, 8, 512, 64, device="cuda")
    print("\nwhich backends accept these inputs:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch narrates every backend it rejects
        """
        Not suppressing warnings prints this Error when float32 tries to use FLASH_ATTENTION
        
            UserWarning: Expected query, key and value to all be of dtype: {Half, BFloat16}. Got Query dtype: float, Key dtype: float, and Value dtype: float instead.
        """
        for dt in (torch.float32, torch.bfloat16):
            ok = []
            for be in (
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ):
                try:
                    with sdpa_kernel(be):
                        F.scaled_dot_product_attention(
                            q.to(dt), q.to(dt), q.to(dt), is_causal=True
                        )
                    ok.append(be.name)
                except RuntimeError:
                    pass
            print(f"  {dt!s:<16} {', '.join(ok)}")

    # 16. and the rest of the speedup is behind the DTYPE, not behind this file.
    #    flash needs fp16/bf16, so in fp32 it is unavailable on every card --
    #    including this one, which supports bf16 perfectly well. The model just
    #    is not running in it. Mixed precision is a separate change.
    print(f"\nbf16 supported on this GPU: {torch.cuda.is_bf16_supported()}")
    m = SDPAttention(512, 8).to("cuda")
    xb = torch.randn(4, 1024, 512, device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dt in (torch.float32, torch.bfloat16):
            xd = xb.to(dt)
            m.to(dt)
            for _ in range(3):
                m(xd)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            for _ in range(20):
                m(xd)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1e3
            mem = (torch.cuda.max_memory_allocated() - base) / 2**20
            try:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    m(xd)
                flash = "flash RUNS"
            except RuntimeError:
                flash = "flash unavailable"
            print(f"  {dt!s:<16} {ms:5.2f} ms   {mem:>3.0f} MB peak   {flash}")

    print("\nok")
