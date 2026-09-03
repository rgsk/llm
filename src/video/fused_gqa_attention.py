import torch
from common import KVCache
from dropout import Dropout
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from softmax import softmax
from torch import Tensor


class FusedGQAttention(Module):
    """FusedQKVAttention with fewer key/value heads than query heads.

    An artifact of the series, not the layer to build on: it exists because the
    eager path is where grouped-query attention is legible. gqa_attention.py is
    the one to use, and it gets the same result from one flag.

    Everything about the fusion and the KV cache is fused_qkv_attention.py's
    story and is unchanged here. What that file ends on is the cache's price --
    2 * n_layer * B * T * E values held in memory, re-read once per generated
    token, which is why decoding is bandwidth-bound.

    n_kv_head is the knob on that memory. Query heads are what a position asks;
    keys and values are what the sequence offers, and nothing says you need one
    set of offers per question. Project n_kv_head < n_head of them and let a
    group of n_rep = n_head // n_kv_head query heads share a pair: the cache
    shrinks by exactly n_rep, and so does the k/v half of the qkv matrix.

        n_kv_head == n_head    -> MHA, the default, and identical to before
        1 < n_kv_head < n_head -> GQA (Ainslie 2023), what nearly everything ships
        n_kv_head == 1         -> MQA (Shazeer 2019), one k/v for the whole layer

    Unlike the cache, this changes the function -- parameters are removed and
    capacity goes with them. It is a trade, and a cheap one, because nothing
    about the ATTENTION shrinks: there are still nh * T * T_kv scores. Every
    query head keeps its own full row of attention weights and its own slice of
    the output. All that is shared is where a group looks and what it collects.

    Which leaves the question of how to attend with mismatched head counts, and
    it has an obvious answer and a right one. The obvious answer is to widen k
    and v back to nh heads with repeat_interleave. The right one is to leave
    them narrow and fold the group into q instead: the scores become
    [B, nkv, n_rep*T, T_kv] rather than [B, nh, T, T_kv] -- the same numbers,
    reshaped, since query heads of a group all attend over the same k. Widening
    rebuilds, once per layer per token, the very tensor the cache exists not to
    store -- and measured on this card it hands back the entire decode speedup,
    landing within noise of the MHA it was supposed to beat. The storage saving
    survives; the latency saving does not. Tests 8 and 9 measure it.
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        n_kv_head: int | None = None,
    ):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        n_kv_head = n_head if n_kv_head is None else n_kv_head
        assert n_head % n_kv_head == 0, (
            "n_kv_head must divide n_head: every kv head serves the same group size"
        )
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head  # q heads per kv head
        self.head_size = n_embed // n_head
        # [E -> E + 2*kv_dim]. at n_kv_head == n_head that is 3E again, split the
        # same [E, E, E] way, so the default layer and its checkpoints are unmoved
        self.qkv = Linear(n_embed, n_embed + 2 * n_kv_head * self.head_size, bias=False)
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
        nh, nkv, hs = self.n_head, self.n_kv_head, self.head_size
        # one matmul, then split into q [B, T, E] and k, v [B, T, nkv*hs]
        qkv = self.qkv(x)
        q, k, v = qkv.split([E, nkv * hs, nkv * hs], dim=-1)

        # split the last axis into heads and move them next to the batch. q gets
        # nh of them, k and v only nkv -- a group of queries per key/value pair
        q = q.view(B, T, nh, hs).transpose(1, 2)  # [B, nh,  T, hs]
        k = k.view(B, T, nkv, hs).transpose(1, 2)  # [B, nkv, T, hs]
        v = v.view(B, T, nkv, hs).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            # [B, nkv, T_past, hs] ++ [B, nkv, T, hs] -> [B, nkv, T_kv, hs]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)  # the NARROW k and v: this is what GQA is for

        T_kv = k.size(2)
        T_past = T_kv - T
        assert T_kv <= self.tril.size(0), "sequence outgrew block_size"

        # k and v stay NARROW -- attend by folding the group into q instead.
        # kv head g owns the contiguous q-head block g*n_rep .. (g+1)*n_rep-1
        # (that is what interleaving means; .repeat would stripe them), so those
        # heads reshape into one stack of rows sharing g's keys. at n_rep == 1
        # this is the plain [B, nh, T, hs] the layer always had
        q = q.reshape(B, nkv, self.n_rep * T, hs)  # [B, nkv, n_rep*T, hs]

        scores = q @ k.transpose(-2, -1) * hs**-0.5  # [B, nkv, n_rep*T, T_kv]
        # q rows are positions T_past .. T_kv-1, so the mask is that row band of
        # tril, not its top-left corner. With T == 1 the band is a single row of
        # all True: the newest token may look at every cached position. Rows now
        # come in n_rep head-blocks, each holding the same T positions, so the
        # band tiles: .repeat, because the blocks are heads, not positions
        causal = self.tril[T_past:T_kv, :T_kv].repeat(self.n_rep, 1)  # [n_rep*T, T_kv]
        scores = scores.masked_fill(~causal, float("-inf"))
        w = softmax(scores, dim=-1)
        w = self.attn_dropout(w)
        out = w @ v  # [B, nkv, n_rep*T, hs]
        out = out.view(B, nh, T, hs)  # ungroup: row block j is head g*n_rep + j
        out = out.transpose(1, 2).reshape(B, T, E)  # heads back side by side
        out = self.resid_dropout(self.proj(out))
        return (out, new_cache) if use_cache else out


if __name__ == "__main__":
    import time

    from fused_qkv_attention import FusedQKVAttention

    torch.manual_seed(0)
    B, T, E, NH = 2, 8, 32, 4
    HS = E // NH
    x = torch.randn(B, T, E)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Eb, NHb = 512, 8  # a realistic layer, for the measurements at the end

    # 1. the anchor: with n_kv_head == n_head this IS FusedQKVAttention. qkv is
    #    [3E, E] again and the split sizes [E, E, E] are what .split(E) already
    #    did, so the weights do not merely match in shape, they mean the same
    #    thing and load across. Everything below is measured against this
    mha = FusedGQAttention(E, NH, T)
    ref = FusedQKVAttention(E, NH, T)
    assert mha.n_kv_head == NH and mha.n_rep == 1
    assert mha.qkv.weight.shape == ref.qkv.weight.shape == (3 * E, E)
    assert mha.state_dict().keys() == ref.state_dict().keys()
    mha.load_state_dict(ref.state_dict())
    assert (mha(x) - ref(x)).abs().max() < 1e-6

    xg, xr = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    mha(xg).square().sum().backward()
    ref(xr).square().sum().backward()
    assert (xg.grad - xr.grad).abs().max() < 1e-6
    assert all(
        (a.grad - b.grad).abs().max() < 1e-6
        for a, b in zip(mha.parameters(), ref.parameters())
    )
    print("n_kv_head == n_head is FusedQKVAttention, weights and grads included")

    # 2. what GQA changes is one axis. q is untouched, k and v get narrower
    gqa = FusedGQAttention(E, NH, T, n_kv_head=2)
    mqa = FusedGQAttention(E, NH, T, n_kv_head=1)
    assert gqa.n_rep == 2 and mqa.n_rep == NH
    assert gqa.qkv.weight.shape == (E + 2 * (2 * HS), E)
    assert mqa.qkv.weight.shape == (E + 2 * HS, E)
    assert gqa(x).shape == mqa(x).shape == (B, T, E)
    try:
        FusedGQAttention(E, NH, T, n_kv_head=3)  # 3 does not divide 4
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "n_kv_head" in str(e)

    # 3. the grouping, by hand, against the SMALL k and v -- q head h attends
    #     with kv head h // n_rep, and there is no repeat_interleave in sight
    q_, k_, v_ = gqa.qkv(x).split([E, 2 * HS, 2 * HS], dim=-1)
    q_ = q_.view(B, T, NH, HS).transpose(1, 2)
    k_ = k_.view(B, T, 2, HS).transpose(1, 2)
    v_ = v_.view(B, T, 2, HS).transpose(1, 2)
    heads = []
    for h in range(NH):
        g = h // gqa.n_rep
        s_ = q_[:, h] @ k_[:, g].transpose(-2, -1) * HS**-0.5  # [B, T, T]
        w_ = softmax(s_.masked_fill(~gqa.tril[:T, :T], float("-inf")), dim=-1)
        heads.append(w_ @ v_[:, g])
    assert (gqa.proj(torch.cat(heads, dim=-1)) - gqa(x)).abs().max() < 1e-6
    print("q head h attends with kv head h // n_rep")

    # 4. and this is what sharing does NOT mean. the scores are still
    #     [B, nh, T, T_kv]: every query head keeps its own attention pattern and
    #     its own output slice. zero kv head 0's KEY rows and exactly the two
    #     heads in its group move -- the other group is untouched to the bit
    probe_g = FusedGQAttention(E, NH, T, n_kv_head=2)
    with torch.no_grad():
        probe_g.proj.weight.copy_(torch.eye(E))  # so slices stay separable
        probe_g.proj.bias.zero_()
        before_g = probe_g(x).clone()
        probe_g.qkv.weight[E : E + HS].zero_()  # q occupies [0, E), keys start there
    after_g = probe_g(x)
    moved = [
        (after_g[..., h * HS : (h + 1) * HS] - before_g[..., h * HS : (h + 1) * HS])
        .abs()
        .max()
        .item()
        for h in range(NH)
    ]
    assert moved[0] > 1e-3 and moved[1] > 1e-3  # group 0 = q heads 0, 1
    assert moved[2] < 1e-6 and moved[3] < 1e-6  # group 1 never saw that key
    print("per-head change after zeroing kv head 0:", [f"{m:.3f}" for m in moved])

    # 5. sharing k/v does not weaken causality: that is a claim about columns
    #    of the score matrix, and grouping only touches rows
    out = gqa(x)
    for t_ in range(1, T):
        x2 = x.clone()
        x2[:, t_] += 10.0
        out2 = gqa(x2)
        assert (out2[:, :t_] - out[:, :t_]).abs().max() < 1e-5, f"leak at t={t_}"
        assert (out2[:, t_] - out[:, t_]).abs().max() > 1e-3
    print("causality holds for all t")

    d = FusedGQAttention(E, NH, T, dropout=0.5, n_kv_head=2)
    d.train()
    assert not torch.equal(d(x), d(x))
    d.eval()
    assert torch.equal(d(x), d(x))

    # 6. the payoff, and the cache protocol is unchanged: still one tuple in,
    #     one tuple out, just n_rep times less of it
    for m, nkv in ((mha, NH), (gqa, 2), (mqa, 1)):
        full_m = m(x)
        cache_m, steps_m = None, []
        for t_ in range(T):
            out_t, cache_m = m(x[:, t_ : t_ + 1], cache_m, use_cache=True)
            steps_m.append(out_t)
        assert cache_m[0].shape == cache_m[1].shape == (B, nkv, T, HS)
        assert (torch.cat(steps_m, dim=1) - full_m).abs().max() < 1e-6
        #    and the case that exercises both mask corrections at once: several
        #    new tokens on top of a cache, so the band is shifted down by T_past
        #    AND tiled across n_rep head blocks. get either wrong and this is
        #    the only test that notices
        pre_m, c_m = m(x[:, :3], use_cache=True)
        rest_m, c_m = m(x[:, 3:], c_m, use_cache=True)
        assert (torch.cat([pre_m, rest_m], dim=1) - full_m).abs().max() < 1e-6
        n_cached = sum(c.numel() for c in cache_m)
        kind = {NH: "mha", 1: "mqa"}.get(nkv, "gqa")
        print(
            f"  n_kv_head={nkv}  {kind}  cache {n_cached:>5} values  "
            f"{2 * B * NH * T * HS / n_cached:.0f}x smaller"
        )

    # 7. the same weights read the same way by the SDPA version: one matrix
    #     laid out [E, nkv*hs, nkv*hs], two implementations, one answer
    from gqa_attention import GQAttention

    sdpa_gqa = GQAttention(E, NH, 2)
    assert sdpa_gqa.state_dict().keys() == gqa.state_dict().keys()
    sdpa_gqa.load_state_dict(gqa.state_dict())
    assert (sdpa_gqa(x) - gqa(x)).abs().max() < 1e-6
    print("eager GQA agrees with the SDPA one")

    # 8. the other way to write this layer, kept because the comparison below
    #     needs it: widen k and v to nh heads and attend head-for-head. It is
    #     the obvious implementation, it is what this file did first, and it is
    #     not wrong -- same numbers, cached and uncached, to 1e-6
    def widened(m, x_, kv_cache=None):
        """forward, expanding k and v to nh heads instead of folding q into nkv."""
        Bw, Tw, Ew = x_.shape
        nkvw, hsw = m.n_kv_head, m.head_size
        qw, kw, vw = m.qkv(x_).split([Ew, nkvw * hsw, nkvw * hsw], dim=-1)
        qw = qw.view(Bw, Tw, m.n_head, hsw).transpose(1, 2)
        kw = kw.view(Bw, Tw, nkvw, hsw).transpose(1, 2)
        vw = vw.view(Bw, Tw, nkvw, hsw).transpose(1, 2)
        if kv_cache is not None:
            kw = torch.cat([kv_cache[0], kw], dim=2)
            vw = torch.cat([kv_cache[1], vw], dim=2)
        if m.n_rep > 1:  # the line the whole comparison is about
            kw = kw.repeat_interleave(m.n_rep, dim=1)  # [B, nh, T_kv, hs]
            vw = vw.repeat_interleave(m.n_rep, dim=1)
        T_kvw = kw.size(2)
        sw = qw @ kw.transpose(-2, -1) * hsw**-0.5  # [B, nh, T, T_kv]
        sw = sw.masked_fill(~m.tril[T_kvw - Tw : T_kvw, :T_kvw], float("-inf"))
        ow = softmax(sw, dim=-1) @ vw
        return m.proj(ow.transpose(1, 2).reshape(Bw, Tw, Ew))

    for m in (mha, gqa, mqa):
        assert (widened(m, x) - m(x)).abs().max() < 1e-6
        _, c22 = m(x[:, :5], use_cache=True)  # and on top of a cache, where the
        step22, _ = m(x[:, 5:6], c22, use_cache=True)  # two mask shapes diverge
        assert (widened(m, x[:, 5:6], c22) - step22).abs().max() < 1e-6
    print("widening k/v computes the same thing as folding q")

    # 9. ...and it does not cost the same. Decode is bandwidth-bound, and
    #     widening reads the narrow cache, writes an n_rep times bigger copy,
    #     then reads THAT back -- undoing per step exactly what the smaller
    #     cache bought. The storage saving survives it; the speed saving does
    #     not, and the n_kv_head=8 row is the baseline that shows how little is
    #     left. This is the whole reason forward folds q instead.
    #
    #     Two things to read honestly off the table. The MB column is not the
    #     copy alone: every path pays a torch.cat that reallocates the WHOLE
    #     cache each step, which is why even mha-fold shows 134 MB and why real
    #     serving code preallocates a buffer and writes into it. And the cache
    #     here is synthesised rather than prefilled -- both paths get identical
    #     tensors, and an eager 2048-token prefill would build a score matrix
    #     bigger than this card
    HSb = Eb // NHb
    Bd, Tkv = (16, 2048) if dev == "cuda" else (2, 128)

    def bench(fn, n=20) -> tuple[float, float]:
        for _ in range(3):
            fn()
        base = 0
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        mb = (
            (torch.cuda.max_memory_allocated() - base) / 2**20 if dev == "cuda" else 0.0
        )
        return ms, mb

    print(f"\n  ONE DECODE STEP, {dev}: B={Bd}, nh={NHb}, hs={HSb}, T_kv={Tkv}")
    print(
        f"  {'n_kv_head':>9}  {'kind':>4}  {'cache held':>10}  {'widen k/v':>18}  "
        f"{'fold q (this file)':>18}  {'gain':>6}"
    )
    speedups = []
    for nkv in (NHb, 2, 1):
        m = FusedGQAttention(Eb, NHb, Tkv + 1, n_kv_head=nkv).to(dev)
        cache = (
            torch.randn(Bd, nkv, Tkv, HSb, device=dev),
            torch.randn(Bd, nkv, Tkv, HSb, device=dev),
        )
        xd1 = torch.randn(Bd, 1, Eb, device=dev)  # one new token, mid-stream
        with torch.no_grad():
            assert (
                m(xd1, cache, use_cache=True)[0] - widened(m, xd1, cache)
            ).abs().max() < 1e-4
            # default args, so the lambdas bind this iteration's layer and cache
            w_ms, w_mb = bench(lambda m=m, x_=xd1, c=cache: widened(m, x_, c))
            f_ms, f_mb = bench(lambda m=m, x_=xd1, c=cache: m(x_, c, use_cache=True))
        held = sum(c.numel() * c.element_size() for c in cache) / 2**20
        kind = {NHb: "mha", 1: "mqa"}.get(nkv, "gqa")
        print(
            f"  {nkv:>9}  {kind:>4}  {held:>7.1f} MB  {w_ms:>7.2f} ms {w_mb:>6.1f} MB  "
            f"{f_ms:>7.2f} ms {f_mb:>6.1f} MB  {w_ms / f_ms:>6.1f}x"
        )
        if nkv == NHb:
            mha_ms = f_ms  # n_rep == 1: both columns are the same code path
        else:
            assert f_ms < w_ms
            speedups.append((nkv, mha_ms / w_ms, mha_ms / f_ms))
        del m, cache, xd1

    #     and this is the line that matters: against the mha baseline, widening
    #     keeps almost none of the speed, while folding keeps all of it
    for nkv, w_x, f_x in speedups:
        print(
            f"    n_kv_head={nkv} vs mha ({mha_ms:.2f} ms):  widen {w_x:.1f}x   fold {f_x:.1f}x"
        )

    #     prefill is the other half of the honest answer: there the T*T_kv
    #     attention math dominates and the copy barely registers. This trade
    #     only bites at T=1, which is exactly where models spend their time
    Bp, Tp = (2, 256) if dev == "cuda" else (2, 64)
    mp = FusedGQAttention(Eb, NHb, Tp, n_kv_head=2).to(dev)
    xp = torch.randn(Bp, Tp, Eb, device=dev)
    with torch.no_grad():
        wp_ms, wp_mb = bench(lambda: widened(mp, xp))
        fp_ms, fp_mb = bench(lambda: mp(xp))
    print(
        f"\n  PREFILL B={Bp}, T={Tp}, n_kv_head=2:  widen {wp_ms:.2f} ms {wp_mb:.1f} MB"
        f"   fold {fp_ms:.2f} ms {fp_mb:.1f} MB   {wp_ms / fp_ms:.2f}x"
    )

    print("\nok")
