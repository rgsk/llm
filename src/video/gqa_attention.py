import torch
import torch.nn.functional as F
from torch import Tensor

from dropout import Dropout
from kv_cache import KVCache
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from rope import apply_rope, rope_tables


class GQAttention(Module):
    """SDPAttention with fewer key/value heads than query heads.

    The KV cache made decoding fast and made it expensive: 2 * n_layer * B * T_kv
    * n_head * head_size numbers, re-read from memory for every single token
    generated. Decoding is bandwidth-bound, and the cache *is* the bandwidth. The
    obvious lever is to make it smaller.

    So: keep all n_head query heads, project only n_kv_head keys and values, and
    let a group of q heads share one k/v pair. Query head h reads kv head
    h // n_rep, where n_rep = n_head // n_kv_head. The cache shrinks by exactly
    n_rep, and so does the qkv matrix's k and v half.

        n_kv_head == n_head  ->  MHA, this file is SDPAttention with extra steps
        n_kv_head == 1       ->  MQA (Shazeer 2019), one k/v for the whole layer
        1 < n_kv_head < n_head -> GQA (Ainslie 2023), the middle everyone ships

    Two things to be clear about, because both are easy to state wrongly.

    1. Unlike SDPA, this is NOT the same arithmetic. SDPA computed the identical
       function faster; this computes a different, smaller function. Parameters
       are removed and capacity goes with them. GQA is a trade, and the reason
       it is everywhere is that the curve is lopsided -- MQA loses noticeable
       quality, MQA -> GQA-8 wins most of it back for 1/8th of the MHA cache.
       What is shared is only where each head *looks* and what it *collects*;
       the queries stay fully independent, and so does the output projection.

    2. Nothing is expanded back to n_head at all. The obvious way to attend
       with mismatched head counts is to repeat_interleave k and v up to nh --
       correct, and it rebuilds once per token the very tensor the small cache
       exists not to store, which costs back the whole decode speedup (measured
       in fused_gqa_attention.py, test 9). enable_gqa=True has the kernel
       broadcast instead, and it never materialises the copy. The eager file
       reaches the same place by folding the group into q; here it is a flag.

    use_rope=True adds rotary positions, and the mismatched head counts cost
    nothing: the rotation acts per head on (T, head_size), so a k with n_rep
    fewer heads rotates by exactly the same table. What shrinks is the rotated
    cache -- fewer keys to rotate going in, and fewer to re-read every step.
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int | None = None,
        dropout: float = 0.0,
        block_size: int | None = None,
        use_rope: bool = False,
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
        kv_dim = n_kv_head * self.head_size  # < n_embed unless this is MHA
        # the projection stops being square: [E -> E + 2*kv_dim], not [E -> 3E]
        self.qkv = Linear(n_embed, n_embed + 2 * kv_dim, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        self.dropout_p = dropout
        self.resid_dropout = Dropout(dropout)
        # only needed to size a preallocated cache. None keeps the old
        # grow-by-copying behaviour, which is what training uses anyway
        self.block_size = block_size
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
        nh, nkv, hs = self.n_head, self.n_kv_head, self.head_size
        # three unequal pieces now, so split takes a list of sizes
        q, k, v = self.qkv(x).split([E, nkv * hs, nkv * hs], dim=-1)

        q = q.view(B, T, nh, hs).transpose(1, 2)  # [B, nh,  T, hs]
        k = k.view(B, T, nkv, hs).transpose(1, 2)  # [B, nkv, T, hs]
        v = v.view(B, T, nkv, hs).transpose(1, 2)

        if self.use_rope:
            # q is [B, nh, T, hs] and k is [B, nkv, T, hs], and the rotation
            # does not care: it acts on (T, hs) and broadcasts over whatever
            # head axis it is handed. grouping is a fact about heads, rotation
            # is a fact about positions, and the two never meet -- the same
            # reason the causal mask needed no change for GQA.
            # before the append, so the cache holds keys already rotated
            T_past = 0 if kv_cache is None else kv_cache.pos
            cos = self.rope_cos[T_past : T_past + T]
            sin = self.rope_sin[T_past : T_past + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if use_cache and kv_cache is None:
            # the layer allocates it because only the layer knows n_kv_head and
            # head_size; B, device and dtype come from the data on first write.
            # capacity is block_size, and without one this falls back to the
            # tuple's grow-by-copying
            shape = None if self.block_size is None else (B, nkv, self.block_size, hs)
            kv_cache = KVCache(shape)
        if kv_cache is not None:
            # the NARROW k and v go in -- [B, nkv, T_kv, hs], n_rep smaller
            k, v = kv_cache.append(k, v)

        T_kv = k.size(2)
        T_past = T_kv - T
        if kv_cache is None:
            is_causal, attn_mask = True, None
        elif T == 1:
            is_causal, attn_mask = False, None
        else:
            # queries sit at T_past .. T_kv-1, and is_causal aligns top-left.
            # same shift SDPAttention needed; grouping does not touch the mask,
            # which is a fact about positions, not about heads
            i = torch.arange(T_past, T_kv, device=x.device).unsqueeze(1)
            j = torch.arange(T_kv, device=x.device)
            is_causal, attn_mask = False, j <= i  # [T, T_kv], True = visible

        # k and v stay narrow. enable_gqa tells the kernel that q has n_rep
        # times more heads and to broadcast kv head g across the contiguous q
        # block g*n_rep .. (g+1)*n_rep-1 -- the same grouping test 3 computes by
        # hand, done without ever materialising the wide copy
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
            enable_gqa=self.n_rep > 1,
        )

        out = out.transpose(1, 2).reshape(B, T, E)
        out = self.resid_dropout(self.proj(out))
        return (out, kv_cache) if use_cache else out


if __name__ == "__main__":
    from sdpa_attention import SDPAttention

    torch.manual_seed(0)
    B, T, E, NH = 2, 8, 32, 4
    HS = E // NH
    x = torch.randn(B, T, E)

    # 1. the degenerate case has to be exactly the old layer. n_kv_head == n_head
    #    makes qkv square again, with the same [E, E, E] split, so the weights
    #    are not merely the same shape -- they mean the same thing, and load
    #    across. if this drifts, nothing below is measuring what it claims
    mha = GQAttention(E, NH, NH)
    sdpa = SDPAttention(E, NH)
    assert mha.state_dict().keys() == sdpa.state_dict().keys()
    assert all(a.shape == b.shape for a, b in zip(mha.parameters(), sdpa.parameters()))
    mha.load_state_dict(sdpa.state_dict())
    assert (mha(x) - sdpa(x)).abs().max() < 1e-6
    assert mha.n_rep == 1

    #    ...gradients too, since a forward-only match can hide a broken view
    xg, xs = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    mha(xg).square().sum().backward()
    sdpa(xs).square().sum().backward()
    assert (xg.grad - xs.grad).abs().max() < 1e-6
    assert all(
        (a.grad - b.grad).abs().max() < 1e-6
        for a, b in zip(mha.parameters(), sdpa.parameters())
    )
    print("n_kv_head == n_head is SDPAttention, weights and grads included")

    # 2. what each setting costs. only the k and v halves of qkv shrink; q and
    #    proj are untouched, which is why the parameter saving is milder than
    #    the cache saving people quote
    print(f"\n  {'n_kv_head':>9}  {'kind':>4}  {'qkv weight':>12}  {'params':>7}")
    for nkv in (NH, 2, 1):
        m = GQAttention(E, NH, nkv)
        kind = {NH: "mha", 1: "mqa"}.get(nkv, "gqa")
        assert m.qkv.weight.shape == (E + 2 * nkv * HS, E)
        assert m(x).shape == (B, T, E)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {nkv:>9}  {kind:>4}  {tuple(m.qkv.weight.shape)!s:>12}  {n:>7}")

    #    and a group size that does not divide is rejected at construction, not
    #    at the reshape three lines into forward
    try:
        GQAttention(E, NH, 3)  # 3 does not divide 4
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "n_kv_head" in str(e)

    # 3. the grouping is the whole claim, so compute it by hand: q head h against
    #    the SMALL kv head h // n_rep, per head, no repeat_interleave anywhere
    gqa = GQAttention(E, NH, 2)
    q, k, v = gqa.qkv(x).split([E, 2 * HS, 2 * HS], dim=-1)
    q = q.view(B, T, NH, HS).transpose(1, 2)
    k = k.view(B, T, 2, HS).transpose(1, 2)
    v = v.view(B, T, 2, HS).transpose(1, 2)
    tril = torch.ones(T, T, dtype=torch.bool).tril()

    def by_hand(kv_of_head) -> Tensor:
        heads = []
        for h in range(NH):
            g = kv_of_head(h)
            s = q[:, h] @ k[:, g].transpose(-2, -1) * HS**-0.5  # [B, T, T]
            w = s.masked_fill(~tril, float("-inf")).softmax(dim=-1)
            heads.append(w @ v[:, g])
        return gqa.proj(torch.cat(heads, dim=-1))  # heads side by side -> [B, T, E]

    assert (by_hand(lambda h: h // gqa.n_rep) - gqa(x)).abs().max() < 1e-6
    #    and the off-by-one-convention version is a genuinely different answer,
    #    not a rounding difference. this is repeat_interleave vs repeat
    assert (by_hand(lambda h: h % gqa.n_kv_head) - gqa(x)).abs().max() > 1e-2
    print("q head h attends with kv head h // n_rep, and striping is not the same")

    #    ...and sharing a k/v pair does not merge the heads that share it. every
    #    q head still gets its own row of attention weights and its own slice of
    #    the output -- what is shared is where the group looks, not what it
    #    computes. set proj to the identity so the slices stay separable, zero
    #    kv head 0's KEY rows, and exactly its two q heads move
    probe = GQAttention(E, NH, 2)
    with torch.no_grad():
        probe.proj.weight.copy_(torch.eye(E))
        probe.proj.bias.zero_()
        before = probe(x).clone()
        probe.qkv.weight[E : E + HS].zero_()  # q occupies [0, E), keys start there
    after = probe(x)
    moved = [
        (after[..., h * HS : (h + 1) * HS] - before[..., h * HS : (h + 1) * HS])
        .abs()
        .max()
        .item()
        for h in range(NH)
    ]
    assert moved[0] > 1e-3 and moved[1] > 1e-3  # group 0 = q heads 0, 1
    assert moved[2] < 1e-6 and moved[3] < 1e-6  # group 1 never saw that key
    print(
        "one kv head serves exactly n_rep query heads:", [f"{m_:.3f}" for m_ in moved]
    )

    # 4. sharing k/v does not weaken causality: it is a statement about columns
    out = gqa(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        out2 = gqa(x2)
        assert (out2[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"
        assert (out2[:, t] - out[:, t]).abs().max() > 1e-3
    print("causality holds for all t")

    d = GQAttention(E, NH, 1, dropout=0.5)
    d.train()
    assert not torch.equal(d(x), d(x))
    d.eval()
    assert torch.equal(d(x), d(x))

    # 5. the cache: same protocol, smaller tensors. decoding one token at a time
    #    must still reproduce the parallel forward
    for nkv in (NH, 2, 1):
        m = GQAttention(E, NH, nkv)
        full = m(x)
        cache, steps = None, []
        for t in range(T):
            step, cache = m(x[:, t : t + 1], cache, use_cache=True)
            steps.append(step)
        assert cache[0].shape == cache[1].shape == (B, nkv, T, HS)
        assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-6

        #    and the prefill-then-continue path, where is_causal cannot be used
        pre, c = m(x[:, :3], use_cache=True)
        rest, c = m(x[:, 3:], c, use_cache=True)
        assert (torch.cat([pre, rest], dim=1) - full).abs().max() < 1e-6
        assert c[0].shape == (B, nkv, T, HS)
    print("decodes incrementally at every n_kv_head, cache stays [B, n_kv, T, hs]")

    # 6. the cache buffer. Hand the layer a block_size and it preallocates
    #    [B, nkv, block_size, hs] once and writes each step at a cursor; leave
    #    it out and the cache grows by copying, exactly as before. Same tokens
    #    out either way -- the difference is how many allocations it took
    buf = GQAttention(E, NH, 2, block_size=T)
    grow = GQAttention(E, NH, 2)  # no block_size -> the naive mode
    grow.load_state_dict(buf.state_dict())
    full = buf(x)
    cb = cg = None
    ptrs, ob, og = set(), [], []
    #    under no_grad, and not incidentally: k comes out of qkv, so it requires
    #    grad whenever the weights do, and an in-place write into a buffer the
    #    previous step's graph still needs is exactly what append refuses.
    #    generate() is already decorated this way -- the buffer just makes the
    #    requirement load-bearing instead of merely sensible
    with torch.no_grad():
        for t_ in range(T):
            step_b, cb = buf(x[:, t_ : t_ + 1], cb, use_cache=True)
            step_g, cg = grow(x[:, t_ : t_ + 1], cg, use_cache=True)
            ptrs.add(cb[0].data_ptr())  # where this step's k actually lives
            ob.append(step_b)
            og.append(step_g)
    assert (torch.cat(ob, dim=1) - full).abs().max() < 1e-6
    assert (torch.cat(ob, dim=1) - torch.cat(og, dim=1)).abs().max() < 1e-6
    assert len(ptrs) == 1, "the buffer should be allocated once, not per step"
    assert cb.pos == T and cb.k.shape == (B, 2, T, HS)  # full capacity, all used
    assert cg.shape is None and cg.k.shape == (B, 2, T, HS)  # grown to fit
    print(f"decode {T} tokens -- buffer: 1 allocation   grow: {T}")

    # 7. rope, and the one thing GQA could plausibly have broken: q has nh
    #    heads and k has nkv of them, so a positional scheme that mixed heads
    #    together would need special handling here. rope does not mix them. it
    #    acts on (T, head_size) and broadcasts over the head axis, so rotating
    #    the narrow k and then letting the kernel broadcast it is the same as
    #    broadcasting first and rotating the wide copy. that commuting is what
    #    makes "no change needed" a fact rather than a hope
    rp = GQAttention(E, NH, 2, block_size=T, use_rope=True)
    cos, sin = rp.rope_cos, rp.rope_sin
    k_narrow = rp.qkv(x).split([E, 2 * HS, 2 * HS], dim=-1)[1]
    k_narrow = k_narrow.view(B, T, 2, HS).transpose(1, 2)  # [B, nkv, T, hs]
    rot_then_wide = apply_rope(k_narrow, cos, sin).repeat_interleave(rp.n_rep, dim=1)
    wide_then_rot = apply_rope(k_narrow.repeat_interleave(rp.n_rep, dim=1), cos, sin)
    assert (rot_then_wide - wide_then_rot).abs().max() == 0
    print("rope commutes with the kv-head broadcast, exactly")

    #    so the degenerate case still lands on the layer it should: with
    #    n_kv_head == n_head this has to be SDPAttention with rope, the same
    #    way test 1 pinned it without rope
    rp_mha = GQAttention(E, NH, NH, block_size=T, use_rope=True)
    ref = SDPAttention(E, NH, 0.0, T, use_rope=True)
    assert rp_mha.state_dict().keys() == ref.state_dict().keys()
    rp_mha.load_state_dict(ref.state_dict())
    assert (rp_mha(x) - ref(x)).abs().max() < 1e-6
    #    and rope costs no parameters at any group size
    assert sum(p_.numel() for p_ in rp.parameters()) == sum(
        p_.numel() for p_ in GQAttention(E, NH, 2).parameters()
    )
    assert not any("rope" in k_ for k_ in rp.state_dict())

    #    the cache still decodes. note block_size now does two jobs -- it sizes
    #    the rope tables AND selects the preallocated cache -- so use_rope=True
    #    implies the buffer mode. that coupling is harmless (training never uses
    #    a cache, and preallocating is the better mode anyway) but it means the
    #    grow path has to be asked for on purpose: clear block_size after the
    #    tables are built. both modes advance kv_cache.pos, which is where
    #    T_past comes from, so both have to land in the same place
    for grow in (False, True):
        m = GQAttention(E, NH, 2, use_rope=True, block_size=T)
        m.load_state_dict(rp.state_dict())
        if grow:
            m.block_size = None  # tables keep their size; the cache stops preallocating
        full = m(x)
        #    no_grad for the same reason test 6 needs it: the buffer writes in
        #    place, and the rotation is upstream of that write
        with torch.no_grad():
            c, steps = None, []
            for t in range(T):
                step, c = m(x[:, t : t + 1], c, use_cache=True)
                steps.append(step)
            assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-6
            pre, c2 = m(x[:, :3], use_cache=True)
            rest, c2 = m(x[:, 3:], c2, use_cache=True)
            assert (torch.cat([pre, rest], dim=1) - full).abs().max() < 1e-6
    print("rope + gqa decodes incrementally, grown cache and preallocated alike")

    # 8. the payoff, measured on the real config: 8 layers, E=512, 8 heads,
    #    a full 512-token context. cache bytes are counted off the tensors the
    #    layer actually handed back, then multiplied by n_layer
    n_layer, Tb, Eb, nhb = 8, 512, 512, 8
    xb = torch.randn(1, Tb, Eb)
    print(f"\n  KV cache for {n_layer}L / E={Eb} / {nhb} heads, B=1, T={Tb}, bf16")
    print(
        f"  {'n_kv_head':>9}  {'kind':>4}  {'per layer':>10}  {'total':>9}  {'vs mha':>7}"
    )
    base = None
    for nkv in (8, 4, 2, 1):
        m = GQAttention(Eb, nhb, nkv).to(torch.bfloat16)
        _, (ck, cv) = m(xb.to(torch.bfloat16), use_cache=True)
        per_layer = (ck.numel() + cv.numel()) * ck.element_size()
        total = per_layer * n_layer
        base = base or total
        kind = {nhb: "mha", 1: "mqa"}.get(nkv, "gqa")
        print(
            f"  {nkv:>9}  {kind:>4}  {per_layer / 2**20:>7.1f} MB  "
            f"{total / 2**20:>6.1f} MB  {base / total:>6.1f}x"
        )
        assert total == 2 * n_layer * 1 * Tb * nkv * (Eb // nhb) * 2

    # 9. ...and why anyone bothered, at a size this repo will not run. the same
    #    arithmetic on Llama-2-70B's geometry, for ONE sequence
    L, NHL, HSL, CTX = 80, 64, 128, 4096

    def cache_gb(nkv: int) -> float:
        return 2 * L * CTX * nkv * HSL * 2 / 2**30

    print(f"\n  70B geometry ({L}L, {NHL} heads, hs={HSL}), one {CTX}-token sequence")
    print(f"    mha  (n_kv_head={NHL:>2}): {cache_gb(NHL):>5.1f} GB")
    print(
        f"    gqa  (n_kv_head={8:>2}): {cache_gb(8):>5.1f} GB   <- what Llama-2-70B ships"
    )
    print(f"    mqa  (n_kv_head={1:>2}): {cache_gb(1):>5.1f} GB")
    """
    Test 8 is the table, test 9 is the reason. At this repo's scale MHA's cache
    is 8 MB and nobody cares. At 70B it is 10.7 GB for a single sequence -- more
    than the weights of a 7B model, for one user's context -- and it is re-read
    once per generated token. GQA-8 turns that into 1.3 GB, which is the
    difference between batching 4 requests on a card and batching 32. Quality
    costs a fraction of a point. That is the entire trade.
    """

    print("\nok")
