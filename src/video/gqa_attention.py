import torch
import torch.nn.functional as F
from torch import Tensor

from dropout import Dropout
from kv_cache import KVCache
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from ring_kv_cache import RingKVCache
from rope import apply_rope, rope_tables
from sliding_window import sliding_window_mask, window_mask_from_positions


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

    window=W masks each query to the W keys ending at it (sliding_window.py),
    and costs nothing here for the same reason rope does: a mask is a statement
    about columns, grouping is a statement about heads, and the two never meet.
    They cut different axes of the same cache -- GQA divides it by n_rep, a
    window would bound it by W -- except that a window alone does not shrink
    the cache at all. It only stops the model looking at what is still there.

    ring=True is what closes that gap (ring_kv_cache.py). The cache becomes a
    RingKVCache whose capacity IS the window, so the entry for position p lives
    in slot p % W and memory per layer stops growing with the run. Two things
    follow. The cached keys are no longer in position order, so the mask comes
    from the cache's `positions` vector rather than from arange. And a single
    query needs no mask AT ALL -- every slot the ring still holds is inside its
    window by construction -- so rung 2 takes back out of the decode path the
    mask rung 1 put into it.
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int | None = None,
        dropout: float = 0.0,
        block_size: int | None = None,
        use_rope: bool = False,
        window: int | None = None,
        ring: bool = False,
    ):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        assert window is None or window >= 1, "a window has to include the query itself"
        assert not ring or window is not None, (
            "a ring buffer's capacity IS the window, so ring=True needs one"
        )
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
        self.window = window
        self.ring = ring
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

        # captured BEFORE the append, which advances it. a ring's pos keeps
        # counting tokens past its capacity, so this stays the absolute position
        # of the first new token however many times the buffer has been round
        T_past = 0 if kv_cache is None else kv_cache.pos
        if self.use_rope:
            # q is [B, nh, T, hs] and k is [B, nkv, T, hs], and the rotation
            # does not care: it acts on (T, hs) and broadcasts over whatever
            # head axis it is handed. grouping is a fact about heads, rotation
            # is a fact about positions, and the two never meet -- the same
            # reason the causal mask needed no change for GQA.
            # before the append, so the cache holds keys already rotated
            cos = self.rope_cos[T_past : T_past + T]
            sin = self.rope_sin[T_past : T_past + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if use_cache and kv_cache is None:
            # the layer allocates it because only the layer knows n_kv_head and
            # head_size; B, device and dtype come from the data on first write.
            # capacity is block_size, and without one this falls back to the
            # tuple's grow-by-copying
            if self.ring:
                # capacity is the WINDOW, not block_size: the cache stops
                # sizing itself to the run and starts sizing itself to what the
                # mask can still reach
                kv_cache = RingKVCache((B, nkv, self.window, hs))
            else:
                shape = (
                    None if self.block_size is None else (B, nkv, self.block_size, hs)
                )
                kv_cache = KVCache(shape)
        if kv_cache is not None:
            # the NARROW k and v go in -- [B, nkv, T_kv, hs], n_rep smaller
            k, v = kv_cache.append(k, v)

        T_kv = k.size(2)
        ring = isinstance(kv_cache, RingKVCache)
        # a window nothing has fallen out of yet is not a window
        unwindowed = self.window is None or T_kv <= self.window
        if unwindowed and kv_cache is None:
            is_causal, attn_mask = True, None
        elif (unwindowed or ring) and T == 1:
            # one query, and everything it may attend to is in front of it: a
            # full cache within the window, or a ring, which holds nothing else
            is_causal, attn_mask = False, None
        elif ring:
            # the cached keys are in SLOT order, which stopped being position
            # order the first time the cursor went round. only the cache knows
            # what is where, so the mask is built from its positions vector.
            # nothing needs un-permuting: softmax is permutation-equivariant
            # over the key axis, and rope baked each key's position into it
            is_causal = False
            attn_mask = window_mask_from_positions(
                torch.arange(T_past, T_past + T, device=x.device),
                kv_cache.key_positions,
                self.window,
            )
        else:
            # queries sit at T_past .. T_kv-1, and is_causal aligns top-left --
            # the same shift SDPAttention needed, plus the band when a window
            # is set. neither is a statement about heads, so grouping changes
            # nothing here: the mask broadcasts over the head axis either way
            is_causal = False
            attn_mask = sliding_window_mask(T, T_kv, self.window, x.device)

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

    # 8. the sliding window, which grouping does not touch for exactly the
    #    reason rope did not: the mask indexes COLUMNS and grouping indexes
    #    HEADS. The strongest way to say that is the degenerate case -- with
    #    n_kv_head == n_head this has to be SDPAttention with the same window,
    #    to the bit, the same claim test 1 made before either flag existed
    W = 3
    win = GQAttention(E, NH, NH, window=W)
    ref_win = SDPAttention(E, NH, window=W)
    win.load_state_dict(ref_win.state_dict())
    assert (win(x) - ref_win(x)).abs().max() == 0

    #    and at a real group size, against the same by-hand grouping test 3
    #    used -- q head h against kv head h // n_rep -- now with the band
    #    applied to the scores. no SDPA and no enable_gqa anywhere in here
    gw = GQAttention(E, NH, 2, window=W)
    q, k, v = gw.qkv(x).split([E, 2 * HS, 2 * HS], dim=-1)
    q = q.view(B, T, NH, HS).transpose(1, 2)
    k = k.view(B, T, 2, HS).transpose(1, 2)
    v = v.view(B, T, 2, HS).transpose(1, 2)
    band = sliding_window_mask(T, T, W)
    heads = []
    for h in range(NH):
        g = h // gw.n_rep
        sc = q[:, h] @ k[:, g].transpose(-2, -1) * HS**-0.5
        heads.append(sc.masked_fill(~band, -torch.inf).softmax(dim=-1) @ v[:, g])
    assert (gw.proj(torch.cat(heads, dim=-1)) - gw(x)).abs().max() < 1e-6
    #    ...and it is a different function from the unwindowed layer, so the
    #    agreement above is not two no-ops matching
    plain_gw = GQAttention(E, NH, 2)
    plain_gw.load_state_dict(gw.state_dict())
    assert (gw(x) - plain_gw(x)).abs().max() > 1e-2
    print(f"window W={W} is the same band under grouping, hand-computed per head")

    # 9. the window through the preallocated cache, which is the combination
    #    this layer is the only one to have: buffer mode writes at a cursor and
    #    the window skips its mask while T_kv <= W, so a decode crosses a branch
    #    boundary while writing in place. Both rope settings, since rope reads
    #    T_past off the same cursor the buffer advances
    for use_rope in (False, True):
        m = GQAttention(E, NH, 2, block_size=T, use_rope=use_rope, window=W)
        full = m(x)
        with torch.no_grad():
            c, steps = None, []
            for t in range(T):
                step, c = m(x[:, t : t + 1], c, use_cache=True)
                steps.append(step)
            assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-6
            for split in (1, W, W + 1, T - 1):  # either side of it, and on it
                pre, c2 = m(x[:, :split], use_cache=True)
                rest, c2 = m(x[:, split:], c2, use_cache=True)
                assert (torch.cat([pre, rest], dim=1) - full).abs().max() < 1e-6
        #    and the cache is exactly as large as it was without a window. this
        #    rung buys a receptive field, not a byte -- W keys are attended and
        #    T_kv keys are stored. closing that gap is the ring buffer's job
        assert c2.pos == T and c2[0].shape == (B, 2, T, HS)
    print("windowed decode holds through the buffer, and the cache is no smaller")

    # 10. the ring buffer, which is that last sentence stopped being true. The
    #     cache becomes capacity-W storage that wraps, so the keys the window
    #     can no longer reach are no longer kept. THE claim is that this costs
    #     nothing: eviction is not an approximation of the windowed model, it
    #     IS the windowed model, so a ring decode has to reproduce the parallel
    #     windowed forward exactly -- the same bar test 9's full cache cleared
    for use_rope in (False, True):
        m = GQAttention(E, NH, 2, block_size=T, use_rope=use_rope, window=W, ring=True)
        ref = GQAttention(E, NH, 2, block_size=T, use_rope=use_rope, window=W)
        ref.load_state_dict(m.state_dict())
        full = m(x)
        #     ring changes the CACHE and nothing else, so with no cache at all
        #     the two layers are the same function to the bit
        assert (full - ref(x)).abs().max() == 0
        with torch.no_grad():
            c, steps = None, []
            for t in range(T):
                step, c = m(x[:, t : t + 1], c, use_cache=True)
                steps.append(step)
            assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-6
            #     and the payoff, which test 9 could not assert: the cache
            #     never grew past the window. it holds the last W positions,
            #     in slot order, and pos keeps counting tokens regardless
            assert c.k.shape == (B, 2, W, HS)
            assert c.pos == T and c.fill == W
            assert sorted(c.key_positions.tolist()) == list(range(T - W, T))
            #     prefill-then-continue still works, including when the prefill
            #     is longer than the ring: the chunk gets back every key it is
            #     entitled to, and only its tail is retained
            for split in (1, W, W + 1, T - 1):
                pre, c2 = m(x[:, :split], use_cache=True)
                rest, c2 = m(x[:, split:], c2, use_cache=True)
                assert (torch.cat([pre, rest], dim=1) - full).abs().max() < 1e-6, split
                assert c2.k.shape == (B, 2, W, HS)
    print(f"ring decode == windowed forward, and the cache stays {W} wide")

    #     ...and a ring with no window has no capacity to be, so it is refused
    #     at construction rather than defaulting to something arbitrary
    try:
        GQAttention(E, NH, 2, ring=True)
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "capacity IS the window" in str(e)

    # 11. the same thing over a run long enough for the cursor to go round
    #     eight times, because "it wraps correctly" is the one claim a
    #     T-token test cannot make. 64 tokens through a ring of 8, against a
    #     cache that keeps all 64 and masks. Same logits, flat memory
    Wl, Nl = 8, 64
    xl = torch.randn(B, Nl, E)
    ring_m = GQAttention(E, NH, 2, window=Wl, ring=True)
    keep_m = GQAttention(E, NH, 2, window=Wl)  # no block_size -> grow mode
    keep_m.load_state_dict(ring_m.state_dict())
    with torch.no_grad():
        cr = ck = None
        or_, ok_ = [], []
        for t in range(Nl):
            sr, cr = ring_m(xl[:, t : t + 1], cr, use_cache=True)
            sk, ck = keep_m(xl[:, t : t + 1], ck, use_cache=True)
            or_.append(sr)
            ok_.append(sk)
            assert cr.k.size(2) == Wl  # flat, every step
    assert (torch.cat(or_, dim=1) - torch.cat(ok_, dim=1)).abs().max() < 1e-6
    assert ck.k.size(2) == Nl and cr.pos == ck.pos == Nl
    print(
        f"{Nl} tokens, cursor round {Nl // Wl}x -- ring cache {Wl} wide, "
        f"kept cache {Nl}, identical logits"
    )

    # 12. the payoff, measured on the real config: 8 layers, E=512, 8 heads,
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

    # 13. ...and why anyone bothered, at a size this repo will not run. the same
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
    Test 12 is the table, test 13 is the reason. At this repo's scale MHA's cache
    is 8 MB and nobody cares. At 70B it is 10.7 GB for a single sequence -- more
    than the weights of a 7B model, for one user's context -- and it is re-read
    once per generated token. GQA-8 turns that into 1.3 GB, which is the
    difference between batching 4 requests on a card and batching 32. Quality
    costs a fraction of a point. That is the entire trade.
    """

    print("\nok")
