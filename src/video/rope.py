import torch
import torch.nn.functional as F
from kv_cache import KVCache
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from torch import Tensor


def rope_inv_freq(head_size: int, base: float = 10000.0, device=None) -> Tensor:
    """theta_i = base^(-2i/hs), [hs/2] -- the half of RoPE that never moves.

    Position is the other half, and the only one that has to wait for a forward
    pass. Splitting them is what makes the angles buildable on the fly.
    """
    assert head_size % 2 == 0, "head_size must be even: the dims rotate in pairs"
    i = torch.arange(0, head_size, 2, device=device, dtype=torch.float32)
    return base ** (-i / head_size)


def rope_angles(
    offset: int, seq_len: int, head_size: int, base: float = 10000.0, device=None
) -> tuple[Tensor, Tensor]:
    """cos and sin at positions offset .. offset+seq_len-1, both [seq_len, hs/2].

    rope_tables' rows, evaluated where they are needed instead of stored. The
    same numbers -- the table was only ever a cache of this formula -- except
    that `offset` is an int rather than an index into something block_size rows
    tall. There is no position this cannot reach.

    Everything here is float32 and none of it is a buffer, which is one
    decision, not two. Under bf16 the product pos * theta_i loses the angle
    outright at long offsets, and a registered inv_freq would be rounded to 8
    mantissa bits by the first .to(bfloat16) with no way back (test 15). Built
    from an int and a Python float each call, there is nothing for a cast to
    reach. That costs ~7us per layer per decode step against holding the
    tensor; a decode step is milliseconds, and this way it cannot go quietly
    wrong.
    """
    inv_freq = rope_inv_freq(head_size, base, device)
    pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    angles = pos.unsqueeze(1) * inv_freq  # [seq_len, hs/2]
    return angles.cos(), angles.sin()


def rope_tables(block_size: int, head_size: int, base: float = 10000.0):
    """cos and sin of pos * theta_i, both [block_size, head_size / 2].

    Precomputed rows 0 .. block_size-1, which is what a layer wants when it
    knows its length up front. Written in terms of rope_angles to say the thing
    plainly: the table IS the formula at offset 0, with a last row chosen by
    whoever called it.
    """
    return rope_angles(0, block_size, head_size, base)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each adjacent dim pair of x by its position's angle.

    x is [..., T, hs]; cos and sin are [T, hs/2] and broadcast over the rest.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]  # [..., T, hs/2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class RopeAttention(Module):
    """Causal self-attention with rotary positions, and nothing else.

    Deliberately smaller than the other attention files: no dropout, no KV
    cache, no GQA. Those come back when RoPE is merged into
    FusedQKVAttention and GQAttention; here they would only hide the two
    lines that matter.

    Sinusoidal positions are *added* to x at the bottom of the stack, and the
    relative signal has to survive every projection on the way up. RoPE keeps
    the same clock hands -- same theta_i = base^(-2i/hs) -- and moves them to
    where the dot product actually happens: q and k are rotated by their own
    position just before the scores are formed.

    Rotating query m and key n by m*theta and n*theta makes their dot product
    a function of m - n alone, because R(m)^T R(n) = R(n - m). The model never
    sees an absolute position; it sees a distance, exactly, for free.

    The rotation is per head and per dim pair, so it touches no weights and
    adds no parameters. v is NOT rotated -- position belongs in the scores,
    not in what gets collected.
    """

    def __init__(self, n_embed: int, n_head: int, block_size: int):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        cos, sin = rope_tables(block_size, self.head_size)
        # rebuilt by __init__ from the formula, so nothing to checkpoint
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, E = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(E, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, E)
        return self.proj(out)


class RopeKVAttention(Module):
    """RopeAttention plus a KV cache, and the one thing that pairing needs.

    RopeAttention rotates by 0 .. T-1 because it always sees the whole
    sequence. With a cache it does not: x holds only the new tokens, and those
    sit at T_past .. T_past+T-1, where T_past is what the cache already holds.
    Rotate them at 0 instead and generation still runs and still returns
    plausible logits -- every token just quietly believes it is at the start of
    the sequence, and test 13 is what catches it.

    That is the whole delta. Two pieces of the bookkeeping are worth naming:

      - k is rotated BEFORE it is appended, so the cache holds rotated keys and
        nothing is ever rotated twice. v is not rotated at all -- position
        belongs in the scores, not in what gets collected.
      - the constructor takes no block_size. There is no table to size: the
        angles come from rope_angles(T_past, T, inv_freq), and an offset is an
        int. This layer decodes at position 100,000 as readily as at 10.

    Which is what makes it worth having as its own file entry rather than a
    flag on SDPAttention: it shows where the ceiling actually is. Not
    positions -- those were always a formula, and a formula has no last row.
    Memory: the cache takes one k and one v per token, forever, and test 14
    watches it grow. Bounding that is eviction, which needs a window and a ring
    buffer. RoPE is what makes eviction legitimate, not what performs it --
    scores depend on m - n, so dropping old keys does not change what the
    survivors mean to each other (test 5, again).

    Still deliberately small: no dropout, no GQA, no preallocation. Grow mode
    is the honest cache here, because a preallocated one needs a capacity, and
    a capacity is precisely what this class exists not to have.
    """

    def __init__(self, n_embed: int, n_head: int, base: float = 10000.0):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        self.base = base  # a float, not a tensor: nothing here to move or cast

    def angles(self, offset: int, seq_len: int, device, dtype) -> tuple[Tensor, Tensor]:
        """Where this layer thinks its tokens are. One line, and a seam.

        Everything positional the layer does goes through here, so overriding
        it is how a variant changes position without touching attention: test
        15 does it to hold theta_i in bf16, and re-basing a window into
        [0, W) -- rung 3 -- is the same override with a different offset.
        """
        cos, sin = rope_angles(offset, seq_len, self.head_size, self.base, device)
        return cos.to(dtype), sin.to(dtype)  # cast the result, not the math

    def forward(
        self,
        x: Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        B, T, E = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(E, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # read BEFORE the append, which advances it
        T_past = 0 if kv_cache is None else kv_cache.pos
        cos, sin = self.angles(T_past, T, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if use_cache and kv_cache is None:
            kv_cache = KVCache()  # grow mode: no block_size, no capacity to outgrow
        if kv_cache is not None:
            k, v = kv_cache.append(k, v)  # already-rotated keys go in

        T_kv = k.size(2)
        if T_kv == T:
            is_causal, attn_mask = True, None  # nothing cached: the plain tril
        elif T == 1:
            is_causal, attn_mask = False, None  # one query, all of it behind
        else:
            # queries sit at T_past .. T_kv-1, and is_causal would align them
            # top-left instead. two aranges say where they really are
            is_causal = False
            q_pos = torch.arange(T_past, T_kv, device=x.device).unsqueeze(1)
            attn_mask = torch.arange(T_kv, device=x.device) <= q_pos

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
        out = out.transpose(1, 2).reshape(B, T, E)
        out = self.proj(out)
        return (out, kv_cache) if use_cache else out


if __name__ == "__main__":
    import math

    from fused_qkv_attention import FusedQKVAttention
    from sinusoidal import SinusoidalEmbedding
    from torch import nn

    torch.manual_seed(0)
    B, T, E, NH = 2, 16, 32, 4
    HS = E // NH
    x = torch.randn(B, T, E)
    rope = RopeAttention(E, NH, T)

    # 1. a drop-in for the attention it replaces: same shape, and exactly the
    #    same parameters. the rotation is a function of position, not a weight,
    #    so the whole positional scheme costs zero of them
    fused = FusedQKVAttention(E, NH, T)
    assert rope(x).shape == (B, T, E)
    assert [n for n, _ in rope.named_parameters()] == [
        n for n, _ in fused.named_parameters()
    ]
    assert sum(p.numel() for p in rope.parameters()) == 3 * E * E + E * E + E
    #    and the tables are buffers rebuilt from the formula -- no checkpoint keys
    assert rope.state_dict().keys() == {"qkv.weight", "proj.weight", "proj.bias"}
    assert [n for n, _ in rope.named_buffers()] == ["rope_cos", "rope_sin"]

    # 2. what the rotation is: a rotation. it preserves every head's norm, so
    #    it cannot change how "loud" a query or key is -- only which direction
    #    it points. and position 0 rotates by 0*theta = 0, i.e. not at all
    xh = x.view(B, T, NH, HS).transpose(1, 2)  # [B, nh, T, hs], what forward rotates
    r = apply_rope(xh, rope.rope_cos, rope.rope_sin)
    assert (r.norm(dim=-1) - xh.norm(dim=-1)).abs().max() < 1e-5
    assert torch.equal(r[:, :, 0], xh[:, :, 0])

    # 3. THE property, the reason the file exists. put the SAME vector at every
    #    position, so the only thing that can vary between scores is position.
    #    the score matrix then comes out constant along each diagonal: what a
    #    query at m and a key at n produce depends on m - n and nothing else.
    #    compare with sinusoidal.py test 4, which showed the same fact about
    #    PE[p] . PE[q]; the difference is where it lives, not what it says
    qv, kv = torch.randn(HS), torch.randn(HS)
    q1 = qv.expand(1, 1, T, HS)
    k1 = kv.expand(1, 1, T, HS)
    cos, sin = rope.rope_cos, rope.rope_sin
    s = (apply_rope(q1, cos, sin) @ apply_rope(k1, cos, sin).transpose(-2, -1))[0, 0]
    for d in range(-T + 1, T):
        diag = torch.diagonal(s, offset=d)
        assert (diag - diag[0]).abs().max() < 1e-4, f"not constant at offset {d}"
    print("scores depend on m - n alone")

    # 4. and the distance is SIGNED, not just a magnitude. the mirror pair
    #    (m, n) and (n, m) sit the same number of positions apart, and RoPE
    #    still tells them apart: R(n - m) is the transpose of R(m - n), so the
    #    cos half of the score is shared and the sin half flips. a key BEFORE
    #    the query and a key AFTER it are different events, which is the whole
    #    reason this encodes order rather than proximity.

    #    in short: "a, then b three positions later" does not score the same as
    #    "b, then a three positions later"
    assert (s - s.T).abs().max() > 1e-3
    print(f"q != k   d=+2: {s[8, 10]:.4f}   d=-2: {s[10, 8]:.4f}")
    #    the exception is the degenerate case where q and k are the same
    #    vector. writing one pair out, the score is
    #
    #        q1^2 cos(r*th) + q2^2 cos(r*th) + q2*q1 sin(r*th) - q1*q2 sin(r*th)
    #
    #    and the last two cancel, leaving |q_pair|^2 * cos(r*th) -- even in r,
    #    so the matrix goes exactly symmetric. (the notebook's q = k = ones(2)
    #    is this case: |q_pair|^2 = 2, hence its 2*cos.) worth pinning down,
    #    because a symmetric score matrix in the wild means q and k have
    #    collapsed onto each other, not that RoPE is direction-blind

    same = (apply_rope(q1, cos, sin) @ apply_rope(q1, cos, sin).transpose(-2, -1))[0, 0]
    assert (same - same.T).abs().max() == 0
    print(f"q == k   d=+2: {same[8, 10]:.4f}   d=-2: {same[10, 8]:.4f}")
    #    swapping which vector plays query mirrors the matrix instead
    swapped = (apply_rope(k1, cos, sin) @ apply_rope(q1, cos, sin).transpose(-2, -1))[
        0, 0
    ]
    assert (swapped - s.T).abs().max() < 1e-5

    # 5. the same statement said the way that matters for a KV cache: slide the
    #    whole window forward and the scores do not move. this is what makes it
    #    legitimate to drop old cache entries later -- position enters as a
    #    difference, so re-basing the window is not a change of meaning
    q2, k2 = torch.randn(1, 1, T, HS), torch.randn(1, 1, T, HS)
    half = T // 2
    early = apply_rope(q2[:, :, :half], cos[:half], sin[:half]) @ apply_rope(
        k2[:, :, :half], cos[:half], sin[:half]
    ).transpose(-2, -1)
    shifted = apply_rope(q2[:, :, :half], cos[half:], sin[half:]) @ apply_rope(
        k2[:, :, :half], cos[half:], sin[half:]
    ).transpose(-2, -1)
    assert (early - shifted).abs().max() < 1e-4
    print(f"same scores at positions 0-{half - 1} and {half}-{T - 1}")

    # 6. against the paper's own formulation. RoPE was written with complex
    #    numbers: read each dim pair as one complex number and multiply by
    #    e^(i * pos * theta). that IS the 2x2 rotation, and it is an
    #    independent implementation, so it catches a swapped sin or a
    #    transposed rotation that the property tests above would not
    xc = torch.view_as_complex(xh.reshape(B, NH, T, HS // 2, 2))
    inv_freq = 10000.0 ** (-torch.arange(0, HS, 2) / HS)
    freqs = torch.polar(torch.ones(T, HS // 2), torch.arange(T).unsqueeze(1) * inv_freq)
    ref = torch.view_as_real(xc * freqs.view(1, 1, T, HS // 2)).flatten(-2)
    assert (r - ref).abs().max() < 1e-5

    # 7. and against the closed form for a single pair, by hand: the paper's
    #    theta_i, the paper's 2x2 matrix, one position, one pair
    i, pos = 3, 11
    theta = 10000.0 ** (-2 * i / HS)
    c, sn = math.cos(pos * theta), math.sin(pos * theta)
    a, b = xh[0, 0, pos, 2 * i : 2 * i + 2]
    assert abs(r[0, 0, pos, 2 * i].item() - (a * c - b * sn).item()) < 1e-5
    assert abs(r[0, 0, pos, 2 * i + 1].item() - (a * sn + b * c).item()) < 1e-5

    # 8. the control that isolates what the rotation is doing: the same layer,
    #    the same weights, tables set to identity (cos=1, sin=0) so nothing
    #    rotates. outputs have to differ, or none of the above matters.
    #    position 0 is the exception, and for a reason worth saying out loud:
    #    it rotates by 0*theta either way, and causally it can only see itself,
    #    so both layers agree there exactly
    flat = RopeAttention(E, NH, T)
    flat.load_state_dict(rope.state_dict())
    flat.rope_cos, flat.rope_sin = torch.ones_like(cos), torch.zeros_like(sin)
    assert (rope(x) - flat(x)).abs().max() > 1e-3
    assert (rope(x)[:, 0] - flat(x)[:, 0]).abs().max() < 1e-6

    # 9. causality: perturbing token t cannot change any output before t
    out = rope(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        out2 = rope(x2)
        assert (out2[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"
        assert (out2[:, t] - out[:, t]).abs().max() > 1e-3
    print("causality holds for all t")

    # 10. sinusoidal and rope agree about *what* position is and disagree about
    #    where to put it. both are built from theta_i = base^(-2i/d); one adds a
    #    vector to the residual stream, the other rotates q and k. so the
    #    sinusoidal table's own pairs are rotated by exactly the same angles
    #    take the unit vector (1, 0) in every pair, rotate it by position, and
    #    the sinusoidal table falls out -- read as (cos, sin) where
    #    SinusoidalEmbedding writes (sin, cos), which is the same two numbers
    #    in the other order
    enc = SinusoidalEmbedding(T, HS)
    e0 = torch.zeros(1, 1, T, HS)
    e0[..., 0::2] = 1.0
    traced = apply_rope(e0, cos, sin)[0, 0]  # [T, hs]
    assert (traced[:, 0::2] - enc.pe[:, 1::2]).abs().max() < 1e-4  # cos half
    assert (traced[:, 1::2] - enc.pe[:, 0::2]).abs().max() < 1e-4  # sin half
    print("rotating the unit vector by position traces the sinusoidal table")

    # 11. against a torch mirror: same forward, same grads. there is no
    #    nn.RoPE, so the reference rotates with the complex formulation and
    #    leans on nn.Linear and F.scaled_dot_product_attention for the rest
    class RefRope(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(E, 3 * E, bias=False)
            self.proj = nn.Linear(E, E)
            self.register_buffer("freqs", freqs)

        def rot(self, t):  # [B, nh, T, hs]
            tc = torch.view_as_complex(t.reshape(*t.shape[:-1], HS // 2, 2))
            return torch.view_as_real(tc * self.freqs.view(1, 1, T, HS // 2)).flatten(
                -2
            )

        def forward(self, x):
            q, k, v = self.qkv(x).split(E, dim=-1)
            q, k, v = (t.view(B, T, NH, HS).transpose(1, 2) for t in (q, k, v))
            o = F.scaled_dot_product_attention(
                self.rot(q), self.rot(k), v, is_causal=True
            )
            return self.proj(o.transpose(1, 2).reshape(B, T, E))

    ref_mod = RefRope()
    assert rope.state_dict().keys() == {k for k in ref_mod.state_dict() if k != "freqs"}
    rope.load_state_dict(
        {k: v for k, v in ref_mod.state_dict().items() if k != "freqs"}
    )

    xm = x.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    om, orf = rope(xm), ref_mod(xr)
    assert (om - orf).abs().max() < 1e-6
    om.square().sum().backward()
    orf.square().sum().backward()
    gd = max(
        (a.grad - b.grad).abs().max().item()
        for a, b in zip(rope.parameters(), ref_mod.parameters())
    )
    assert gd < 1e-5 and (xm.grad - xr.grad).abs().max() < 1e-5

    # ------------------------------------------ angles on the fly + a cache

    # 12. the swap that licenses the rest: rope_angles(off, n) is the table's
    #     rows off .. off+n-1, exactly, at every offset a cache can ask for --
    #     the start, the middle, the last row, a whole prefill. if this were
    #     only approximately true, swapping one for the other would change the
    #     model. and then the point of the formula: a table has a last row,
    #     chosen by whoever sized it, and a formula does not. position 100,000
    #     costs the same as position 3 and needed nobody to predict it
    for off, n in ((0, T), (5, 3), (T - 1, 1), (1, T - 1)):
        c_fly, s_fly = rope_angles(off, n, HS)
        assert (c_fly - cos[off : off + n]).abs().max() < 1e-6, (off, n)
        assert (s_fly - sin[off : off + n]).abs().max() < 1e-6, (off, n)
    assert rope_angles(100_000, 1, HS)[0].shape == (1, HS // 2)
    #     the table's answer to that same question is not a wrong number, it is
    #     an empty tensor -- which is why running past it surfaces as a
    #     broadcast error in apply_rope rather than as quietly bad output
    assert cos[100_000:100_001].shape == (0, HS // 2)
    print("angles on the fly: same numbers as the table, and no last row")

    # 13. THE test, the one every cached layer has to pass: prefill part of the
    #     sequence, then decode the rest one token at a time, and land on what
    #     the parallel forward already produced. it fails if the layer rotates
    #     the concatenated k instead of only the new rows, or rotates every
    #     step as if it were at position 0. the reference is RopeAttention with
    #     the same weights, so this also pins the drop-in claim: the cache is an
    #     addition, not a different layer
    kvr = RopeKVAttention(E, NH)
    assert kvr.state_dict().keys() == rope.state_dict().keys()
    kvr.load_state_dict(rope.state_dict())
    full = rope(x)
    assert (kvr(x) - full).abs().max() < 1e-6

    head, c = kvr(x[:, :5], use_cache=True)
    outs = [head]
    for t in range(5, T):
        step, c = kvr(x[:, t : t + 1], c, use_cache=True)
        outs.append(step)
    assert (torch.cat(outs, dim=1) - full).abs().max() < 1e-5
    print("prefill + decode == full forward")

    # 14. and the headline: it decodes past any block_size, because it never
    #     had one. RopeAttention has to be told how long the run will be to
    #     serve as the reference at all -- being told is exactly the constraint
    #     generate.py asserts -- and the layer under test is told nothing and
    #     agrees anyway. so what is left is not positions, which were only ever
    #     a formula. it is bytes: the cache kept one k and one v per token and
    #     grows linearly forever. bounding that is rung 2, and rope is what
    #     makes bounding it legitimate rather than what does it -- scores
    #     depend on m - n, so the keys that survive eviction still mean to each
    #     other exactly what they meant before (test 5)
    LONG = 3 * T
    xl = torch.randn(B, LONG, E)
    ref_long = RopeAttention(E, NH, LONG)
    ref_long.load_state_dict(rope.state_dict())
    outs, c, widths = [], None, []
    for t in range(LONG):
        step, c = kvr(xl[:, t : t + 1], c, use_cache=True)
        outs.append(step)
        widths.append(c[0].size(2))
    assert (torch.cat(outs, dim=1) - ref_long(xl)).abs().max() < 1e-5
    assert widths == list(range(1, LONG + 1)) and c.pos == LONG
    print(f"{LONG} tokens with no block_size, and a cache {LONG} slots wide")

    # 15. why none of this is a buffer. bf16 keeps 8 mantissa bits, so a
    #     theta_i rounded to it carries ~0.2% relative error and the angle
    #     pos * theta_i multiplies that by pos -- cos then walks off by
    #     radians, not by epsilon. Module.to casts every float buffer
    #     (module.py:91-95), so an inv_freq registered the way the tables are
    #     would be wrecked by the first .to(bfloat16), and .float() afterwards
    #     recovers nothing. built from an int and a Python float each call,
    #     there is nothing for a cast to reach
    cast = rope_inv_freq(HS).bfloat16().float()  # what such a buffer would hold
    for far in (512, 5000):
        err = (rope_angles(far, 1, HS)[0] - (far * cast).cos()).abs().max()
        print(f"  pos {far:>5}: a bf16-cast inv_freq costs {err:.4f} in cos")
    assert (rope_angles(5000, 1, HS)[0] - (5000 * cast).cos()).abs().max() > 0.1

    #     and the same mistake inside a layer, which is where it would actually
    #     bite. angles() is the only seam it needs, so the variant is four
    #     lines and everything else -- weights, cache, attention -- is held
    #     identical between the two
    class Bf16FreqRope(RopeKVAttention):
        """theta_i at bf16 precision: the buffer that is deliberately not there."""

        def angles(self, offset, seq_len, device, dtype):
            inv = rope_inv_freq(self.head_size, self.base, device).bfloat16().float()
            pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)
            a = pos.unsqueeze(1) * inv
            return a.cos().to(dtype), a.sin().to(dtype)

    worse = Bf16FreqRope(E, NH)
    worse.load_state_dict(rope.state_dict())

    def step_at(layer: RopeKVAttention, pos: int) -> Tensor:
        """One decode step with the query at `pos`, in whatever dtype the layer
        holds. The cache is advanced rather than filled: 5000 real steps prove
        the same thing and take a minute, and every layer is handed the
        identical state either way."""
        xin = x.to(layer.qkv.weight.dtype)
        _, c = layer(xin[:, :3], use_cache=True)
        c.pos = pos
        return layer(xin[:, 3:4], c, use_cache=True)[0].float()

    drift = {}
    for pos in (0, 512, 5000, 50_000):
        ok = step_at(kvr, pos)
        drift[pos] = ((ok - step_at(worse, pos)).abs().max() / ok.abs().max()).item()
    print("  the same layer with theta_i in bf16, output drift by position:")
    print("   " + "   ".join(f"{pos}: {d:.2%}" for pos, d in drift.items()))
    #     which is the shape of the whole problem. the error is not in the
    #     rotation, it is in the arm being long -- nothing whatsoever where a
    #     block_size-bounded model ever ran, and a twentieth of the step's own
    #     output by the time a ring buffer has let the run get interesting.
    #     a bug that only appears past the horizon you just removed
    assert drift[0] < 1e-4  # nothing to see wherever the tables were tested
    assert drift[5000] > 0.02  # same layer, same weights, 5000 positions in
    assert drift[0] < drift[512] < drift[5000] < drift[50_000]

    #     done right, the same layer runs entirely in bf16 and stays put:
    #     ~0.6% of its own output at every distance, which is bf16's noise on
    #     weights and activations and has nothing to do with position. flat is
    #     the whole claim -- the rotation stays exact however far out it goes,
    #     and only the final cast is lossy
    kv16 = RopeKVAttention(E, NH)
    kv16.load_state_dict(rope.state_dict())
    kv16.to(torch.bfloat16)
    assert not list(kv16.buffers())  # there was nothing for .to() to reach
    flat = {}
    for pos in drift:
        ok = step_at(kvr, pos)
        flat[pos] = ((ok - step_at(kv16, pos)).abs().max() / ok.abs().max()).item()
    print("  and with the angles built right, the same layer all in bf16:")
    print("   " + "   ".join(f"{pos}: {d:.2%}" for pos, d in flat.items()))
    assert max(flat.values()) < 0.02  # small
    assert flat[50_000] < 2 * flat[0]  # ...and, unlike drift, not growing
    assert flat[50_000] < drift[50_000] / 10  # where the other was 13% off
    #     worth reading the two rows against each other at position 0, where
    #     the rounded theta is the BETTER of the two -- it does its arithmetic
    #     in float32 while this one is bf16 throughout. that is what a dtype
    #     bug looks like from inside a short test: fine, fine, then 13% off

    print("ok")
