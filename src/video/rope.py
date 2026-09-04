import torch
import torch.nn.functional as F
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from torch import Tensor


def rope_tables(block_size: int, head_size: int, base: float = 10000.0):
    """cos and sin of pos * theta_i, both [block_size, head_size / 2]."""
    assert head_size % 2 == 0, "head_size must be even: the dims rotate in pairs"
    inv_freq = base ** (-torch.arange(0, head_size, 2) / head_size)  # [hs/2]
    angles = torch.arange(block_size).unsqueeze(1) * inv_freq  # [T, hs/2]
    return angles.cos(), angles.sin()


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

    print("ok")
