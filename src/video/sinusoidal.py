import math

import torch
from module import Module
from torch import Tensor


class SinusoidalEmbedding(Module):
    """Positions as frequencies, from "Attention Is All You Need".

    A drop-in for Embedding(block_size, n_embed) on the position side: same call,
    same output shape, zero parameters. Column pair (2i, 2i+1) is one clock hand
    turning at its own rate, and position pos is written as where every hand
    points:

        PE[pos, 2i]   = sin(pos / 10000^(2i/E))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/E))

    i = 0 turns once per ~6 positions, the last pair once per ~63000. Together
    they give every position a code no other position shares, the way hour and
    minute hands separate every minute of the day.

    Two properties come out of that, and both are tested below:

    1. PE[p] . PE[q] depends only on p - q. Not approximately -- the cross terms
       collapse to sum_i cos(theta_i (p - q)) exactly. A table of learned vectors
       has no such structure; this one is *about* distance.
    2. PE[pos + k] is a fixed linear map applied to PE[pos] -- a rotation by
       theta_i * k in each pair, the same rotation whatever pos is.

    Point 2 is the whole reason this file exists before RoPE. The relative
    information is already here, but it arrives *added to* the token embedding at
    the bottom of the stack and has to survive every projection on the way up.
    RoPE keeps the same rotation and moves it onto q and k inside attention,
    where the dot product actually happens.
    """

    def __init__(self, block_size: int, n_embed: int):
        assert n_embed % 2 == 0, "n_embed must be even: the columns come in pairs"
        self.block_size = block_size
        self.n_embed = n_embed
        pos = torch.arange(block_size).unsqueeze(1)  # [T, 1]
        i = torch.arange(0, n_embed, 2)  # [E/2] -- 0, 2, 4, ...
        theta = pos / 10000 ** (i / n_embed)  # [T, E/2] angle per position
        pe = torch.empty(block_size, n_embed)
        pe[:, 0::2] = theta.sin()
        pe[:, 1::2] = theta.cos()
        # a buffer, not a Parameter: nothing here is learned. persistent=False
        # keeps it out of state_dict, since __init__ rebuilds it exactly
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, idx: Tensor) -> Tensor:
        return self.pe[idx]  # [...] of positions -> [..., n_embed]


if __name__ == "__main__":
    from embedding import Embedding

    T, E = 64, 32
    enc = SinusoidalEmbedding(T, E)

    # 1. the oracle is the paper's formula, written as the loop it describes.
    #    math.* works in double and the buffer is fp32, so the gap is rounding,
    #    ~1e-6 by pos 58 where theta has grown large
    ref = torch.empty(T, E, dtype=torch.float64)
    for pos in range(T):
        for i in range(0, E, 2):
            ref[pos, i] = math.sin(pos / 10000 ** (i / E))
            ref[pos, i + 1] = math.cos(pos / 10000 ** (i / E))
    assert (enc.pe - ref).abs().max() < 1e-5

    # 2. a drop-in for the learned table: same call, same shape, no parameters
    learned = Embedding(T, E)
    pos = torch.arange(T)
    assert enc(pos).shape == learned(pos).shape == (T, E)
    assert list(enc.named_parameters()) == []
    assert sum(p.numel() for p in learned.parameters()) == T * E
    # and nothing to save -- the formula is the checkpoint
    assert enc.state_dict() == {}
    print(f"learned: {T * E} params   sinusoidal: 0")

    # 3. every position gets its own code, and every value is a sin or a cos
    assert enc.pe.abs().max() <= 1.0
    assert len({tuple(row.tolist()) for row in enc.pe}) == T
    # position 0 is all sin(0), cos(0) -> the alternating 0, 1 pattern
    assert torch.equal(enc(torch.tensor(0))[0::2], torch.zeros(E // 2))
    assert torch.equal(enc(torch.tensor(0))[1::2], torch.ones(E // 2))

    # 4. THE property: the dot product between two positions depends only on the
    #    distance between them, exactly. sin*sin + cos*cos collapses the pair to
    #    cos(theta*(p-q)), and pos itself drops out of every term
    g = enc.pe @ enc.pe.T  # [T, T]
    for d in range(T):
        diag = torch.diagonal(g, offset=d)  # every pair exactly d apart
        assert (diag - diag[0]).abs().max() < 1e-4, f"not constant at distance {d}"
    print("dot product is a function of p - q alone")

    #    it peaks at distance 0 and falls away -- but only for a few positions.
    #    after that it flattens into a bumpy plateau and never returns to zero,
    #    because a sum of cosines at many frequencies is not a decaying kernel.
    #    "sinusoidal encodes distance" is a local statement; at range this
    #    carries an ordering, not a metric
    by_distance = torch.tensor([g[0, d] for d in range(T)])
    #    at d=0 each pair contributes sin^2 + cos^2 = 1, so the peak is E/2
    assert by_distance[0] == by_distance.max()
    assert abs(by_distance[0].item() - E / 2) < 1e-4
    assert all(by_distance[d] > by_distance[d + 1] for d in range(4))  # then not
    assert by_distance[4] < by_distance[5]  # the plateau starts here
    assert by_distance[T // 2 :].max() < 0.75 * by_distance[0]
    print("dot by distance:", [f"{v:.1f}" for v in by_distance[:10]], "...")

    #    and that sum is exactly what the identity predicts: E/2 cosines, one
    #    per column pair, evaluated at the distance
    theta = 10000 ** (-torch.arange(0, E, 2) / E)  # [E/2]
    closed = torch.cos(torch.arange(T).unsqueeze(1) * theta).sum(1)  # [T]
    assert (by_distance - closed).abs().max() < 1e-4

    # a learned table has no such structure: its diagonals are unrelated numbers
    gl = learned.weight @ learned.weight.T
    d5 = torch.diagonal(gl, offset=5)
    assert (d5 - d5[0]).abs().max() > 1.0

    # 5. the bridge to RoPE: shifting by k is a ROTATION by theta_i * k, the same
    #    rotation at every pos. build R from the definition and check it moves
    #    every position by k
    k = 7
    i = torch.arange(0, E, 2)
    ang = k / 10000 ** (i / E)  # [E/2] angle per pair, no pos in sight
    s, c = ang.sin(), ang.cos()

    p = enc.pe[: T - k]  # [T-k, E]
    sin_p, cos_p = p[:, 0::2], p[:, 1::2]
    # sin(a+b) = sin a cos b + cos a sin b ;  cos(a+b) = cos a cos b - sin a sin b
    rot = torch.empty_like(p)
    rot[:, 0::2] = sin_p * c + cos_p * s
    rot[:, 1::2] = cos_p * c - sin_p * s
    assert (rot - enc.pe[k:]).abs().max() < 1e-5
    print(f"shift by {k} == a fixed rotation, independent of position")

    # 6. it is a function, not a table: asking for a longer context recomputes
    #    the same numbers, so the shared prefix is identical. a learned table
    #    would have to grow new rows and train them from scratch
    longer = SinusoidalEmbedding(4 * T, E)
    assert (longer.pe[:T] - enc.pe).abs().max() == 0
    #    (extrapolating past block_size still needs a bigger buffer -- forward
    #     indexes a table. the point is only that the values are already decided)

    # 7. one buffer, and .to() DOES cast it -- unlike the tril in attention,
    #    which is bool and gets moved without a dtype change. the rule is the
    #    same one torch uses: cast floating buffers, leave the rest alone.
    #    note the cast happens after the fact, so it widens fp32 values rather
    #    than recomputing the formula in double
    assert [n for n, _ in enc.named_buffers()] == ["pe"]
    d64 = SinusoidalEmbedding(T, E).to(torch.float64)
    assert d64.pe.dtype == torch.float64
    assert (d64.pe - enc.pe.double()).abs().max() == 0

    # 8. what it is for: added to the token embedding, it makes otherwise
    #    identical tokens distinguishable by where they sit
    tok = Embedding(10, E)
    ids = torch.tensor([3, 3, 3])
    assert (tok(ids)[0] == tok(ids)[2]).all()  # same token, same vector
    h = tok(ids) + enc(torch.arange(3))
    assert (h[0] - h[2]).abs().max() > 1e-3  # not any more

    print("ok")
