import torch
from torch import Tensor


def multinomial(probs: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Draw one index per row, with probability proportional to `probs`.
    The num_samples=1 case of torch.multinomial, by hand.

    Inverse-CDF sampling: a token with probability p occupies exactly p of the
    unit interval, so drawing u ~ Uniform[0, 1) and finding which slice it lands in
    samples from the distribution.

        probs:  0.1   0.2    0.05   0.65
        cdf:    0.1   0.3    0.35   1.0
        line: [--0.1--|---0.3---|0.35|--------1.0--------]
                bucket0  bucket1  b2        bucket3
    """
    B, V = probs.shape
    cdf = probs.cumsum(dim=-1)  # [B, V]
    u = torch.rand(B, 1, device=probs.device, generator=generator)
    idx = (cdf < u).sum(dim=-1, keepdim=True)  # how many slices u passed
    return idx.clamp(max=V - 1)  # float slop can push u past cdf[-1]


if __name__ == "__main__":
    V, N = 8, 40000
    target = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.0625, 0.0, 0.0, 0.0])
    probs = target.repeat(N, 1)

    # 1. the empirical distribution matches the target, as closely as torch's
    mine = multinomial(probs, torch.Generator().manual_seed(0)).flatten()
    ref = torch.multinomial(
        probs, 1, generator=torch.Generator().manual_seed(0)
    ).flatten()
    f_mine = torch.bincount(mine, minlength=V).float() / N
    f_ref = torch.bincount(ref, minlength=V).float() / N
    print("target:", [f"{p:.4f}" for p in target.tolist()])
    print("ours  :", [f"{p:.4f}" for p in f_mine.tolist()])
    print("torch :", [f"{p:.4f}" for p in f_ref.tolist()])
    err_mine = (f_mine - target).abs().max().item()
    err_ref = (f_ref - target).abs().max().item()
    print(f"max deviation -- ours {err_mine:.4f}   torch {err_ref:.4f}")
    assert err_mine < 0.01
    assert err_mine < err_ref * 3  # not systematically worse than torch

    # 2. zero-probability tokens are NEVER drawn
    assert f_mine[5:].sum() == 0

    # 3. a one-hot distribution is deterministic
    onehot = torch.zeros(100, V)
    onehot[:, 3] = 1.0
    assert (multinomial(onehot) == 3).all()

    # 4. shape, dtype, and range
    out = multinomial(probs[:16])
    assert out.shape == (16, 1)
    assert out.dtype == torch.int64  # usable as an index straight away
    assert out.min() >= 0 and out.max() < V

    # 5. seeded draws are reproducible
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(multinomial(probs, g1), multinomial(probs, g2))

    # 6. rows are independent: different rows, different distributions
    two = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(500, 1)
    drawn = multinomial(two).flatten()
    assert (drawn[0::2] == 0).all() and (drawn[1::2] == 1).all()

    print("ok")
