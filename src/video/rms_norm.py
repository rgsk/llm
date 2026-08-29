import torch
from module import Module
from parameter import Parameter
from torch import Tensor


class RMSNorm(Module):
    """LayerNorm with the centering step deleted.

    LayerNorm does two things: it re-centers (subtract the mean) and it re-scales
    (divide by the spread). Zhang & Sennrich (2019) tested them separately and
    found the re-scaling is what stabilises training -- the re-centering is close
    to free to remove. So drop the mean, divide by the root-mean-square instead
    of the standard deviation, and drop the bias with it.

    One reduction instead of two, one parameter vector instead of two.

    Calculation:
        rms(x) = sqrt(mean(x^2) + eps)          # over the last dim, C features
        y      = x / rms(x) * gamma
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        ms = (x**2).mean(dim=-1, keepdim=True)  # [..., 1]

        # divide by the root-mean-square; eps goes INSIDE the sqrt (same place
        # LayerNorm adds it) for numerical safety when the token is near-zero.
        x_hat = x / torch.sqrt(ms + self.eps)  # [..., C]

        # learned per-feature scale only — no bias (RMSNorm has no shift).
        return x_hat * self.weight


if __name__ == "__main__":
    from layer_norm import LayerNorm
    from torch import nn

    torch.manual_seed(0)
    E = 32
    mine, ref = RMSNorm(E), nn.RMSNorm(E, eps=1e-5)

    # 1. one parameter, not two -- and no RNG in the init
    assert torch.equal(mine.weight, torch.ones(E))
    assert mine.state_dict().keys() == ref.state_dict().keys() == {"weight"}
    assert not hasattr(mine, "bias")
    assert len(list(mine.parameters())) == 1
    assert len(list(LayerNorm(E).parameters())) == 2

    # 2. forward matches torch on [B, T, E]
    x = torch.randn(4, 8, E)
    print("float32 fwd diff:", (mine(x) - ref(x)).abs().max().item())
    assert (mine(x) - ref(x)).abs().max() < 1e-6

    mine.to(torch.float64)
    ref.double()
    x64 = x.double()
    print("float64 fwd diff:", (mine(x64) - ref(x64)).abs().max().item())
    assert (mine(x64) - ref(x64)).abs().max() < 1e-14
    mine.to(torch.float32)
    ref.float()

    # 3. it normalises the RMS to 1, and does NOT touch the mean.
    #    x + 3 shifts every row off centre; LayerNorm erases that, RMSNorm keeps it
    out = RMSNorm(E)(x)
    assert (out.square().mean(dim=-1) - 1).abs().max() < 1e-3
    shifted = RMSNorm(E)(x + 3.0)
    print(
        f"row means -- in {(x + 3.0).mean(-1).abs().mean():.4f}  "
        f"rms out {shifted.mean(-1).abs().mean():.4f}  "
        f"layer out {LayerNorm(E)(x + 3.0).mean(-1).abs().mean():.2e}"
    )
    assert shifted.mean(dim=-1).abs().mean() > 0.5  # offset survives
    assert LayerNorm(E)(x + 3.0).mean(dim=-1).abs().max() < 1e-5  # offset erased

    # 4. on already-centred rows the two are the SAME layer. mean 0 makes
    #    mean-square == variance, so the only difference left is LayerNorm's bias
    c = x - x.mean(dim=-1, keepdim=True)
    print(
        "centred rows, rms vs layer:",
        (RMSNorm(E)(c) - LayerNorm(E)(c)).abs().max().item(),
    )
    assert (RMSNorm(E)(c) - LayerNorm(E)(c)).abs().max() < 1e-5

    # 5. the gain scales, and there is nothing to shift by
    r = RMSNorm(E)
    with torch.no_grad():
        r.weight.fill_(2.0)
    assert (r(x) / 2.0 - RMSNorm(E)(x)).abs().max() < 1e-6

    # 6. grads match torch, w.r.t. both the weight and the input
    xm = torch.randn(4, 8, E, requires_grad=True)
    xr = xm.detach().clone().requires_grad_(True)
    with torch.no_grad():
        ref.weight.normal_()
    mine.load_state_dict(ref.state_dict())
    mine(xm).square().sum().backward()
    ref(xr).square().sum().backward()
    assert (mine.weight.grad - ref.weight.grad).abs().max() < 1e-4
    assert (xm.grad - xr.grad).abs().max() < 1e-5

    # 7. eps guards an all-zero row instead of producing nan
    zeros = torch.zeros(2, E)
    assert torch.isfinite(RMSNorm(E)(zeros)).all()
    assert (RMSNorm(E)(zeros) == 0).all()

    # 8. it is cheaper: one reduction and one parameter vector fewer
    big = torch.randn(64, 512, 768)
    import time

    rn, ln = RMSNorm(768), LayerNorm(768)
    for f in (rn, ln):
        f(big)  # warm up
    t = {}
    for name, f in (("rms", rn), ("layer", ln)):
        t0 = time.perf_counter()
        for _ in range(20):
            f(big)
        t[name] = time.perf_counter() - t0
    print(
        f"20 passes over [64,512,768] -- rms {t['rms']:.3f}s   layer {t['layer']:.3f}s"
    )

    """
    Test 8, treat as indicative only.
    1.16s → 0.70s, about 40% faster, on CPU with no fused kernels. 
    In a real model both are memory-bound and a tiny fraction of runtime;
    the honest reason RMSNorm won the field is that it's simpler and loses nothing,
    not that it's 40% faster.
    """

    print("ok")
