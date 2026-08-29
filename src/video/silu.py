from module import Module
from torch import Tensor


class SiLU(Module):
    """x * sigmoid(x), also called Swish.

    ReLU throws away everything below zero, and the gradient there is exactly 0:
    a unit that goes negative for every input stops learning. SiLU squashes the
    negative side instead of deleting it -- it dips to about -0.28 near x = -1.28
    and returns to 0, so the gradient is small but never dead.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


if __name__ == "__main__":
    import torch
    from relu import ReLU
    from torch import nn

    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    s = SiLU()

    # 1. matches torch, forward and backward. nn.SiLU is a fused kernel, so it
    #    rounds differently -- the gap is ~8 float32 ulps, not a disagreement
    print("fwd diff:", (s(x) - nn.SiLU()(x)).abs().max().item())
    assert (s(x) - nn.SiLU()(x)).abs().max() < 1e-6
    xm = x.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    s(xm).sum().backward()
    nn.SiLU()(xr).sum().backward()
    assert (xm.grad - xr.grad).abs().max() < 1e-6

    # 2. the shape of the curve: it is NOT monotonic, unlike ReLU
    g = torch.linspace(-6, 6, 1201)
    y = s(g)
    lo = y.argmin()
    print(f"minimum {y[lo]:.4f} at x = {g[lo]:.3f}")
    assert abs(y.min() + 0.2785) < 1e-3  # the dip
    assert abs(g[lo] + 1.2785) < 1e-2
    assert not (y[1:] >= y[:-1]).all()  # dips, then recovers

    # 3. it agrees with ReLU where ReLU is confident, and differs near zero
    big = torch.tensor([-30.0, -20.0, 20.0, 30.0])
    assert (s(big) - ReLU()(big)).abs().max() < 1e-6
    # it converges slowly though: at x = -10 the gap is still 4.5e-4
    assert (s(torch.tensor([-10.0])) - ReLU()(torch.tensor([-10.0]))).abs() > 1e-4
    small = torch.tensor([-1.0, 0.0, 1.0])
    print("silu:", [f"{v:.4f}" for v in s(small).tolist()])
    print("relu:", [f"{v:.4f}" for v in ReLU()(small).tolist()])
    assert (s(small) - ReLU()(small)).abs().max() > 0.2

    # 4. no dead zone: gradient is nonzero everywhere ReLU's is zero
    neg = torch.linspace(-8, -0.01, 100).requires_grad_(True)
    s(neg).sum().backward()
    assert (neg.grad != 0).all()
    neg2 = torch.linspace(-8, -0.01, 100).requires_grad_(True)
    ReLU()(neg2).sum().backward()
    assert (neg2.grad == 0).all()  # ReLU: every one of these units is dead

    # 5. exactly 0 at 0, and no parameters to learn
    assert s(torch.zeros(3)).abs().max() == 0
    assert list(s.parameters()) == []

    print("ok")
