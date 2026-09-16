from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from backend import USE_TORCH
from module import Module
from parameter import Parameter


class OurLayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # /n, not /(n-1)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias


if TYPE_CHECKING:
    LayerNorm = nn.LayerNorm  # see backend.py
else:
    LayerNorm = nn.LayerNorm if USE_TORCH else OurLayerNorm


if __name__ == "__main__":
    # OurLayerNorm by name: under VIDEO_BACKEND=torch the alias is nn.LayerNorm
    from torch import nn

    E = 32
    mine, ref = OurLayerNorm(E), nn.LayerNorm(E)

    # 1. init is deterministic: ones and zeros, no RNG
    assert torch.equal(mine.weight, torch.ones(E))
    assert torch.equal(mine.bias, torch.zeros(E))
    assert mine.state_dict().keys() == ref.state_dict().keys()
    assert mine.eps == ref.eps == 1e-5

    # 2. forward matches on [B, T, E]
    x = torch.randn(4, 8, E)
    print("float32 fwd diff:", (mine(x) - ref(x)).abs().max().item())
    assert (mine(x) - ref(x)).abs().max() < 1e-6

    # in float64 the residual is pure arithmetic reordering
    mine.to(torch.float64)
    ref.double()
    x64 = x.double()
    print("float64 fwd diff:", (mine(x64) - ref(x64)).abs().max().item())
    assert (mine(x64) - ref(x64)).abs().max() < 1e-14
    mine.to(torch.float32)
    ref.float()

    # 3. it actually normalizes: mean 0, var 1 per row
    out = OurLayerNorm(E)(x)
    assert out.mean(dim=-1).abs().max() < 1e-6
    assert (out.var(dim=-1, unbiased=False) - 1).abs().max() < 1e-3

    # 4. affine params do what they say
    ln = OurLayerNorm(E)
    with torch.no_grad():
        ln.weight.fill_(2.0)
        ln.bias.fill_(5.0)
    assert ((ln(x) - 5.0) / 2.0 - OurLayerNorm(E)(x)).abs().max() < 1e-5

    # 5. grads match, w.r.t. params and input
    xm = torch.randn(4, 8, E, requires_grad=True)
    xr = xm.detach().clone().requires_grad_(True)
    mine.load_state_dict(ref.state_dict())
    mine(xm).square().sum().backward()
    ref(xr).square().sum().backward()
    assert (mine.weight.grad - ref.weight.grad).abs().max() < 1e-4
    assert (mine.bias.grad - ref.bias.grad).abs().max() < 1e-4
    assert (xm.grad - xr.grad).abs().max() < 1e-5

    # 6. biased vs unbiased variance really matters
    class Wrong(OurLayerNorm):
        def forward(self, x):
            var = x.var(dim=-1, keepdim=True)  # torch defaults to unbiased=True
            xhat = (x - x.mean(-1, keepdim=True)) / torch.sqrt(var + self.eps)
            return xhat * self.weight + self.bias

    print("unbiased-var error:", (Wrong(E)(x) - ref(x)).abs().max().item())
    assert (Wrong(E)(x) - ref(x)).abs().max() > 1e-3

    print("ok")
