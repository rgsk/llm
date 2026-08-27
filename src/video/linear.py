import math

import torch
from module import Module
from parameter import Parameter
from torch import Tensor


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # torch inits with kaiming_uniform_(a=sqrt(5)), which reduces to U(+-1/sqrt(fan_in))
        bound = 1 / math.sqrt(in_features)
        self.weight = Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound)
        )
        self.bias = (
            Parameter(torch.empty(out_features).uniform_(-bound, bound))
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T  # [..., in] @ [in, out] -> [..., out]
        if self.bias is not None:
            out = out + self.bias
        return out


if __name__ == "__main__":
    from torch import nn

    # 1. same init, draw for draw, from the same seed
    torch.manual_seed(0)
    mine = Linear(4, 8)
    torch.manual_seed(0)
    ref = nn.Linear(4, 8)
    assert torch.equal(mine.weight, ref.weight)
    assert torch.equal(mine.bias, ref.bias)
    print("init matches exactly")

    # ...and the derivation is right: std of U(-b, b) is b/sqrt(3)
    big = Linear(10000, 1).weight
    assert abs(big.std().item() / (1 / math.sqrt(10000) / math.sqrt(3)) - 1) < 0.05

    # 2. same names and shapes
    assert mine.state_dict().keys() == ref.state_dict().keys()
    assert mine.weight.shape == ref.weight.shape == (8, 4)  # [out, in]

    # 3. same forward on 2D and on [B, T, E]
    mine.load_state_dict(ref.state_dict())
    for shape in [(5, 4), (2, 3, 4)]:
        x = torch.randn(shape)
        assert (mine(x) - ref(x)).abs().max() == 0
        assert mine(x).shape == ref(x).shape

    # 4. same grads, w.r.t. params and input
    xm = torch.randn(2, 3, 4, requires_grad=True)
    xr = xm.detach().clone().requires_grad_(True)
    mine(xm).square().sum().backward()
    ref(xr).square().sum().backward()
    assert (mine.weight.grad - ref.weight.grad).abs().max() == 0
    assert (mine.bias.grad - ref.bias.grad).abs().max() == 0
    assert (xm.grad - xr.grad).abs().max() == 0

    # 5. bias=False: no bias key at all, matching torch
    torch.manual_seed(1)
    nb = Linear(4, 8, bias=False)
    torch.manual_seed(1)
    nbref = nn.Linear(4, 8, bias=False)
    assert nb.bias is None and nbref.bias is None
    assert list(nb.state_dict()) == list(nbref.state_dict()) == ["weight"]
    assert torch.equal(nb.weight, nbref.weight)
    x = torch.randn(5, 4)
    assert (nb(x) - nbref(x)).abs().max() == 0

    print("ok")
