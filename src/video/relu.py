import torch
from module import Module
from torch import Tensor


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x > 0, x, 0.0)


if __name__ == "__main__":
    from torch import nn

    mine, ref = ReLU(), nn.ReLU()

    # 1. no parameters at all
    assert list(mine.parameters()) == []
    assert mine.state_dict() == {}

    # 2. forward matches exactly, any shape
    for shape in [(7,), (4, 8, 32)]:
        x = torch.randn(shape)
        assert torch.equal(mine(x), ref(x))

    # 3. gradient at exactly 0 -- torch picks the 0 subgradient
    xm = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    xr = xm.detach().clone().requires_grad_(True)
    mine(xm).sum().backward()
    ref(xr).sum().backward()
    assert xm.grad.tolist() == xr.grad.tolist() == [0.0, 0.0, 1.0]

    # 4. the other spellings disagree at 0
    def grad_at_zero(f):
        x = torch.zeros(1, requires_grad=True)
        f(x).sum().backward()
        return x.grad.item()

    assert grad_at_zero(lambda x: x.clamp(min=0)) == 1.0
    assert grad_at_zero(lambda x: torch.maximum(x, torch.zeros_like(x))) == 0.5
    assert grad_at_zero(mine) == 0.0

    # 5. dead units: negatives get exactly zero gradient
    x = torch.randn(1000, requires_grad=True)
    mine(x).sum().backward()
    assert (x.grad[x < 0] == 0).all()
    print(f"dead fraction: {(x.grad == 0).float().mean():.2f}")

    print("ok")
