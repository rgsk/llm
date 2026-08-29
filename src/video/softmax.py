import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()  # subtracting the max: exp() now peaks at exp(0) == 1
    return e / e.sum(dim=dim, keepdim=True)


if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)

    def naive(x, dim=-1):
        e = x.exp()
        return e / e.sum(dim=dim, keepdim=True)

    # 1. matches F.softmax
    x = torch.randn(4, 8, 16)
    assert (softmax(x) - F.softmax(x, dim=-1)).abs().max() < 1e-7
    assert (softmax(x, dim=1) - F.softmax(x, dim=1)).abs().max() < 1e-7

    # 2. rows are a probability distribution
    assert (softmax(x).sum(-1) - 1).abs().max() < 1e-6
    assert (softmax(x) >= 0).all()

    # 3. the max subtraction is the whole point
    big = torch.tensor([[1000.0, 1001.0, 1002.0]])
    print("naive big: ", naive(big).tolist())
    print("stable big:", softmax(big).tolist())
    assert naive(big).isnan().any()
    assert torch.allclose(softmax(big), F.softmax(big, dim=-1))

    # and it changes nothing mathematically: softmax is shift-invariant
    assert (softmax(x) - softmax(x + 12.34)).abs().max() < 1e-6

    # tiny values underflow to a uniform answer instead of 0/0
    small = torch.tensor([[-1000.0, -1000.0, -1000.0]])
    print("naive small: ", naive(small).tolist())
    print("stable small:", softmax(small).tolist())
    assert naive(small).isnan().any()
    assert torch.allclose(softmax(small), torch.full((1, 3), 1 / 3))

    # 4. masked rows: -inf entries get exactly 0 probability
    masked = torch.tensor([[1.0, 2.0, float("-inf"), float("-inf")]])
    p = softmax(masked)
    assert p[0, 2] == 0 and p[0, 3] == 0
    assert abs(p.sum().item() - 1.0) < 1e-6
    assert torch.equal(p, F.softmax(masked, dim=-1))

    # 5. grads match
    xm = torch.randn(4, 8, 16, requires_grad=True)
    xr = xm.detach().clone().requires_grad_(True)
    softmax(xm).square().sum().backward()
    F.softmax(xr, dim=-1).square().sum().backward()
    assert (xm.grad - xr.grad).abs().max() < 1e-7

    print("ok")
