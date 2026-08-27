import torch
from torch import Tensor


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    return z - z.exp().sum(dim=dim, keepdim=True).log()  # log(sum(exp)) done stably


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    n = targets.size(0)
    logp = log_softmax(logits, dim=-1)  # [N, V]
    picked = logp[torch.arange(n), targets]  # [N] -- the correct class's log-prob
    return -picked.mean()


if __name__ == "__main__":
    import math

    import torch.nn.functional as F
    from softmax import softmax

    N, V = 64, 4096

    # 1. matches F.cross_entropy and F.log_softmax
    logits = torch.randn(N, V)
    targets = torch.randint(0, V, (N,))
    assert (log_softmax(logits) - F.log_softmax(logits, -1)).abs().max() < 1e-5
    assert (
        cross_entropy(logits, targets) - F.cross_entropy(logits, targets)
    ).abs() < 1e-5

    # 2. loss at init: uniform logits -> ln(V)
    flat = torch.full((N, V), 0)
    print(
        f"uniform loss: {cross_entropy(flat, targets).item():.4f}   ln({V}) = {math.log(V):.4f}"
    )
    assert abs(cross_entropy(flat, targets).item() - math.log(V)) < 1e-4
    print(
        f"random-init incorrect loss: {cross_entropy(logits * 2, targets).item():.4f}"
    )
    print(f"random-init incorrect loss: {cross_entropy(logits, targets).item():.4f}")
    print(
        f"random-init correct loss: {cross_entropy(logits * 0.02, targets).item():.4f}"
    )

    # 3. why not log(softmax(x)): e^(big negative value) = 0, log(0) = -inf
    bad = torch.tensor([[0.0, -1000.0]])
    t = torch.tensor([1])
    naive = softmax(bad).log()[0, 1]
    stable = log_softmax(bad)[0, 1]
    print("log(softmax(x)) :", naive.item())
    print("log_softmax(x)  :", stable.item())
    assert naive.isinf()
    assert abs(cross_entropy(bad, t).item() - 1000.0) < 1e-3

    # 4. grad wrt logits is (p - onehot) / N
    lg = logits.clone().requires_grad_(True)
    cross_entropy(lg, targets).backward()
    onehot = torch.zeros(N, V)
    onehot[torch.arange(N), targets] = 1.0
    assert (lg.grad - (softmax(logits) - onehot) / N).abs().max() < 1e-8

    # and it matches torch's
    lr = logits.clone().requires_grad_(True)
    F.cross_entropy(lr, targets).backward()
    assert (lg.grad - lr.grad).abs().max() < 1e-8

    # 5. confident-and-right costs ~0, confident-and-wrong costs a lot
    conf = torch.tensor([[10.0, 0.0]])
    assert cross_entropy(conf, torch.tensor([0])).item() < 1e-4
    assert cross_entropy(conf, torch.tensor([1])).item() > 9.0

    print("ok")
