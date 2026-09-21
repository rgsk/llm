from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from backend import USE_TORCH


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    return z - z.exp().sum(dim=dim, keepdim=True).log()  # log(sum(exp)) done stably


def our_cross_entropy(
    logits: Tensor, targets: Tensor, *, ignore_index: int = -100
) -> Tensor:
    """ignore_index is keyword-only, because F.cross_entropy takes `weight` third."""
    n = targets.size(0)
    logp = log_softmax(logits, dim=-1)  # [N, V]
    # ignored targets are not valid columns, so clamp them and drop them after
    picked = logp[torch.arange(n), targets.clamp(min=0)]  # the true class's log-prob
    return -picked[targets != ignore_index].mean()


if TYPE_CHECKING:
    cross_entropy = F.cross_entropy  # see backend.py
else:
    cross_entropy = F.cross_entropy if USE_TORCH else our_cross_entropy


if __name__ == "__main__":
    # our_cross_entropy by name: under VIDEO_BACKEND=torch the alias is F.cross_entropy
    import math

    import torch.nn.functional as F

    from softmax import softmax

    N, V = 64, 4096

    # 1. matches F.cross_entropy and F.log_softmax
    logits = torch.randn(N, V)
    targets = torch.randint(0, V, (N,))
    assert (log_softmax(logits) - F.log_softmax(logits, -1)).abs().max() < 1e-5
    assert (
        our_cross_entropy(logits, targets) - F.cross_entropy(logits, targets)
    ).abs() < 1e-5

    # 2. loss at init: uniform logits -> ln(V)
    flat = torch.full((N, V), 0)
    print(
        f"uniform loss: {our_cross_entropy(flat, targets).item():.4f}   ln({V}) = {math.log(V):.4f}"
    )
    assert abs(our_cross_entropy(flat, targets).item() - math.log(V)) < 1e-4
    print(
        f"random-init incorrect loss: {our_cross_entropy(logits * 2, targets).item():.4f}"
    )
    print(
        f"random-init incorrect loss: {our_cross_entropy(logits, targets).item():.4f}"
    )
    print(
        f"random-init correct loss: {our_cross_entropy(logits * 0.02, targets).item():.4f}"
    )

    # 3. why not log(softmax(x)): e^(big negative value) = 0, log(0) = -inf
    bad = torch.tensor([[0.0, -1000.0]])
    t = torch.tensor([1])
    naive = softmax(bad).log()[0, 1]
    stable = log_softmax(bad)[0, 1]
    print("log(softmax(x)) :", naive.item())
    print("log_softmax(x)  :", stable.item())
    assert naive.isinf()
    assert abs(our_cross_entropy(bad, t).item() - 1000.0) < 1e-3

    # 4. grad wrt logits is (p - onehot) / N
    lg = logits.clone().requires_grad_(True)
    our_cross_entropy(lg, targets).backward()
    onehot = torch.zeros(N, V)
    onehot[torch.arange(N), targets] = 1.0
    assert (lg.grad - (softmax(logits) - onehot) / N).abs().max() < 1e-8

    # and it matches torch's
    lr = logits.clone().requires_grad_(True)
    F.cross_entropy(lr, targets).backward()
    assert (lg.grad - lr.grad).abs().max() < 1e-8

    # 5. confident-and-right costs ~0, confident-and-wrong costs a lot
    conf = torch.tensor([[10.0, 0.0]])
    assert our_cross_entropy(conf, torch.tensor([0])).item() < 1e-4
    assert our_cross_entropy(conf, torch.tensor([1])).item() > 9.0

    # 6. ignore_index: masked rows leave the mean and the gradient alone
    masked = targets.clone()
    masked[::2] = -100
    assert (
        our_cross_entropy(logits, masked) - F.cross_entropy(logits, masked)
    ).abs() < 1e-5
    kept = our_cross_entropy(logits[1::2], targets[1::2])  # the same rows, by hand
    assert (our_cross_entropy(logits, masked) - kept).abs() < 1e-5
    lm = logits.clone().requires_grad_(True)
    our_cross_entropy(lm, masked).backward()
    assert lm.grad[::2].abs().max() == 0

    # and nothing to average is nan, as it is in torch -- a batch of all prompt
    # tokens is a bug, not a free step
    allmasked = torch.full_like(targets, -100)
    assert our_cross_entropy(logits, allmasked).isnan()
    assert F.cross_entropy(logits, allmasked).isnan()

    print("ok")
