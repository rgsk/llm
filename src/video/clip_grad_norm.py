from collections.abc import Iterable

import torch
from parameter import Parameter
from torch import Tensor


@torch.no_grad()
def clip_grad_norm(params: Iterable[Parameter], max_norm: float) -> Tensor:
    """Rescale all gradients so their combined L2 norm is at most max_norm.
    Returns the norm BEFORE clipping -- the number worth logging."""
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = torch.sqrt(sum((g * g).sum() for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:  # only ever shrink, never amplify a small gradient
        for g in grads:
            g.mul_(clip_coef)
    return total_norm


if __name__ == "__main__":
    from linear import Linear
    from relu import ReLU
    from sequential import Sequential

    torch.manual_seed(0)
    mine = Sequential(Linear(8, 16), ReLU(), Linear(16, 4))
    ref = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )
    mine.load_state_dict({k: v.clone() for k, v in ref.state_dict().items()})
    x, y = torch.randn(32, 8), torch.randn(32, 4)

    for max_norm, scale in ((1.0, 100.0), (1.0, 1e-4), (0.5, 1.0)):
        mine.zero_grad()
        ref.zero_grad()
        ((mine(x) - y) * scale).square().mean().backward()
        ((ref(x) - y) * scale).square().mean().backward()

        n_mine = clip_grad_norm(mine.parameters(), max_norm)
        n_ref = torch.nn.utils.clip_grad_norm_(ref.parameters(), max_norm)

        after = torch.sqrt(sum((p.grad * p.grad).sum() for p in mine.parameters()))
        print(f"max_norm={max_norm}  before={n_mine:9.4f}  after={after:.4f}")
        # relative, not absolute: at |g|~6446 float32 has ~5e-4 of slack, and torch
        # takes per-tensor norms then the norm of those while we sum squares directly
        assert torch.isclose(n_mine, n_ref, rtol=1e-5)
        assert all(
            (a.grad - b.grad).abs().max() < 1e-6
            for a, b in zip(mine.parameters(), ref.parameters())
        )
        assert after <= max_norm + 1e-5

    # a gradient already under the threshold is left completely alone
    mine.zero_grad()
    (mine(x) - y).square().mean().mul(1e-6).backward()
    before = [p.grad.clone() for p in mine.parameters()]
    clip_grad_norm(mine.parameters(), 1.0)
    assert all(torch.equal(a, p.grad) for a, p in zip(before, mine.parameters()))
    print("ok")
