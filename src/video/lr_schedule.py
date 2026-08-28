import math


def get_lr(
    step: int, *, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float
) -> float:
    """Learning rate at `step`: linear warmup to max_lr, then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__":
    import torch

    W, M, MAX, MIN = 20, 200, 1e-2, 1e-3
    kw = dict(warmup_steps=W, max_steps=M, max_lr=MAX, min_lr=MIN)
    lrs = [get_lr(s, **kw) for s in range(M + 1)]

    # 1. endpoints
    assert lrs[0] == MAX / W  # step 0 is one warmup unit, not zero
    assert math.isclose(lrs[W - 1], MAX)  # warmup ends AT max_lr
    assert math.isclose(lrs[W], MAX)  # cosine starts there too -- continuous
    assert math.isclose(lrs[M], MIN)  # decay ends exactly at min_lr
    assert get_lr(M + 500, **kw) == MIN  # past the end, stay at min_lr

    # 2. shape: strictly up through warmup, strictly down after
    assert all(lrs[i] < lrs[i + 1] for i in range(W - 1))
    assert all(lrs[i] > lrs[i + 1] for i in range(W, M))

    # peak is at the warmup/decay seam. lrs[W] is one ulp over max_lr because the
    # cosine branch computes min_lr + 1.0*(max_lr - min_lr) instead of max_lr itself
    assert lrs.index(max(lrs)) == W
    assert max(lrs) <= MAX * (1 + 1e-12)

    # min_lr is the floor of the DECAY, not a global floor: warmup starts below it
    assert min(lrs) == lrs[0] < MIN
    assert min(lrs[W:]) == lrs[M] == MIN

    # 3. the cosine half matches torch's CosineAnnealingLR exactly
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=MAX)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=M - W, eta_min=MIN)
    ref = []
    for _ in range(M - W + 1):
        ref.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(lrs[W:], ref))

    # 4. half-way through decay the lr is the midpoint of max and min
    assert math.isclose(get_lr((W + M) // 2, **kw), (MAX + MIN) / 2, rel_tol=1e-3)

    # 5. see it
    # for s in range(0, M + 1, 10):
    #     print(f"{s:>4} {lrs[s]:.5f} {'#' * round(lrs[s] / MAX * 50)}")

    print("ok")
