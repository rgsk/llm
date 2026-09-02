from collections.abc import Iterable

import torch
from module import Module
from parameter import Parameter
from torch import Tensor


class AdamW:
    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        groups = list(params)
        assert groups, "AdamW got no parameters"
        if not isinstance(groups[0], dict):  # a plain iterable of parameters
            groups = [{"params": groups}]
        self.groups = [
            {
                "params": list(g["params"]),
                "weight_decay": g.get("weight_decay", weight_decay),
            }
            for g in groups
        ]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.state: dict[int, tuple[Tensor, Tensor]] = {}

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias_correction1 = 1 - b1**self.t
        bias_correction2 = 1 - b2**self.t

        for group in self.groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if id(p) not in self.state:
                    self.state[id(p)] = (torch.zeros_like(p), torch.zeros_like(p))
                m, v = self.state[id(p)]

                p.mul_(1 - self.lr * wd)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                self.state[id(p)] = (m, v)

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))

    def zero_grad(self) -> None:
        for group in self.groups:
            for p in group["params"]:
                p.grad = None

    def state_dict(self) -> dict:
        """m, v and t -- everything carried between steps. Keyed by POSITION,
        not by id(p): an id is this process's memory address, and means nothing
        in the next one."""
        return {
            "t": self.t,
            "state": [
                [self.state.get(id(p)) for p in g["params"]] for g in self.groups
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        """Rebind the saved moments onto THIS process's parameters, matched up
        group by group in the order state_dict wrote them."""
        assert len(sd["state"]) == len(self.groups), "group count changed"
        self.t = sd["t"]
        self.state = {}
        for group, saved in zip(self.groups, sd["state"]):
            assert len(saved) == len(group["params"]), "parameter count changed"
            for p, mv in zip(group["params"], saved):
                if mv is not None:
                    m, v = mv
                    assert m.shape == p.shape, "parameter shape changed"
                    self.state[id(p)] = (m.to(p.device), v.to(p.device))


def decay_groups(model: Module, weight_decay: float) -> list[dict]:
    """Decay matmul weights only. Biases and norm gains are 1D -- decaying those
    shrinks the normalization/offset the layer exists to provide."""
    params = list(model.parameters())
    return [
        {
            "params": [p for p in params if p.dim() >= 2],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in params if p.dim() < 2],
            "weight_decay": 0.0,
        },
    ]


if __name__ == "__main__":
    import math

    from linear import Linear
    from relu import ReLU
    from sequential import Sequential

    torch.manual_seed(0)

    # 1. step-for-step against torch.optim.AdamW, including a changing lr
    mine = Sequential(Linear(8, 16), ReLU(), Linear(16, 4))
    ref = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )
    mine.load_state_dict({k: v.clone() for k, v in ref.state_dict().items()})

    opt_m = AdamW(mine.parameters(), lr=1e-2, weight_decay=0.1)
    opt_r = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.1)

    x, y = torch.randn(32, 8), torch.randn(32, 4)
    worst = 0.0
    for step in range(50):
        lr = 1e-2 * (0.5 + 0.5 * step / 50)  # vary it, as the schedule will
        opt_m.lr = lr
        for gp in opt_r.param_groups:
            gp["lr"] = lr

        opt_m.zero_grad()
        opt_r.zero_grad()
        (mine(x) - y).square().mean().backward()
        (ref(x) - y).square().mean().backward()
        opt_m.step()
        opt_r.step()
        worst = max(
            worst,
            max(
                (a - b).abs().max().item()
                for a, b in zip(mine.parameters(), ref.parameters())
            ),
        )
    print(f"max param divergence over 50 steps: {worst:.3e}")
    assert worst < 1e-6

    # 2. decoupled decay: with zero gradient the weight shrinks by exactly
    #    (1 - lr*wd) per step -- no adaptive scaling involved
    class One(Module):
        def __init__(self):
            self.w = Parameter(torch.ones(3))

    o = One()
    opt = AdamW(o.parameters(), lr=0.1, weight_decay=0.5)
    o.w.grad = torch.zeros(3)
    for _ in range(3):
        opt.step()
    assert torch.allclose(o.w, torch.full((3,), (1 - 0.1 * 0.5) ** 3))

    # 3. bias correction makes the FIRST step ~lr in magnitude, regardless of
    #    gradient scale -- and in the direction that decreases the loss
    for scale in (1e-4, 1.0, 1e4, -1.0, -1e4):
        o = One()
        opt = AdamW(o.parameters(), lr=0.1, weight_decay=0.0)
        o.w.grad = torch.full((3,), scale)
        opt.step()
        moved = (1.0 - o.w[0]).item()
        expected = math.copysign(0.1, scale)  # step is lr * sign(g)
        print(f"grad {scale:9.0e} -> moved {moved:+.6f}")
        assert math.isclose(moved, expected, abs_tol=1e-4)

    # 1b. the grouped path, against torch with the same groups
    mine2 = Sequential(Linear(8, 16), ReLU(), Linear(16, 4))
    ref2 = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )
    mine2.load_state_dict({k: v.clone() for k, v in ref2.state_dict().items()})

    rp = list(ref2.parameters())
    opt_m = AdamW(decay_groups(mine2, 0.1), lr=1e-2)
    opt_r = torch.optim.AdamW(
        [
            {"params": [p for p in rp if p.dim() >= 2], "weight_decay": 0.1},
            {"params": [p for p in rp if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=1e-2,
    )
    for _ in range(50):
        opt_m.zero_grad()
        opt_r.zero_grad()
        (mine2(x) - y).square().mean().backward()
        (ref2(x) - y).square().mean().backward()
        opt_m.step()
        opt_r.step()
    assert (
        max(
            (a - b).abs().max().item()
            for a, b in zip(mine2.parameters(), ref2.parameters())
        )
        < 1e-6
    )

    # every parameter lands in exactly one group
    from gpt import GPT

    g = decay_groups(GPT(4096, 32, 64, 4, 3), 0.1)
    assert len(g[0]["params"]) == 47 and len(g[1]["params"]) == 23

    # 4. a saved optimizer resumes bit-exactly: 5 + 5 steps == 10 steps
    def make():
        torch.manual_seed(0)
        m = Sequential(Linear(8, 16), ReLU(), Linear(16, 4))
        return m, AdamW(decay_groups(m, 0.1), lr=1e-2)

    def run(model, opt, n):
        for _ in range(n):
            opt.zero_grad()
            (model(x) - y).square().mean().backward()
            opt.step()

    a, opt_a = make()
    run(a, opt_a, 10)

    b, opt_b = make()
    run(b, opt_b, 5)
    blob = {"model": b.state_dict(), "opt": opt_b.state_dict()}

    c, opt_c = make()
    c.load_state_dict(blob["model"])
    opt_c.load_state_dict(blob["opt"])
    run(c, opt_c, 5)
    assert opt_c.t == 10
    assert all((p - q).abs().max() == 0 for p, q in zip(a.parameters(), c.parameters()))

    # and the state is not decoration -- weights alone land somewhere else
    d, opt_d = make()
    d.load_state_dict(blob["model"])
    run(d, opt_d, 5)
    assert (
        max((p - q).abs().max() for p, q in zip(a.parameters(), d.parameters())) > 1e-4
    )

    print("ok")
