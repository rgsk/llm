import torch
from module import Module
from torch import Tensor


class Dropout(Module):
    """Zero each activation independently with probability p, then scale the
    survivors by 1/(1-p) so the layer keeps its expected value.

    Scaling at train time -- "inverted dropout" -- is what makes eval a plain
    identity. Scaling by (1-p) at eval instead gives the same answer, but leaves
    a correction factor baked into the deployed model forever.

    No parameters. It is a Module only so that .eval() can reach it.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p < 1.0, "p is the DROP probability; p=1 would zero everything"
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            # self.p == 0.0 guard is worth keeping, even if torch.rand_like(x) >= self.p
            # already handles 0.0 case
            # without it, p=0.0
            # still allocates rand_like(x), and on the [B, nh, T, T] attention weights
            # to produce identical output
            # that is hundreds of MB of pure waste per forward
            return x
        keep = torch.rand_like(x) >= self.p  # a fresh mask on every call
        return x * keep / (1.0 - self.p)


if __name__ == "__main__":
    from torch import nn

    torch.manual_seed(0)
    x = torch.randn(200, 300)
    P = 0.3
    d, ref = Dropout(P), nn.Dropout(P)

    # 1. eval is exactly the identity -- not approximately
    d.eval(), ref.eval()
    assert (d(x) - x).abs().max() == 0
    assert (ref(x) - x).abs().max() == 0

    # 2. train mode drops about p of the entries -- the same rate torch does
    d.train(), ref.train()
    out, out_ref = d(x), ref(x)
    frac, frac_ref = (out == 0).float().mean(), (out_ref == 0).float().mean()
    print(f"zeroed  ours {frac:.4f}   torch {frac_ref:.4f}   target {P:.4f}")
    assert abs(frac - P) < 0.01 and abs(frac_ref - P) < 0.01

    # 3. survivors are scaled by exactly 1/(1-p), not passed through untouched
    kept = out != 0
    assert (out[kept] - x[kept] / (1 - P)).abs().max() < 1e-6

    # 4. the expected value survives -- that is what the scaling buys.
    #    the variance does not: dropout is noise, and 1/(1-p) inflates the spread
    print(f"mean    in {x.mean():.4f}   out {out.mean():.4f}")
    print(f"std     in {x.std():.4f}   out {out.std():.4f}")
    assert abs(out.mean() - x.mean()) < 0.02
    assert abs(out.std() / x.std() - (1 / (1 - P)) ** 0.5) < 0.05

    # 4b. the same thing seen per ELEMENT rather than in aggregate, on a tensor
    #     that is nothing like a normal distribution. Every survivor is multiplied
    #     by exactly 1/(1-p) whatever its magnitude, so averaging enough draws
    #     hands back x itself.
    a = torch.arange(1.0, 11.0)
    print("x            ", [f"{v:.3f}" for v in a.tolist()])
    print("one draw     ", [f"{v:.3f}" for v in d(a).tolist()])
    print("out / x      ", [f"{v:.3f}" for v in (d(a) / a).tolist()])  # 0 or 1/(1-p)
    N = 10000
    avg = torch.stack([d(a) for _ in range(N)]).mean(0)
    print(f"mean of {N}", [f"{v:.3f}" for v in avg.tolist()])
    #     sd of each element's average is x_i * sqrt(p/(1-p)/N) -- it grows with
    #     x_i, which is why a single flat tolerance would be wrong here
    assert ((avg - a).abs() < 5 * a * (P / (1 - P) / N) ** 0.5).all()

    # 5. a fresh mask every call
    assert not torch.equal(d(x), d(x))

    # 6. gradient reaches survivors only, scaled by the same factor
    xg = x.clone().requires_grad_(True)
    o = d(xg)
    o.sum().backward()
    g, scale = xg.grad, 1 / (1 - P)
    assert ((g == 0) | ((g - scale).abs() < 1e-6)).all()  # only 0 or 1/(1-p)
    assert torch.equal(g == 0, o == 0)  # dropped units get no gradient

    # 7. p=0 is the identity in train mode too
    assert (Dropout(0.0)(x) - x).abs().max() == 0

    # 8. the flag propagates from a parent. Dropout has no parameters, so this
    #    is the ONLY thing that makes it behave differently at inference
    class Net(Module):
        def __init__(self):
            self.drop = Dropout(0.5)

        def forward(self, x):
            return self.drop(x)

    n = Net()
    assert n.drop.training
    n.eval()
    assert not n.drop.training
    assert (n(x) - x).abs().max() == 0

    # 9. p must be a probability, and p=1 is refused rather than dividing by zero
    for bad in (1.0, 1.5, -0.1):
        try:
            Dropout(bad)
            raise SystemExit(f"should have failed: {bad}")
        except AssertionError:
            pass

    # 10. why `>=` and not `>`. torch.rand samples [0, 1): 0.0 IS attainable,
    #     about 1 draw in 45 million, while 1.0 never is. So the low boundary is
    #     the only one reachable -- and that is exactly where p=0 sits. Seed 12
    #     plants one in this tensor, at index 411302.
    class LooseDropout(Module):
        """The same layer written with `>`."""

        def __init__(self, p):
            self.p = p

        def forward(self, x):
            return x * (torch.rand_like(x) > self.p) / (1.0 - self.p)

    ones = torch.ones(1 << 20)
    torch.manual_seed(12)
    loose = LooseDropout(0.0)(ones)
    hit = (loose == 0).nonzero().flatten().tolist()
    print(f"p=0 with `>` : dropped {len(hit)} of {ones.numel():,} at {hit}")
    assert len(hit) == 1  # a "disabled" dropout that still drops

    torch.manual_seed(12)  # same draws, including that 0.0
    assert (Dropout(0.0)(ones) - ones).abs().max() == 0  # ours: exactly identity

    print("ok")
