from dropout import Dropout
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from silu import SiLU
from torch import Tensor


class GatedFeedForward(Module):
    """SwiGLU: the FFN's single hidden activation is replaced by a product of two.

    FeedForward computes  down(act(up(x))).
    This computes          down(act(gate(x)) * up(x)).

    The `up` path carries the values; the `act(gate)` path is a learned, per-unit
    multiplier that can scale any of them toward zero. ReLU also gates -- but on
    the value itself, so a unit's magnitude and whether it survives are the same
    number. Splitting them is the whole idea (Shazeer, 2020).

    gate and up come from ONE Linear of width 2*hidden, for the same reason qkv
    is one matrix: they read the same x and write disjoint halves.
    """

    def __init__(self, n_embed: int, dropout: float = 0.0):
        # param-matched to the ReLU FFN: that has 2 * E * 4E weights, SwiGLU has
        # 3 * E * h, so h = 8E/3. Rounded to a multiple of 64 for GEMM shapes.
        hidden = round(8 * n_embed / 3 / 64) * 64  # E=512 -> 1344
        self.gate_up = Linear(n_embed, 2 * hidden, bias=False)
        self.activation = SiLU()
        self.down = ResidualProj(hidden, n_embed, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)  # each [B, T, h]
        return self.dropout(self.down(self.activation(gate) * up))  # [B, T, E]


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from feed_forward import FeedForward
    from torch import nn

    torch.manual_seed(0)
    B, T, E = 2, 8, 512
    x = torch.randn(B, T, E)
    ff = GatedFeedForward(E)
    hidden = round(8 * E / 3 / 64) * 64

    # 1. shape preserved, and the hidden width is the param-matched one
    assert ff(x).shape == (B, T, E)
    assert hidden == 1344
    assert ff.gate_up.weight.shape == (2 * hidden, E)
    assert ff.down.weight.shape == (E, hidden)

    # 2. the point of h = 8E/3: match the ReLU FFN's parameter count, so any
    #    quality difference is the architecture and not extra capacity.
    #    3*E*h == 2*E*4E  =>  h = 8E/3 = 1365.33 here. At exactly that h the
    #    weights match to 0.02%; rounding down to a multiple of 64 gives up
    #    1.56%, and the ReLU FFN's biases account for the last 0.1%
    n_gated = sum(p.numel() for p in ff.parameters())
    n_relu = sum(p.numel() for p in FeedForward(E).parameters())
    print(
        f"params -- swiglu {n_gated:,}   relu ffn {n_relu:,}   "
        f"ratio {n_gated / n_relu:.4f}"
    )
    assert 3 * E * round(8 * E / 3) / (2 * E * 4 * E) > 0.999  # exact h: dead on
    assert abs(n_gated / n_relu - 1) < 0.02  # rounded h: close enough

    # 3. the keys main.py's checkpoints were written with, and no biases
    assert [n for n, _ in ff.named_parameters()] == ["gate_up.weight", "down.weight"]
    assert ff.gate_up.bias is None and ff.down.bias is None

    # 4. chunk really splits [gate | up], matching the (two h) layout on disk
    w = ff.gate_up(x)
    gate, up = ff.gate_up(x).chunk(2, dim=-1)
    assert torch.equal(gate, w[..., :hidden]) and torch.equal(up, w[..., hidden:])
    assert torch.equal(ff(x), ff.dropout(ff.down(ff.activation(gate) * up)))

    # 5. it is position-wise: every token goes through alone
    one = ff(x[:, [3], :])
    assert (one[:, 0] - ff(x)[:, 3]).abs().max() < 1e-5

    # 6. the gate can switch a hidden unit off without touching `up`.
    #    driving gate unit 0 very negative kills it; `up` is unchanged
    probe = GatedFeedForward(E)
    with torch.no_grad():
        probe.down.weight.zero_()
        probe.down.weight[0, 0] = 1.0  # read out hidden unit 0 only
        before = probe(x)[..., 0].clone()
        probe.gate_up.weight[0] *= 0.0  # gate 0 -> silu(0) == 0
    after = probe(x)[..., 0]
    assert before.abs().max() > 0
    assert after.abs().max() == 0

    # 7. against a torch mirror: same forward, same grads
    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up = nn.Linear(E, 2 * hidden, bias=False)
            self.down = nn.Linear(hidden, E, bias=False)

        def forward(self, x):
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.down(F.silu(gate) * up)

    ref = Ref()
    assert ff.state_dict().keys() == ref.state_dict().keys()
    ff.load_state_dict(ref.state_dict())
    xm = x.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    om, orf = ff(xm), ref(xr)
    print("vs torch mirror:", (om - orf).abs().max().item())
    assert (om - orf).abs().max() < 1e-5
    om.square().sum().backward()
    orf.square().sum().backward()
    assert (xm.grad - xr.grad).abs().max() < 1e-4
    assert all(
        (a.grad - b.grad).abs().max() < 1e-3
        for a, b in zip(ff.parameters(), ref.parameters())
    )

    # 8. down is a ResidualProj, so GPT._init_weights shrinks it like attn.proj
    assert isinstance(ff.down, ResidualProj)

    print("ok")
