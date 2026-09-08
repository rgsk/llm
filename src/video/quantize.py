"""int8 weight-only quantization: store w as small ints plus a float scale."""

import torch
from torch import Tensor

from linear import Linear
from module import Module

QMAX = 127  # symmetric range, so -128 goes unused and 0 maps to exactly 0


def quantize(w: Tensor, dim: int | None = -1) -> tuple[Tensor, Tensor]:
    """w -> (int8, scale), with w ~= q * scale. dim=None is one scale for all of w."""
    amax = w.abs().amax() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    # an all-zero row gives scale 0, and w / 0 is nan. the int8 cast below
    # happens to turn that nan into 0 on both cpu and cuda, but that is
    # undefined behaviour, and any variant that skips the cast keeps the nan
    scale = (amax / QMAX).clamp(min=torch.finfo(w.dtype).tiny)
    # the clamp is defensive under this scale rule -- w/scale peaks at 127 and
    # cannot round past it -- but any other rule (percentile clipping, a shared
    # scale) can, and int8 wraps 128 to -128 in silence
    q = (w / scale).round().clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale


def dequantize(q: Tensor, scale: Tensor) -> Tensor:
    return q.to(scale.dtype) * scale


class QuantizedLinear(Module):
    """A Linear whose weight is stored int8. Built from a trained Linear, not trained."""

    def __init__(self, linear: Linear, dim: int | None = -1):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        q, scale = quantize(linear.weight.data, dim)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        self.bias = None
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())

    def forward(self, x: Tensor) -> Tensor:
        # the whole matrix is reconstructed every call: this saves memory at
        # rest, not compute. a real int8 kernel multiplies without dequantizing
        out = x @ dequantize(self.qweight, self.scale).T
        if self.bias is not None:
            out = out + self.bias
        return out


def nbytes(sd: dict[str, Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in sd.values())


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. the round-trip error is bounded by half a step, by construction --
    # rounding to the nearest of 255 levels cannot be off by more than half of
    # the gap between them. this is the one guarantee the whole scheme rests on
    w = torch.randn(64, 128)
    for dim in (-1, None):
        q, s = quantize(w, dim)
        err = (dequantize(q, s) - w).abs()
        assert (err <= s / 2 + 1e-9).all(), dim
        assert q.dtype == torch.int8

    # 2. every scale group spends its full range: the largest magnitude in the
    # group lands exactly on the rail. asserting <= QMAX instead would only be
    # testing the clamp, and a global == QMAX would pass on one lucky row
    q_c, _ = quantize(w, dim=-1)
    q_t, _ = quantize(w, dim=None)
    assert (q_c.abs().amax(dim=-1) == QMAX).all()  # EVERY row, not just the loudest
    assert q_t.abs().max() == QMAX

    # and it is only ever one rail -- whichever tail is longer
    hi, lo = q_c.amax(-1) == QMAX, q_c.amin(-1) == -QMAX
    assert (hi | lo).all()

    # 3. zero stays zero. symmetric quantization has no zero point, so an exact
    # 0 in must be an exact 0 out -- otherwise every masked or pruned weight
    # would drift
    z = torch.zeros(4, 8)
    z[0, 0] = 1.0
    qz, sz = quantize(z)
    assert (dequantize(qz, sz)[1:] == 0).all()  # all-zero rows survive, no nan
    assert dequantize(qz, sz)[0, 0] == 1.0

    # 4. it is a RELATIVE precision scheme: scale the weights by 1000 and the
    # integers are identical, the scale absorbs it, and the error scales too.
    # this is why quantization does not care about the units of a layer
    q_1x, s_1x = quantize(w)
    q_1000x, s_1000x = quantize(w * 1000)
    assert torch.equal(q_1000x, q_1x)
    assert torch.allclose(s_1000x, s_1x * 1000)

    # 5. per-channel vs per-tensor, on a matrix with one planted outlier row.
    # per-tensor lets that row set the scale for everything, so the other 63
    # rows lose most of their levels -- the entire argument for per-channel
    out = torch.randn(64, 128)
    out[0] *= 100
    q_c, s_c = quantize(out, dim=-1)
    q_t, s_t = quantize(out, dim=None)
    err_c = (dequantize(q_c, s_c) - out)[1:].abs().mean()
    err_t = (dequantize(q_t, s_t) - out)[1:].abs().mean()
    print(
        f"outlier matrix, error on the other rows: per-channel {err_c:.2e}  per-tensor {err_t:.2e}  ({err_t / err_c:.0f}x)"
    )
    assert err_t > 50 * err_c

    # ...and on a well-behaved matrix the gap nearly vanishes, which is the
    # honest other half: per-channel is insurance, not a free win everywhere
    q_c, s_c = quantize(w, dim=-1)
    q_t, s_t = quantize(w, dim=None)
    r = (
        (dequantize(q_t, s_t) - w).abs().mean()
        / (dequantize(q_c, s_c) - w).abs().mean()
    ).item()
    print(f"gaussian matrix, same comparison: {r:.2f}x")
    assert r < 2

    # 6. QuantizedLinear answers what Linear answers, to quantization error
    lin = Linear(128, 64)
    ql = QuantizedLinear(lin)
    x = torch.randn(32, 128)
    ref, got = lin(x), ql(x)
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"QuantizedLinear relative error: {rel:.2e}")
    assert rel < 1e-2
    assert got.shape == ref.shape

    # 7. 4x smaller at rest. not exactly 4x: the scales are fp32 and the bias is
    # left alone, both deliberate -- they are a rounding error of the total and
    # quantizing them is where accuracy actually goes
    big = Linear(1024, 1024)
    small = QuantizedLinear(big)
    a, b = nbytes(big.state_dict()), nbytes(small.state_dict())
    print(f"state dict: {a / 2**10:.0f} KB -> {b / 2**10:.0f} KB  ({a / b:.2f}x)")
    assert 3.9 < a / b < 4.0

    # 8. the buffers travel in the state dict, so a quantized model saves and
    # loads like any other. int8 stays int8 through the round trip
    sd = small.state_dict()
    assert sd["qweight"].dtype == torch.int8
    assert set(sd) == {"qweight", "scale", "bias"}
    QuantizedLinear(big).load_state_dict(sd)
    assert not list(small.parameters())  # nothing here is trainable

    # 9. bias=False carries through
    nb = QuantizedLinear(Linear(16, 8, bias=False))
    assert nb.bias is None and set(nb.state_dict()) == {"qweight", "scale"}
    assert nb(torch.randn(4, 16)).shape == (4, 8)

    print("ok")
