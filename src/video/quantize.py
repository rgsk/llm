"""int8 weight-only quantization: store w as small ints plus a float scale."""

import torch
from torch import Tensor

from embedding import Embedding
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
    """A Linear whose weight is stored int8. Built from a trained Linear, not trained.

    `qs` supplies an already-quantized weight, so a tied pair can share one.
    """

    def __init__(self, linear: Linear, dim: int | None = -1, qs=None):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        q, scale = qs if qs is not None else quantize(linear.weight.data, dim)
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


class QuantizedEmbedding(Module):
    """An Embedding whose table is stored int8. Rows are looked up, then dequantized.

    With dim=-1 the scale is per row, i.e. per token -- the same axis a tied
    lm_head needs for per-output-channel, which is why one quantization serves
    both. The cost is that the error now enters the residual stream at layer 0.
    """

    def __init__(self, embedding: Embedding, dim: int | None = -1, qs=None):
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        q, scale = qs if qs is not None else quantize(embedding.weight.data, dim)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)

    def forward(self, idx: Tensor) -> Tensor:
        scale = self.scale if self.scale.ndim == 0 else self.scale[idx]
        return dequantize(self.qweight[idx], scale)


def nbytes(sd: dict[str, Tensor]) -> int:
    """Bytes at rest. Keyed by storage, so a tied weight counts once, as on disk."""
    seen = {t.data_ptr(): t.numel() * t.element_size() for t in sd.values()}
    return sum(seen.values())


def quantize_model(
    model: Module, dim: int | None = -1, skip: tuple[str, ...] = ()
) -> Module:
    """Swap every Linear and Embedding for its int8 form, in place.

    `skip` matches dotted paths. Weights that were tied stay tied: they are
    quantized once and the buffers are shared, so the pair shrinks instead of
    silently unsharing into an fp32 copy plus an int8 one.
    """
    shared: dict[int, tuple[Tensor, Tensor]] = {}

    def qs_for(w: Tensor) -> tuple[Tensor, Tensor]:
        # not setdefault: that would evaluate quantize() before checking the key
        key = w.data_ptr()
        if key not in shared:
            shared[key] = quantize(w, dim)
        return shared[key]

    def walk(m: Module, prefix: str) -> None:
        for name, child in list(m.__dict__.items()):
            if not isinstance(child, Module):
                continue
            path = f"{prefix}{name}"
            if any(path == s or path.endswith(f".{s}") for s in skip):
                continue
            if isinstance(child, Linear):
                setattr(m, name, QuantizedLinear(child, dim, qs_for(child.weight.data)))
            elif isinstance(child, Embedding):
                setattr(
                    m, name, QuantizedEmbedding(child, dim, qs_for(child.weight.data))
                )
            else:
                walk(child, f"{path}.")

    walk(model, "")
    return model


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

    # 10. quantize_model reaches every Linear and nothing else. ResidualProj
    # subclasses Linear so it is caught too; Embedding and the norms are not,
    # which is the point -- this swaps a layer type, it does not walk tensors
    from dataclasses import asdict

    from gpt import GPT
    from gpt_config import GPTConfig

    def count(m: Module, cls: type) -> int:
        return isinstance(m, cls) + sum(count(c, cls) for c in m.children())

    def fresh() -> GPT:
        torch.manual_seed(0)
        return GPT(**asdict(cfg))

    cfg = GPTConfig(vocab_size=512, block_size=32, n_embed=64, n_head=4, n_layer=2)
    m = fresh()
    n_linear, n_embed = count(m, Linear), count(m, Embedding)
    assert n_linear > 0 and n_embed == 2  # token table and learned positions
    quantize_model(m)
    assert count(m, Linear) == 0 and count(m, QuantizedLinear) == n_linear
    assert count(m, Embedding) == 0 and count(m, QuantizedEmbedding) == n_embed
    assert type(m.ln_f).__name__ not in ("QuantizedLinear", "QuantizedEmbedding")

    # 11. the model still answers. KL against the fp32 logits is the metric that
    # matters -- it measures the same input's answer moving, not a loss average
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    m2 = fresh().eval()
    with torch.no_grad():
        ref = m2(x)
        got = quantize_model(m2)(x)
    lp_ref, lp_got = ref.log_softmax(-1), got.log_softmax(-1)
    kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1).mean().item()
    print(f"logit KL after quantizing every Linear: {kl:.2e} nats")
    assert kl < 1e-3

    # 12. the tie survives, and that is what makes the pair shrink. quantizing
    # ONE side of a tied weight is the trap: it unshares, leaving an fp32 table
    # plus a fresh int8 copy, so the pair gets BIGGER. quantizing both from one
    # shared buffer is the only version that actually saves anything
    def pair_bytes(model: Module) -> int:
        sd = model.state_dict()
        head = ("token_embedding_table", "lm_head")
        return nbytes({k: v for k, v in sd.items() if k.startswith(head)})

    base = fresh()
    assert base.lm_head.weight is base.token_embedding_table.weight  # tied in fp32
    both = quantize_model(fresh())
    head_only = quantize_model(fresh(), skip=("token_embedding_table",))
    neither = quantize_model(fresh(), skip=("token_embedding_table", "lm_head"))

    # still one storage under two names, exactly as the fp32 tie was. compared
    # by data_ptr, not `is`: register_buffer detaches, so each name holds its
    # own view object -- what is shared is the memory, which is the point
    tbl = both.token_embedding_table
    assert both.lm_head.qweight.data_ptr() == tbl.qweight.data_ptr()
    assert both.lm_head.scale.data_ptr() == tbl.scale.data_ptr()

    for name, model in (
        ("fp32", base),
        ("both", both),
        ("head only", head_only),
        ("neither", neither),
    ):
        sd = model.state_dict()
        print(
            f"  {name:<10} total {nbytes(sd) / 2**10:7.1f} KB"
            f"   embed+head {pair_bytes(model) / 2**10:6.1f} KB"
        )
    assert pair_bytes(both) < pair_bytes(base) / 3.5  # shared: a real 4x
    assert pair_bytes(head_only) > pair_bytes(base)  # unshared: strictly worse
    assert pair_bytes(neither) == pair_bytes(base)

    # --- the rest needs the trained checkpoint, and is skipped without it ---
    from paths import CKPT_DIR

    ckpt = CKPT_DIR / "big_2026-08-30_09-09-16.pt"
    if not ckpt.exists():
        print("no trained checkpoint; skipping the real-weight probes")
        print("ok")
        raise SystemExit

    from checkpoint import load_checkpoint
    from dataset import BinDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 13. the end-to-end check on trained weights. KL against the fp32 logits
    # separates the two schemes cleanly where bpc barely can: over the full val
    # split that is +0.0001 bpc for per-channel against +0.0005 for per-tensor,
    # a 27s sweep for the same ordering this gets in under a second. the spread
    # in the weights that explains it is quantize.ipynb, probe 2
    model, _ = load_checkpoint(ckpt, dev)
    val = BinDataset("val")
    bs = model.block_size
    x = torch.stack([val.tokens(i * bs, bs) for i in range(8)]).to(dev)
    val.close()

    with torch.no_grad():
        ref = model(x).float().log_softmax(-1)
    fp32_mb = nbytes(model.state_dict()) / 2**20

    kls = {}
    for label, d in (("per-channel", -1), ("per-tensor", None)):
        m, _ = load_checkpoint(ckpt, dev)
        quantize_model(m, dim=d)
        with torch.no_grad():
            lp = m(x).float().log_softmax(-1)
        kls[label] = (ref.exp() * (ref - lp)).sum(-1).mean().item()
        mb = nbytes(m.state_dict()) / 2**20
        print(f"  {label:<12} KL {kls[label]:.2e} nats   {fp32_mb:.1f} -> {mb:.1f} MB")

    # per-channel is 6x more accurate for under 1% more bytes: the scales are
    # one float per row against a whole matrix of int8
    assert kls["per-tensor"] > 3 * kls["per-channel"]
    assert kls["per-channel"] < 1e-3

    print("ok")
