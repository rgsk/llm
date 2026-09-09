"""Ship the quantized model: a file that is small on disk, and a load that never
builds the fp32 model it came from.

`quantize.py` makes a small model out of a big one, which means the big one has
to exist first -- 213 MB of peak memory to end up holding 26. That is fine for
measuring and useless for serving. What is missing is the other half: write the
int8 weights to their own checkpoint, then rebuild the model around them
directly, so fp32 is paid once offline and never again.
"""

import contextlib
from collections.abc import Generator
from dataclasses import asdict
from pathlib import Path

import torch
from checkpoint import load_checkpoint
from embedding import Embedding
from gpt import GPT
from gpt_config import GPTConfig
from linear import Linear
from module import Module
from parameter import Parameter
from quantize import QuantizedEmbedding, QuantizedLinear, quantize_model
from torch import Tensor


def parent_of(model: Module, path: str) -> tuple[Module, str]:
    """The module holding `path`, and the attribute name under it. "blocks.0.attn"
    works because ModuleList stores its children as attributes "0", "1", ..."""
    *parents, name = path.split(".")
    m = model
    for p in parents:
        m = getattr(m, p)
    return m, name


def quant_plan(model: Module) -> dict[str, int]:
    """Dotted path -> bits, for every quantized layer.

    The state dict alone cannot say this. `qweight` is uint8 at every sub-byte
    width, so the file has no way to tell int4 from int6, and a layer that was
    skipped looks like a layer that was never there.
    """
    plan: dict[str, int] = {}

    def walk(m: Module, prefix: str) -> None:
        for name, child in m.__dict__.items():
            if not isinstance(child, Module):
                continue
            path = f"{prefix}{name}"
            if isinstance(child, QuantizedLinear | QuantizedEmbedding):
                plan[path] = child.bits
            else:
                walk(child, f"{path}.")

    walk(model, "")
    return plan


def save_quantized(path: Path, model: Module, cfg: GPTConfig, **meta) -> None:
    """The quantized weights, the architecture, and which layers were quantized.

    torch.save writes each STORAGE once, so a tied pair costs one copy on disk --
    the same accounting `nbytes` does in memory.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "quant": quant_plan(model),
            **meta,
        },
        path,
    )


@contextlib.contextmanager
def empty_weights() -> Generator[None]:
    """Inside this block, Linear and Embedding allocate on `meta`: shapes, no bytes.

    Only those two, deliberately. Their weights are the ones the checkpoint is
    about to replace, so allocating them is pure waste. Everything else -- norm
    weights, rope tables, causal masks -- is built for real, because a
    `persistent=False` buffer is by design not in the file and nothing else will
    rebuild it.

    Patching the class is how accelerate's `init_empty_weights` does it too; the
    alternative, `with torch.device("meta")`, catches every factory call in
    `__init__` including those buffers.
    """
    originals = {cls: cls.__init__ for cls in (Linear, Embedding)}

    def on_meta(init):
        def wrapper(self, *args, **kwargs):
            with torch.device("meta"):
                init(self, *args, **kwargs)

        return wrapper

    for cls, init in originals.items():
        cls.__init__ = on_meta(init)
    try:
        yield
    finally:
        for cls, init in originals.items():
            cls.__init__ = init


def assign_state_dict(model: Module, sd: dict[str, Tensor], device: str) -> None:
    """Point every parameter and buffer AT the file's tensor, instead of copying into it.

    `load_state_dict` copies, which needs the destination already allocated at
    full size -- exactly what a meta skeleton does not have. It also breaks
    weight tying: one storage in the file becomes two after two `copy_` calls.
    Assigning keeps whatever the file shared, as long as the move to `device`
    happens once per storage, which is what `moved` is for.

    It rebinds the attribute rather than writing `holder.data`: `set_data`
    refuses a meta variable, and rebinding is also what makes a tied pair one
    object again instead of two views.
    """
    own = dict(model.named_parameters(remove_duplicate=False))
    own.update(
        {
            n: b
            for n, b in model.named_buffers(
                remove_duplicate=False, persistent_only=True
            )
        }
    )
    assert not (missing := own.keys() - sd.keys()), f"missing: {sorted(missing)}"
    assert not (extra := sd.keys() - own.keys()), f"unexpected: {sorted(extra)}"

    moved: dict[tuple[int, torch.Size, bool], Tensor] = {}
    for name, holder in own.items():
        t = sd[name]
        assert holder.shape == t.shape, (
            f"{name}: skeleton wants {tuple(holder.shape)}, file has {tuple(t.shape)}"
        )
        # keyed by (storage, shape): two names over one storage are the tie, two
        # different views of one storage would not be interchangeable
        trainable = isinstance(holder, Parameter)
        key = (t.data_ptr(), t.shape, trainable)
        if key not in moved:
            on_device = t.to(device)
            moved[key] = Parameter(on_device) if trainable else on_device
        setattr(*parent_of(model, name), moved[key])


def load_quantized(path: Path, device: str = "cpu") -> tuple[GPT, dict]:
    """Rebuild the quantized model without ever materialising the fp32 one.

    mmap=True leaves the tensors on disk; each is read exactly once, when
    `assign_state_dict` moves it to `device`. With device="cpu" they are never
    read at all -- the weights stay memory-mapped, so they cost page cache that
    the kernel can drop, not resident memory.
    """
    saved = torch.load(path, map_location="cpu", mmap=True)
    assert "quant" in saved, f"{path} is not a quantized checkpoint"
    cfg = GPTConfig.from_dict(saved["config"])
    sd = saved["model"]

    with empty_weights():
        model = GPT(**asdict(cfg))

    for path_, bits in saved["quant"].items():
        parent, name = parent_of(model, path_)
        child = getattr(parent, name)
        cls = QuantizedLinear if isinstance(child, Linear) else QuantizedEmbedding
        # the meta layer is here for its shapes; qs= is what stops the
        # constructor from trying to quantize a weight that holds no data
        qs = (sd[f"{path_}.qweight"], sd[f"{path_}.scale"])
        setattr(parent, name, cls(child, bits=bits, qs=qs))

    assign_state_dict(model, sd, device)
    # non-persistent buffers were built for real, on whatever the default device
    # was; this moves them, and is a no-op for everything assign already placed
    model.to(device)
    return model.eval(), {k: v for k, v in saved.items() if k != "model"}


def quantize_checkpoint(
    src: Path,
    dst: Path,
    bits: int = 8,
    dim: int | None = -1,
    skip: tuple[str, ...] = (),
    override: dict[str, int] | None = None,
    device: str = "cpu",
    **cfg_overrides,
) -> Path:
    """fp32 checkpoint in, quantized checkpoint out. The conversion is offline and
    happens once; every load after it is the cheap one."""
    model, meta = load_checkpoint(src, device, **cfg_overrides)
    cfg = GPTConfig.from_dict(meta.pop("config"))
    quantize_model(model, dim=dim, bits=bits, skip=skip, override=override)
    save_quantized(dst, model, cfg, **meta)
    return dst


if __name__ == "__main__":
    import tempfile

    from checkpoint import save_checkpoint
    from generate import generate
    from quantize import nbytes, quantizable

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=32, n_embed=64, n_head=4, n_layer=2)

    def fresh() -> GPT:
        torch.manual_seed(0)
        return GPT(**asdict(cfg)).eval()

    def get(model: Module, path: str) -> Module:
        return getattr(*parent_of(model, path))

    x = torch.randint(0, cfg.vocab_size, (2, 16))

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # 1. the skeleton has every shape and almost none of the bytes. what goes
        # to meta is exactly what the file will replace; a norm weight is built
        # for real because it is small, and a persistent=False buffer because
        # nothing in the checkpoint could rebuild it
        with empty_weights():
            skel = GPT(**asdict(cfg))
        assert all(get(skel, p).weight.is_meta for p in quantizable(skel))
        real = [(n, p) for n, p in skel.named_parameters() if not p.is_meta]
        assert real and all("ln" in n or "norm" in n for n, _ in real), real
        held = sum(p.numel() * p.element_size() for _, p in real)
        assert held < 0.01 * nbytes(fresh().state_dict())

        # ...and the patch comes back off, including when the block raises
        assert not Linear(4, 4).weight.is_meta
        with contextlib.suppress(RuntimeError), empty_weights():
            raise RuntimeError("boom")
        assert not Embedding(4, 4).weight.is_meta

        # 2. THE test: a model rebuilt from the small file computes bit-for-bit
        # what the quantized model in memory computed. not allclose -- the int8
        # values went to disk and came back, so anything but == 0 is a bug in the
        # round trip rather than quantization error, which was already paid
        m = quantize_model(fresh())
        with torch.no_grad():
            want = m(x)
        path = d / "q8.pt"
        save_quantized(path, m, cfg, step=7, val_loss=1.5)
        loaded, meta = load_quantized(path)
        with torch.no_grad():
            assert (loaded(x) - want).abs().max() == 0

        # 3. the file holds the quantized model and not the fp32 one, and the
        # tied pair is written ONCE. the sum over state_dict KEYS double counts
        # that pair, `nbytes` counts storages, and the difference between them is
        # exactly one copy of it
        fp32_path = d / "fp32.pt"
        save_checkpoint(fp32_path, fresh(), cfg)
        by_key = sum(t.numel() * t.element_size() for t in m.state_dict().values())
        by_storage = nbytes(m.state_dict())
        head = m.lm_head
        dup = (
            head.qweight.numel() * head.qweight.element_size() + head.scale.numel() * 4
        )
        assert by_key - by_storage == dup

        # and torch.save does the same accounting. measured by saving the same
        # tensors with the pair unshared: the file grows by what the tie saved.
        # (the toy model is too small to read this off the total -- zip record
        # overhead is 16% of it, which is why the claim is a difference)
        unshared = dict(m.state_dict())
        for k in ("lm_head.qweight", "lm_head.scale"):
            unshared[k] = unshared[k].clone()
        torch.save(m.state_dict(), tied_path := d / "tied.pt")
        torch.save(unshared, untied_path := d / "untied.pt")
        assert untied_path.stat().st_size - tied_path.stat().st_size > 0.9 * dup
        print(
            f"toy model: fp32 {fp32_path.stat().st_size / 2**10:.0f} KB -> "
            f"int8 {path.stat().st_size / 2**10:.0f} KB "
            f"({fp32_path.stat().st_size / path.stat().st_size:.2f}x)"
        )

        # 4. the tie survives the round trip -- one storage, two names, as it was
        # in fp32 and as it is on disk. this is what assignment buys: a copy into
        # two separately allocated buffers would unshare them
        assert (
            loaded.lm_head.qweight.data_ptr()
            == loaded.token_embedding_table.qweight.data_ptr()
        )
        assert nbytes(loaded.state_dict()) == by_storage

        # and the cost of getting it wrong, made concrete: an unshared pair is a
        # model in memory that is BIGGER than the file it was loaded from
        broken, _ = load_quantized(path)
        broken.lm_head.qweight = broken.lm_head.qweight.clone()  # what a copy does
        qbytes = head.qweight.numel() * head.qweight.element_size()
        assert nbytes(broken.state_dict()) == by_storage + qbytes

        # 5. the check a user would actually make: same file, same text. 32
        # greedy steps, so any drift anywhere shows up as a different token
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        assert torch.equal(
            generate(m, prompt, 32, temperature=0.0),
            generate(load_quantized(path)[0], prompt, 32, temperature=0.0),
        )

        # 6. metadata and config ride along, without the weights following them
        assert meta["step"] == 7 and meta["val_loss"] == 1.5
        assert GPTConfig.from_dict(meta["config"]) == cfg
        assert "model" not in meta

        # 7. mixed precision survives, and this is why the plan is in the file:
        # every qweight here is uint8 whatever the width, so without it a 4-bit
        # layer would be unpacked as 8-bit -- half the weights, silently
        mixed = quantize_model(fresh(), bits=4, override={"blocks.0.attn.proj": 8})
        with torch.no_grad():
            want_mixed = mixed(x)
        mixed_path = d / "mixed.pt"
        save_quantized(mixed_path, mixed, cfg)
        back, _ = load_quantized(mixed_path)
        assert quant_plan(back) == quant_plan(mixed)
        assert back.blocks[0].attn.proj.bits == 8 and back.blocks[1].attn.proj.bits == 4
        with torch.no_grad():
            assert (back(x) - want_mixed).abs().max() == 0

        # 8. a partly quantized model round trips too. the skipped layers are not
        # in the plan, so they rebuild as ordinary fp32 Linear/Embedding and get
        # their weights from the same file
        part = quantize_model(fresh(), skip=("token_embedding_table", "lm_head"))
        with torch.no_grad():
            want_part = part(x)
        part_path = d / "part.pt"
        save_quantized(part_path, part, cfg)
        back, _ = load_quantized(part_path)
        plan = quant_plan(back)
        assert "lm_head" not in plan and "token_embedding_table" not in plan
        assert "blocks.0.attn.heads.0.query" in plan  # nested under a ModuleList
        assert type(back.lm_head) is Linear and not back.lm_head.weight.is_meta
        assert back.lm_head.weight is back.token_embedding_table.weight  # still tied
        with torch.no_grad():
            assert (back(x) - want_part).abs().max() == 0

        # 9. a state dict that does not fit the skeleton is refused by name, the
        # way load_state_dict refuses one -- assignment would otherwise accept
        # any shape at all and fail much later, inside a matmul
        try:
            assign_state_dict(quantize_model(fresh()), {}, "cpu")
            raise SystemExit("should have refused an empty state dict")
        except AssertionError as e:
            assert "missing" in str(e)

        # 10. and an fp32 checkpoint is not silently accepted as a quantized one
        try:
            load_quantized(fp32_path)
            raise SystemExit("should have refused an fp32 checkpoint")
        except AssertionError as e:
            assert "not a quantized checkpoint" in str(e)

    # --- the rest needs the trained checkpoint, and is skipped without it ---
    from paths import CKPT_DIR

    ckpt = CKPT_DIR / "big_2026-08-30_09-09-16.pt"
    if not ckpt.exists():
        print("no trained checkpoint; skipping the real-weight probes")
        print("ok")
        raise SystemExit

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # converted on the device that will serve it, which is not cosmetic: test 13
        q8 = quantize_checkpoint(ckpt, tmp / "big_int8.pt", bits=8, device=dev)
        fp32_mb = ckpt.stat().st_size / 2**20
        int8_mb = q8.stat().st_size / 2**20
        print(
            f"\n27M checkpoint: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({fp32_mb / int8_mb:.2f}x)"
        )

        # 11. the load path, which is the whole point. quantize_model has to build
        # the fp32 model before it can shrink it, so its peak is fp32 + int8 and
        # no machine that cannot hold the big model can run the small one
        def peak(fn):
            if dev == "cpu":
                return fn(), float("nan")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out = fn()
            torch.cuda.synchronize()
            return out, torch.cuda.max_memory_allocated() / 2**20

        def convert():
            model, _ = load_checkpoint(ckpt, dev)
            return quantize_model(model)

        via_fp32, peak_convert = peak(convert)
        resident = nbytes(via_fp32.state_dict()) / 2**20
        del via_fp32
        direct, peak_direct = peak(lambda: load_quantized(q8, dev)[0])

        print(
            f"  load fp32 + quantize   peak {peak_convert:6.1f} MB   from a {fp32_mb:.0f} MB file"
        )
        print(
            f"  load_quantized         peak {peak_direct:6.1f} MB   from a {int8_mb:.0f} MB file"
        )
        print(f"  resident either way    {resident:6.1f} MB")
        if dev == "cuda":
            # the peak is the model and nothing else: no fp32 copy is ever built,
            # and mmap means no second host-side copy of the file either
            assert peak_direct < 1.1 * resident
            assert peak_convert > 5 * peak_direct

        # 12. and it is the same model. identical logits, then identical text --
        # this is the claim the whole file exists to make
        model_fp32, _ = load_checkpoint(ckpt, dev)
        reference = quantize_model(model_fp32)
        x = torch.randint(0, 4096, (2, 64), device=dev)
        with torch.no_grad():
            assert (direct(x) - reference(x)).abs().max() == 0

        prompt = torch.zeros(1, 1, dtype=torch.long, device=dev)
        gen = {"max_new_tokens": 64, "top_k": 40, "temperature": 0.8}
        a = generate(
            reference, prompt, generator=torch.Generator(dev).manual_seed(0), **gen
        )
        b = generate(
            direct, prompt, generator=torch.Generator(dev).manual_seed(0), **gen
        )
        assert torch.equal(a, b)

        # 13. the one place that bit-for-bit claim stops holding: converting on
        # cpu and serving on cuda. `amax` agrees exactly -- a max is order-free --
        # but `amax / 127` lands one ulp apart on ~5% of rows, and a weight
        # sitting on a rounding boundary flips a level because of it. the two
        # models are the same to any measure that matters and are NOT the same
        # tensor, so convert on the device you serve on if you want == 0
        cpu_q8 = quantize_checkpoint(
            ckpt, tmp / "big_int8_cpu.pt", bits=8, device="cpu"
        )
        elsewhere, _ = load_quantized(cpu_q8, dev)
        here, there = direct.state_dict(), elsewhere.state_dict()
        qkeys = [k for k in here if k.endswith("qweight")]
        flipped = sum((here[k] != there[k]).count_nonzero().item() for k in qkeys)
        total = sum(here[k].numel() for k in qkeys)
        with torch.no_grad():
            drift = (elsewhere(x) - direct(x)).abs().max().item()
        print(
            f"  cpu-converted, cuda-served: {flipped} of {total} int8 values differ, "
            f"max logit drift {drift:.2e}"
        )
        assert flipped < 1e-4 * total and drift < 1e-2

        import sys

        from paths import ROOT

        sys.path.append(str(ROOT / "src"))
        from tokenizer import BPETokenizer

        tok = BPETokenizer.load(
            str(ROOT / "artifacts" / "tokenizer" / "bpe_ts_4096.json")
        )
        print(
            f"\nfrom the {int8_mb:.0f} MB file, fp32 never built:\n{tok.decode(b[0].tolist())!r}"
        )

    print("\nok")
