"""LoRA: freeze the base, train W x + (alpha/r) * B A x beside it.

B is zero-init so the wrapped model starts bit-identical to the base; A is
random so B has a gradient. Only qkv and proj are wrapped -- ~0.7% of params.
The checkpoint is the adapter alone, plus the base path to rebuild it from.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from checkpoint import load_checkpoint
from gpt import GPT
from gpt_config import GPTConfig
from linear import Linear
from module import Module
from parameter import Parameter
from paths import CKPT_DIR

ATTN_TARGETS = ("qkv", "proj")  # ResidualProj subclasses Linear, so proj matches


class LoRALinear(Module):
    """A frozen Linear plus a trainable rank-r update."""

    def __init__(self, base: Linear, r: int, alpha: float):
        super().__init__()
        assert r > 0, f"r must be at least 1, got {r}"
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        w = base.weight  # apply_lora runs after .to(device); fresh tensors are cpu
        self.A = Parameter(
            torch.randn(r, base.in_features, device=w.device, dtype=w.dtype)
            / math.sqrt(base.in_features)
        )
        self.B = Parameter(
            torch.zeros(base.out_features, r, device=w.device, dtype=w.dtype)
        )
        self.r = r
        self.scale = alpha / r  # alpha/r keeps the lr independent of r

    def forward(self, x: Tensor) -> Tensor:
        # (x @ A.T) @ B.T: the [out, in] delta is never materialized
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def walk(module: Module) -> Iterator[Module]:
    """Every module under this one, self first -- OurModule has no .modules()."""
    yield module
    for _, child in module.named_children():
        yield from walk(child)


def apply_lora(
    model: Module,
    r: int,
    alpha: float | None = None,
    targets: tuple[str, ...] = ATTN_TARGETS,
) -> Module:
    """Freeze everything, then wrap each Linear named in targets. In place.

    LoRALinear freezes only the layer it wraps, so the blanket freeze is what
    keeps embeddings, norms and lm_head out of the gradient.
    """
    alpha = 2.0 * r if alpha is None else alpha
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = 0
    for m in list(walk(model)):  # list: the loop replaces what the walk returns
        for name in targets:
            child = getattr(m, name, None)
            if isinstance(child, Linear):
                setattr(m, name, LoRALinear(child, r, alpha))
                wrapped += 1
    assert wrapped, f"no unwrapped Linear named {targets} here"  # catches a 2nd call
    return model


def lora_parameters(model: Module) -> list[Parameter]:
    """After apply_lora, requires_grad is the adapter -- nothing else carries one."""
    return [p for p in model.parameters() if p.requires_grad]


def param_counts(model: Module) -> tuple[int, int]:
    """(trainable, total)."""
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in lora_parameters(model)), total


@torch.no_grad()
def merge_lora(model: Module) -> Module:
    """Fold each adapter into its base (W += scale * B @ A) and restore the
    plain Linear. Exact in the algebra, ~1e-2 apart under TF32."""
    for m in list(walk(model)):
        for name, child in list(m.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight += child.scale * child.B @ child.A
                setattr(m, name, child.base)
    return model


def lora_state_dict(model: Module) -> dict[str, Tensor]:
    """A and B only; the rest would be a copy of the base file."""
    return {n: p.data for n, p in model.named_parameters() if p.requires_grad}


def save_lora(
    path: Path,
    model: Module,
    cfg: GPTConfig,
    *,
    base_ckpt: Path,
    r: int,
    alpha: float,
    targets: tuple[str, ...] = ATTN_TARGETS,
    **meta,
) -> None:
    """save_checkpoint's signature, so sft.py can take either."""
    torch.save(
        {
            "model": lora_state_dict(model),
            "config": asdict(cfg),
            "lora": {
                # resolved: a relative path only reloads from the cwd it was fired in
                "base_ckpt": str(Path(base_ckpt).resolve()),
                "r": r,
                "alpha": alpha,
                "targets": list(targets),
            },
            **meta,
        },
        path,
    )


def load_adapter(model: Module, adapter: dict[str, Tensor]) -> Module:
    """Overlay A/B onto an already-wrapped model -- one base, many adapters.

    copy_, not load_state_dict: torch's needs strict=False and ours refuses a
    partial dict, so neither backend takes this as a state_dict.
    """
    own = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.endswith((".A", ".B")) for n in own), (
        "model has no adapters -- call apply_lora first"
    )
    assert own.keys() == adapter.keys(), sorted(own.keys() ^ adapter.keys())
    with torch.no_grad():
        for n, p in own.items():
            assert p.shape == adapter[n].shape, (
                f"{n}: have {tuple(p.shape)}, got {tuple(adapter[n].shape)}"
            )
            p.copy_(adapter[n])
    return model


def load_lora(path: Path, device: str = "cpu", **overrides) -> tuple[GPT, dict]:
    """Rebuild the base, wrap it as it was trained, overlay the adapter."""
    saved = torch.load(path, map_location=device)
    assert "lora" in saved, f"{Path(path).name} is not an adapter checkpoint"
    spec = saved["lora"]
    base = Path(spec["base_ckpt"])
    if not base.exists():  # trained elsewhere (a pod), same name in our CKPT_DIR
        base = CKPT_DIR / base.name
        assert base.exists(), f"base {spec['base_ckpt']} not here, nor {base}"
    model, _ = load_checkpoint(base, device, **overrides)
    apply_lora(model, spec["r"], spec["alpha"], tuple(spec["targets"]))
    load_adapter(model, saved["model"])
    return model, {k: v for k, v in saved.items() if k != "model"}


if __name__ == "__main__":
    # assertions are in test_lora.py; this prints what the wrapping costs
    torch.manual_seed(0)
    cfg = GPTConfig(  # tinystories_cfg, spelled out so this needs no dataset
        vocab_size=4097,
        block_size=512,
        n_embed=512,
        n_head=8,
        n_layer=8,
        attention="flex",
        norm="rms",
        ffn="gated",
        position="rope",
    )

    print(f"{'r':>3} {'alpha':>6} {'trainable':>11} {'of total':>9}  adapter")
    for r in (1, 2, 4, 8, 16, 32):
        n, total = param_counts(apply_lora(GPT(**asdict(cfg)), r))
        print(
            f"{r:>3} {2 * r:>6} {n:>11,} {n / total:>8.2%}  {n * 4 / 2**10:>5.0f} KiB"
        )

    model = GPT(**asdict(cfg))
    x = torch.randint(0, 4097, (1, 16))
    before = model(x)
    apply_lora(model, 8)

    # walk(), not named_modules(): OurModule has no such method
    print("\nwrapped, one block:")
    for m in walk(model.blocks[0]):
        for name, c in m.named_children():
            if isinstance(c, LoRALinear):
                print(
                    f"  {name:<6} [{c.base.out_features:>4}, {c.base.in_features:>4}]"
                    f"  A{tuple(c.A.shape)} B{tuple(c.B.shape)} scale {c.scale:g}"
                )

    print(f"\nB zero-init, max |wrapped - base| = {(model(x) - before).abs().max():g}")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    all_ab = all(n.endswith((".A", ".B")) for n in names)
    print(f"{len(names)} trainable tensors, all A/B: {all_ab}")
    print(f"embedding frozen: {not model.token_embedding_table.weight.requires_grad}")
