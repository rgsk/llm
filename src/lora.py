from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from sft import SFTConfig


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update:
    base(x) + (alpha/r) * B(A(x)).

    B starts at zero so the wrapped layer is bit-identical to the base at init;
    A starts random so B has a nonzero gradient from step 0 (zero/zero would
    leave both gradients zero forever). Computed as (x @ A.T) @ B.T -- the
    [.., r] intermediate -- never materializing the [out, in] delta.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        # Born on the base layer's device/dtype: apply_lora runs after the
        # model has moved to cuda, and fresh tensors default to cpu.
        w = base.weight
        self.A = nn.Parameter(
            torch.randn(r, base.in_features, device=w.device, dtype=w.dtype)
            / math.sqrt(base.in_features)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, r, device=w.device, dtype=w.dtype)
        )
        # alpha/r keeps the update's scale independent of r, so sweeping r
        # does not require re-tuning the LR.
        self.scale = alpha / r

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b t i"]) -> Float[Tensor, "b t o"]:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


ATTN_TARGETS = ("qkv", "proj")


def apply_lora(
    model: nn.Module, r: int, alpha: float, targets: tuple[str, ...] = ATTN_TARGETS
) -> nn.Module:
    """Freeze the whole model, then wrap every nn.Linear whose attribute name is
    in `targets` with a trainable LoRALinear. In-place; returns the model.

    The blanket freeze is the half that is easy to forget: LoRALinear only
    freezes the layer it wraps, and embeddings, norms, and lm_head would
    otherwise still be trainable -- silently turning "LoRA" into "LoRA plus
    full finetuning of everything unwrapped".
    """
    model.requires_grad_(False)
    for module in model.modules():
        for name in targets:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    return model


def test():
    """Per-step sanity check; grows/changes with each step of the build."""
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    from main import GPT, small_cfg

    torch.manual_seed(0)
    model = GPT(small_cfg).cuda()
    with tempfile.TemporaryDirectory() as d:
        base_path = Path(d) / "base.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(small_cfg),
                "step": 0,
                "val_loss": 0.0,
            },
            base_path,
        )

        cfg = LoRAConfig(base_ckpt=str(base_path), r=4, alpha=8.0)
        apply_lora(model, cfg.r, cfg.alpha, cfg.targets)
        with torch.no_grad():  # fake training so the adapters are not a no-op
            for n, p in model.named_parameters():
                if n.endswith(".B"):
                    p.add_(torch.randn_like(p))

        # the exact save train() now does for a partially frozen model
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        adapter_sd = {k: v for k, v in model.state_dict().items() if k in trainable}
        lora_path = Path(d) / "lora.pt"
        torch.save(
            {
                "model": adapter_sd,
                "config": asdict(small_cfg),
                "sft_config": asdict(cfg),
                "step": 7,
                "val_loss": 1.0,
                "metrics": {"val_comp": 1.0, "val_prompt": 4.0, "val_all": 2.0},
            },
            lora_path,
        )
        reloaded = load_lora_ckpt(lora_path)

        full = sum(v.numel() * v.element_size() for v in model.state_dict().values())
        slim = sum(v.numel() * v.element_size() for v in adapter_sd.values())
        x = torch.randint(0, small_cfg.vocab_size, (2, 8), device="cuda")
        assert torch.equal(model(x)[0], reloaded(x)[0]), "round-trip changed outputs"
        print(f"ok: adapter-only round-trip exact; ckpt {slim:,} B vs full {full:,} B")


def load_lora_ckpt(name: str | Path) -> nn.Module:
    """Load an adapter checkpoint written by train() on a wrapped model: rebuild
    the base model from cfg.base_ckpt, wrap it with the saved r/alpha/targets,
    and overlay the saved adapter tensors. Also accepts older full-state-dict
    lora ckpts -- the overlay is just bigger."""
    from main import CKPT_DIR, device
    from sft import load_ckpt

    path = name if isinstance(name, Path) else CKPT_DIR / name
    saved = torch.load(path, map_location=device)
    lcfg = saved["sft_config"]
    model = load_ckpt(lcfg["base_ckpt"])
    apply_lora(model, lcfg["r"], lcfg["alpha"], tuple(lcfg["targets"]))
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    assert not unexpected, unexpected
    # every adapter tensor must come from the file, never from init
    adapter_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    assert adapter_keys <= set(saved["model"]), adapter_keys - set(saved["model"])
    m = saved["metrics"]
    print(
        f"loaded {path.name}: step {saved['step']}, r={lcfg['r']}, "
        f"val_comp {m['val_comp']:.4f}"
    )
    return model


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight (W += scale * B @ A) and put the
    plain nn.Linear back -- zero inference overhead, and the result is a plain
    GPT again. In-place; returns the model.

    Not bit-exact vs the wrapped forward: x@(W + D).T in one matmul rounds
    differently than x@W.T + (x@A.T)@B.T. Exact in the algebra; in fp32 the
    gap is ~1e-6, but under TF32 (main.py sets matmul precision "high") it is
    ~1e-2 -- TF32's 10-bit mantissa, not the merge. Behaviour (greedy argmax,
    accuracy) survives either way; measured 100% reversal after merging."""
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight += child.scale * child.B @ child.A
                setattr(module, name, child.base)
    return model


def load_adapter(model: nn.Module, name: str | Path) -> nn.Module:
    """Hot-swap: overlay a saved adapter's A/B tensors onto an already-wrapped,
    already-loaded model. No base rebuild, so switching behaviours is
    milliseconds -- the "one base, many adapters" serving pattern. The saved
    adapter must have been trained with the same r/targets (shape mismatches
    raise); works with both adapter-only and old full-state-dict ckpts."""
    from main import CKPT_DIR, device

    path = name if isinstance(name, Path) else CKPT_DIR / name
    saved = torch.load(path, map_location=device)
    adapter_sd = {k: v for k, v in saved["model"].items() if k.endswith((".A", ".B"))}
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(adapter_sd) == expected, (
        set(adapter_sd) ^ expected or "model has no adapters -- apply_lora first"
    )
    model.load_state_dict(adapter_sd, strict=False)
    print(f"adapter <- {path.name}")
    return model


@dataclass(frozen=True, kw_only=True)
class LoRAConfig(SFTConfig):
    """SFTConfig plus the adapter knobs. Inheriting means train() and wandb
    logging need no changes -- asdict() picks up r/alpha/targets for free.

    lr is ~5x the full-SFT 2e-4: the adapters are 1.4% of the params starting
    from zero effective update, and the frozen base bounds how much damage a
    hot LR can do. The convention is up to 10x; tune against the A/B."""

    name: str = "lora"
    lr: float = 1e-3
    min_lr: float = 1e-4
    r: int = 8
    alpha: float = 16.0  # 2*r, the usual pairing
    targets: tuple[str, ...] = ATTN_TARGETS


def main():
    """Experiment 1: the A/B against full SFT -- same data, same budget, same
    seed as sft.py's sanity run; only the trainable set differs."""
    from main import CKPT_DIR, latest_ckpt, timestamp
    from sft import held_out, load_ckpt, print_sample, report_samples, train
    from sft_data import build_or_load

    run_training = 0
    run_report_samples = 0

    sanity = 1
    cfg = LoRAConfig(base_ckpt="big_2026-08-16_06-45-06.pt", seed=90)
    if sanity:
        cfg = replace(cfg, max_steps=2000, warmup_steps=200, eval_interval=200)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    lora_ckpt = None
    samples = held_out()
    if run_training:
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        report_samples(model, samples)

        apply_lora(model, cfg.r, cfg.alpha, cfg.targets)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        packed = build_or_load()

        ckpt_path = CKPT_DIR / f"{cfg.name}_{timestamp()}.pt"
        print(
            f"\n=== lora r={cfg.r} alpha={cfg.alpha:g} ({n_train:,} trainable), "
            f"{cfg.max_steps} steps @ lr {cfg.lr:.1e} -> {ckpt_path.name} ==="
        )
        train(model, packed, cfg, ckpt_path)

        print("\n=== after finetuning (best ckpt) ===")
        best = load_lora_ckpt(latest_ckpt(cfg.name))
        report_samples(best, samples)
    elif run_report_samples:
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        report_samples(model, samples)
        lora_model = load_lora_ckpt(lora_ckpt or latest_ckpt(cfg.name))
        print("\n=== finetuned model ===")
        report_samples(lora_model, samples)
    else:
        idx = 9
        model = load_ckpt(cfg.base_ckpt)
        print("\n=== base model, before finetuning ===")
        print_sample(model, samples, idx=idx)
        lora_model = load_lora_ckpt(lora_ckpt or latest_ckpt(cfg.name))
        print("\n=== finetuned model ===")
        print_sample(lora_model, samples, idx=idx)


def main_rev():
    """Reverse-string adapter on the same frozen base. The instruct adapter
    tested format adaptation -- the change LoRA's low-rank bet is made for.
    Reversal is a new capability the base never needed (a position reflection,
    2l-1-t, that no single attention offset expresses), so this asks whether
    that also fits in rank r. Old sft.ipynb result to beat: a 32k-param model
    hit 1.0 in-range and 0.0 on held-out lengths 9-10.

    The 3-8 run generalized down (84.5% on 1-2) but not up (3% on 9-10), where
    a learned "answers end by 8" prior caps output length. Training on 5-10 and
    testing 1-4 separates the two stories: below-range lengths sit outside the
    training set but inside the length prior, so if 1-4 works while 11-12
    fails, the mechanism interpolates and only the stopping rule fails to
    extrapolate."""
    from main import CKPT_DIR, latest_ckpt, timestamp
    from rev_data import build_reverse, evaluate_reverse
    from sft import load_ckpt, train

    run_training = 1
    # a checkpoint filename to continue training from (opt state + step are
    # restored; the cosine is reshaped over this cfg's max_steps), or None
    # for a fresh adapter. Resumes get a "r" name suffix so the source ckpt
    # is not overwritten and the lineage stays readable on disk.
    resume = "rev110w_2026-08-19_15-09-54.pt"
    train_lmin, train_lmax = 1, 10
    # 2x sampling weight on the lengths that lagged under uniform 1-10
    # (l=9: 90%, l=10: 85%, everything else 96-100%).
    weights = [1.0] * 8 + [2.0, 2.0]
    cfg = LoRAConfig(
        base_ckpt="big_2026-08-16_06-45-06.pt",
        name="rev110w" + ("r" if resume else ""),
        seed=90,
        max_steps=4000 if resume else 2000,
        warmup_steps=200,
        eval_interval=200,
    )
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if run_training:
        if resume:
            model = load_lora_ckpt(resume)
        else:
            model = load_ckpt(cfg.base_ckpt)
            apply_lora(model, cfg.r, cfg.alpha, cfg.targets)
        packed = build_reverse(lmin=train_lmin, lmax=train_lmax, weights=weights)
        ckpt_path = CKPT_DIR / f"{cfg.name}_{timestamp()}.pt"
        print(
            f"\n=== rev adapter r={cfg.r} alpha={cfg.alpha:g}, l={train_lmin}-"
            f"{train_lmax}, {cfg.max_steps} steps @ lr {cfg.lr:.1e} "
            f"-> {ckpt_path.name} ==="
        )
        train(
            model,
            packed,
            cfg,
            ckpt_path,
            resume_from=CKPT_DIR / resume if resume else None,
        )

    best = load_lora_ckpt(latest_ckpt(cfg.name))
    # per-length: the instrument that separated dilution from capacity
    for l in range(1, train_lmax + 5):
        acc, samples = evaluate_reverse(best, n=100, lmin=l, lmax=l, slack=10)
        tag = "in   " if train_lmin <= l <= train_lmax else "ABOVE"
        print(f"l={l:>2} [{tag}] acc {acc:5.1%}   e.g. {samples[0]}")


def build_joint(rev_frac_starts: float = 0.33) -> dict:
    """Concatenate the instruct stream and a reverse stream into one packed
    dict. Windows open at example starts, so the mix ratio is controlled by
    start counts: rev examples are generated until they are rev_frac_starts of
    all starts. Concatenated starts stay sorted, which _windows' binary search
    requires."""
    import numpy as np

    from rev_data import build_reverse
    from sft_data import build_or_load

    sft_p = build_or_load()
    n_sft = len(sft_p["train_starts"])
    n_rev = int(n_sft * rev_frac_starts / (1 - rev_frac_starts))
    rev_p = build_reverse(
        n_train=n_rev, n_val=5_000, lmin=1, lmax=10, weights=[1.0] * 8 + [2.0, 2.0]
    )
    packed = {}
    for split in ("train", "val"):
        off = len(sft_p[f"{split}_ids"])
        for k in ("ids", "mask"):
            packed[f"{split}_{k}"] = np.concatenate(
                [sft_p[f"{split}_{k}"], rev_p[f"{split}_{k}"]]
            )
        packed[f"{split}_starts"] = np.concatenate(
            [sft_p[f"{split}_starts"], rev_p[f"{split}_starts"] + off]
        )
    return packed


def main_joint():
    """The stacking table's denominator: ONE rank-8 adapter trained on the
    mixed instruct+reverse stream. If this holds ~1.11 story / ~100% reverse,
    capacity was never the problem and post-hoc merging failed only because
    the deltas never met during training; if it also trades off, rank 8
    cannot hold both skills at once."""
    from main import CKPT_DIR, latest_ckpt, timestamp
    from rev_data import evaluate_reverse
    from sft import held_out, load_ckpt, print_sample, train

    run_training = 0
    cfg = LoRAConfig(
        base_ckpt="big_2026-08-16_06-45-06.pt",
        name="joint",
        seed=90,
        # 3000 steps at a 67/33 window mix: ~2000 instruct-equivalent steps
        # (the budget that reached 1.106) and ~1000 rev-equivalent (well past
        # the ~600-step phase transition).
        max_steps=3000,
        warmup_steps=300,
        eval_interval=300,
    )
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if run_training:
        model = load_ckpt(cfg.base_ckpt)
        apply_lora(model, cfg.r, cfg.alpha, cfg.targets)
        packed = build_joint()
        ckpt_path = CKPT_DIR / f"{cfg.name}_{timestamp()}.pt"
        print(
            f"\n=== joint adapter r={cfg.r}, {cfg.max_steps} steps "
            f"@ lr {cfg.lr:.1e} -> {ckpt_path.name} ==="
        )
        train(model, packed, cfg, ckpt_path)
        print("done -- run main_stack() for the full table incl. the joint row")
    else:
        joint_model = load_lora_ckpt(latest_ckpt(cfg.name))
        print("\n=== finetuned model ===")
        samples = held_out()
        print("\n=== instruct task ===")
        print_sample(joint_model, samples, idx=6)
        print("\n=== reverse word task ===")

        acc, samples = evaluate_reverse(joint_model, n=100, lmin=1, lmax=10)
        print(f"acc {acc:5.1%}   e.g. {samples}")


def main_stack():
    """The 2x2 stacking table: two specialists over one base, evaluated on both
    skills. Off-diagonal cells measure forgetting (what each adapter did to the
    skill it wasn't trained on); the +both row asks whether two independently
    trained deltas compose without ever having met during training. +both is
    rev merged into the weights, instruct hot-loaded on top -- both deltas
    active on every forward pass; any routing is implicit in the inputs."""
    import random

    import numpy as np

    from main import get_batch
    from rev_data import evaluate_reverse
    from sft import estimate_loss, load_ckpt
    from sft_data import build_or_load

    base_ckpt = "big_2026-08-16_06-45-06.pt"
    instruct = "lora_2026-08-19_01-55-50.pt"
    rev = "rev110wr_2026-08-19_15-38-40.pt"
    r, alpha = 8, 16.0

    cfg = LoRAConfig(base_ckpt=base_ckpt, eval_iters=50)
    packed = build_or_load()

    @torch.no_grad()
    def base_lm_loss(model) -> float:
        """LM loss on the pretraining val stream -- the capability the base
        ckpt was scored on (1.3784 at save). Movement here is what finetuning
        did to the original skill, separate from instruct-format story loss."""
        model.eval()
        rng = np.random.default_rng(cfg.seed + 2)
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch("val", cfg.batch_size, model.cfg.block_size, rng)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss)
        return torch.stack(losses).mean().item()

    def story_loss(model) -> float:
        rng = np.random.default_rng(cfg.seed + 1)  # same windows for every row
        return estimate_loss(model, packed, cfg, model.cfg.block_size, rng)["val_comp"]

    def rev_acc(model) -> float:
        random.seed(0)  # same words for every row
        acc, _ = evaluate_reverse(model, n=100, lmin=1, lmax=10)
        return acc

    def build(with_rev: bool, with_instruct: bool):
        model = load_ckpt(base_ckpt)
        if with_rev:
            apply_lora(model, r, alpha)
            load_adapter(model, rev)
            merge_lora(model)
        if with_instruct:
            apply_lora(model, r, alpha)
            load_adapter(model, instruct)
        return model

    def build_joint_row():
        from main import latest_ckpt

        model = load_ckpt(base_ckpt)
        apply_lora(model, r, alpha)
        load_adapter(model, latest_ckpt("joint"))
        return model

    rows = [
        ("base", lambda: build(False, False)),
        ("+instruct", lambda: build(False, True)),
        ("+rev", lambda: build(True, False)),
        ("+both", lambda: build(True, True)),
        ("joint", build_joint_row),
    ]
    print(
        f"\n{'model':<12} {'base val_lm':>11} {'story val_comp':>14} "
        f"{'reverse acc':>12}"
    )
    for name, make in rows:
        try:
            model = make()
        except FileNotFoundError:
            continue  # no joint ckpt trained yet
        print(
            f"{name:<12} {base_lm_loss(model):>11.3f} "
            f"{story_loss(model):>14.3f} {rev_acc(model):>11.0%}"
        )


if __name__ == "__main__":
    # test()
    # main()
    # main_rev()
    main_joint()
    # main_stack()
