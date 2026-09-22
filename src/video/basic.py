"""The whole thing, assembled. Every import below is a file we wrote:
torch.nn, torch.nn.functional and torch.optim appear nowhere in the model path.

    python basic.py pretrain --cfg fineweb_smoke
    python basic.py sft --task reverse
    python basic.py sft --task reverse --lora 8 --lr 1e-3
    python basic.py sft --task joint --lora 8 --lr 1e-3
    python basic.py sample --ckpt artifacts/checkpoints/....pt
"""

import argparse
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import torch

from checkpoint import (
    generate_ckpt_path,
    latest_ckpt,
    load_checkpoint,
    save_checkpoint,
)
from dataset import BinDataset, meta
from generate import generate
from gpt import GPT
from gpt_config import (
    GPTConfig,
    big_cfg,
    fineweb_cfg,
    fineweb_smoke_cfg,
    small_cfg,
    tinystories_cfg,
)
from instruct import Instruct
from joint import Joint
from lora import apply_lora, load_lora, param_counts, save_lora
from reverse import Reverse
from sft import sft
from tokenizer import ENDOFTEXT, tokenizer_for
from train import train
from train_config import (
    TrainConfig,
    big_train,
    fineweb_smoke_train,
    fineweb_train,
    sft_train,
    small_train,
    tinystories_train,
)

tok = tokenizer_for()  # follows VIDEO_DATASET, like BinDataset does

# TF32 tensor cores for fp32 matmuls: ~1.26x on this model, measured.
torch.set_float32_matmul_precision("high")

# a pretraining run is a (model, schedule) pair -- neither half means much alone
CFGS: dict[str, tuple[GPTConfig, TrainConfig]] = {
    "small": (small_cfg, small_train),
    "big": (big_cfg, big_train),
    "fineweb_smoke": (fineweb_smoke_cfg, fineweb_smoke_train),
    "fineweb": (fineweb_cfg, fineweb_train),
    "tinystories": (tinystories_cfg, tinystories_train),
}
# the registry lives here, not in task.py: task.py is what reverse.py imports
TASKS = {"reverse": Reverse, "instruct": Instruct, "joint": Joint}

PROMPT = ENDOFTEXT if ENDOFTEXT in tok.specials else "\n"


def sample(model: GPT, prompt: str = PROMPT, max_new_tokens: int = 300, **kw) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = generate(model, idx, max_new_tokens, **kw)
    return tok.decode(out[0].tolist())


def overrides(args, **names) -> dict:
    """Flags left unset keep the preset's value."""
    return {
        k: getattr(args, a) for k, a in names.items() if getattr(args, a) is not None
    }


def cmd_pretrain(args, device: str) -> Path:
    gpt_cfg, train_cfg = CFGS[args.cfg]
    train_cfg = replace(train_cfg, **overrides(args, max_steps="steps", lr="lr"))
    torch.manual_seed(train_cfg.seed)

    resume = None
    if args.resume:
        # get_lr is a pure function of the step, so the schedule picks up
        # exactly where it left off -- as long as max_steps is the same number
        # it was. Shortening it would finish the cosine early and a later run
        # would warm up again, which is two schedules, not one.
        path = Path(args.ckpt) if args.ckpt else latest_ckpt(train_cfg.name)
        model, resume = load_checkpoint(path, device)
        gpt_cfg = GPTConfig.from_dict(resume["config"])
        print(f"resuming {path.name} at step {resume['step']}")
    else:
        model = GPT(**asdict(gpt_cfg)).to(device)

    print(gpt_cfg, train_cfg, sep="\n")
    n_params = sum(p.numel() for p in model.parameters())

    # 20 tok/param is chinchilla's compute-optimal ratio; below it the model is
    # bigger than the data needs
    tokens = (
        train_cfg.batch_size
        * gpt_cfg.block_size
        * train_cfg.grad_accum_steps
        * train_cfg.max_steps
    )
    print(f"{n_params / 1e6:.2f}M params ({n_params:,})")
    print(
        f"{tokens / 1e6:.0f}M tokens  {tokens / n_params:.1f} tok/param (20 optimal)"
        f"  {tokens / meta['train']['n_tokens']:.2f} epochs"
    )

    train_ds, val_ds = BinDataset("train"), BinDataset("val")
    # a fresh path even on resume, so a diverging continuation cannot eat the
    # checkpoint it started from
    ckpt = generate_ckpt_path(train_cfg.name)
    train(
        model,
        train_cfg,
        gpt_cfg,
        train_ds,
        val_ds,
        device=device,
        ckpt_path=ckpt,
        resume=resume,
    )
    train_ds.close()
    val_ds.close()
    return ckpt


def cmd_sft(args, device: str) -> Path:
    base = Path(args.ckpt) if args.ckpt else latest_ckpt(args.base, max_val_loss=6.0)
    # flex is what SFT trains on; the base was pretrained under plain sdpa, and
    # the weights are the same either way
    model, m = load_checkpoint(base, device, attention="flex")
    gpt_cfg = replace(GPTConfig.from_dict(m["config"]), attention="flex")
    task = TASKS[args.task]()
    save = save_checkpoint
    if args.lora:
        alpha = 2.0 * args.lora
        apply_lora(model, args.lora, alpha)
        n, total = param_counts(model)
        print(f"lora r={args.lora} alpha={alpha:g}  {n:,} trainable ({n / total:.2%})")
        # the base path travels in the file: an adapter alone rebuilds nothing
        save = partial(save_lora, base_ckpt=base, r=args.lora, alpha=alpha)
    cfg = replace(
        sft_train,
        name=f"{'lora' if args.lora else 'sft'}_{task.name}",
        use_wandb=args.wandb,
        **overrides(
            args,
            max_steps="steps",
            lr="lr",
            batch_size="batch_size",
            grad_accum_steps="grad_accum",
            eval_interval="eval_interval",
        ),
    )
    print(f"base {base.name}   val {m['val_loss']:.4f}   task {task.name}")
    print(cfg)
    ckpt = generate_ckpt_path(cfg.name)
    sft(
        model,
        task,
        cfg,
        gpt_cfg,
        block_size=args.block_size,
        device=device,
        ckpt_path=ckpt,
        eval_n=args.eval_n,
        select=args.select,
        save=save,
    )
    return ckpt


def cmd_sample(args, device: str) -> Path:
    return Path(args.ckpt) if args.ckpt else latest_ckpt(args.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    pre = sub.add_parser("pretrain", help="train a model from scratch")
    pre.add_argument("--cfg", default="fineweb_smoke", choices=list(CFGS))
    pre.add_argument("--resume", action="store_true", help="continue a run")
    pre.add_argument("--ckpt", help="which one to resume (default: newest --cfg run)")

    ft = sub.add_parser("sft", help="finetune a checkpoint on a task")
    ft.add_argument("--task", default="reverse", choices=list(TASKS))
    ft.add_argument("--ckpt", help="base checkpoint (default: newest --base run)")
    ft.add_argument("--base", default="fineweb", help="checkpoint name prefix")
    ft.add_argument("--block-size", type=int, default=256, help="training window")
    ft.add_argument("--batch-size", type=int)
    # the OOM lever: the [B, T, vocab] logits are the peak, and accumulating
    # keeps tokens-per-step the same while halving it
    ft.add_argument("--grad-accum", type=int)
    ft.add_argument("--eval-n", type=int, default=200, help="samples per scoreboard")
    # LoRA usually wants ~5-10x the full-finetune lr; pass --lr, it is not measured
    ft.add_argument("--lora", type=int, metavar="R", help="train a rank-R adapter")
    # a generated scoreboard at eval_n 40 carries ~0.1, enough to keep the wrong
    # step; comp is the same number every eval. Pass a task metric to override.
    ft.add_argument("--select", default="comp", help="metric the best ckpt is kept on")
    # instruct generates one story per sample, so the scoreboard is the
    # expensive part of the loop, not the training
    ft.add_argument("--eval-interval", type=int)
    ft.add_argument("--wandb", action="store_true")

    smp = sub.add_parser("sample", help="generate from a checkpoint")
    smp.add_argument("--ckpt")
    smp.add_argument("--name", default="fineweb", help="checkpoint name prefix")
    smp.add_argument("--prompt", default=PROMPT)
    smp.add_argument("--tokens", type=int, default=300)

    for p in (pre, ft):
        p.add_argument("--steps", type=int)
        p.add_argument("--lr", type=float)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    ckpt = {"pretrain": cmd_pretrain, "sft": cmd_sft, "sample": cmd_sample}[args.cmd](
        args, device
    )

    # always sample from the file, never the in-memory model: if the checkpoint
    # is wrong, this is where it shows
    print(f"\nloading {ckpt.name}")
    load = load_lora if getattr(args, "lora", None) else load_checkpoint
    model, m = load(ckpt, device)
    print("  ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)))

    if "task" in m:  # a task model is scored, not sampled from
        scores = TASKS[m["task"]]().evaluate(model, getattr(args, "eval_n", 200))
        print("  ".join(f"{k} {v:.3f}" for k, v in scores.items()))
        return

    prompt = getattr(args, "prompt", PROMPT)
    tokens = getattr(args, "tokens", 300)
    print("\n--- greedy ---")
    print(sample(model, prompt, tokens, temperature=0.0, use_cache=True))
    print("\n--- temperature 0.8, top_p 0.95 ---")
    print(sample(model, prompt, tokens, temperature=0.8, top_p=0.95, use_cache=True))


if __name__ == "__main__":
    main()
