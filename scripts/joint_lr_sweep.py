"""LoRA lr sweep on the joint 67/33 instruct+reverse mix.

Runs on RunPod. Reverse wanted 1e-3..3e-3 and instruct wanted more than it, so
the mix has two appetites and the sweep is the only way to settle it. Reports
both scoreboards, comp, and TinyStories val loss -- reverse showed the lr, not
the parameter count, decides what the base keeps.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path.insert(0, str(ROOT / "src" / "video"))

from checkpoint import load_checkpoint
from dataset import BinDataset
from evaluate import estimate_loss
from gpt_config import GPTConfig
from joint import Joint
from lora import apply_lora, param_counts
from sft import sft
from train_config import sft_train

BASE = ROOT / "artifacts" / "checkpoints" / "tinystories_2026-09-22_13-20-13.pt"
LRS = (3e-4, 1e-3, 3e-3, 1e-2)
DEV, STEPS, R, EVAL_N = "cuda", 600, 8, 40
torch.set_float32_matmul_precision("high")

print(f"gpu {torch.cuda.get_device_name()}", flush=True)
ds, gen = BinDataset("val"), torch.Generator()


def ts_val(model):
    gen.manual_seed(0)  # identical windows for every row
    return estimate_loss(model, ds, 16, 512, iters=50, generator=gen, device=DEV)


base, m = load_checkpoint(BASE, DEV, attention="flex")
gpt_cfg = replace(GPTConfig.from_dict(m["config"]), attention="flex")
base_loss = ts_val(base)
print(f"base ts val {base_loss:.4f}", flush=True)
del base
torch.cuda.empty_cache()

task = Joint()  # 67/33 instruct/reverse
rows = []
for lr in LRS:
    model, _ = load_checkpoint(BASE, DEV, attention="flex")
    apply_lora(model, R)
    n, total = param_counts(model)
    cfg = replace(
        sft_train, name="joint_lr_sweep", lr=lr, max_steps=STEPS,
        warmup_steps=40, eval_interval=200, batch_size=32, use_wandb=False,
    )  # fmt: skip
    print(
        f"\n=== lr {lr:.0e}  r={R}  {n:,} trainable ({n / total:.2%}) ===", flush=True
    )
    t0 = time.perf_counter()
    h = sft(model, task, cfg, gpt_cfg, block_size=512, device=DEV, eval_n=EVAL_N)
    rows.append((lr, h[-1], ts_val(model), time.perf_counter() - t0))
    del model
    torch.cuda.empty_cache()

ds.close()

cols = ("instruct_words", "instruct_words_all", "instruct_stop_rate",
        "reverse_exact_match", "reverse_stop_rate")  # fmt: skip
print(
    f"\n{'lr':>8} {'comp':>8} {'prompt':>8} {'ins_w':>7} {'ins_all':>8}"
    f" {'ins_stop':>9} {'rev_em':>7} {'ts val':>8} {'vs base':>8} {'secs':>6}"
)
for lr, last, tsv, secs in rows:
    got = [last.get(c, float("nan")) for c in cols]
    print(
        f"{lr:>8.0e} {last['comp']:>8.4f} {last['prompt']:>8.4f} {got[0]:>7.3f}"
        f" {got[1]:>8.3f} {got[2]:>9.3f} {got[3]:>7.3f} {tsv:>8.4f}"
        f" {tsv - base_loss:>+8.4f} {secs:>6.0f}"
    )
print(
    "\nreference (separate tasks, 27M base): reverse full SFT em 1.000;"
    " instruct full SFT comp 1.0945. Old track joint LoRA: 0.969 vs 1.000."
)
print("\n------ FINISHED ------", flush=True)  # the pod is rented until this shows
