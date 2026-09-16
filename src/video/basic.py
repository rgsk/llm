"""The whole thing, assembled. Every import below is a file we wrote:
torch.nn, torch.nn.functional and torch.optim appear nowhere in the model path."""

from dataclasses import asdict

import torch

from checkpoint import (
    CKPT_DIR,
    generate_ckpt_path,
    latest_ckpt,
    load_checkpoint,
)
from dataset import BinDataset, meta
from generate import generate
from gpt import GPT
from gpt_config import (  # noqa: F401
    GPTConfig,
    big_cfg,
    fineweb_cfg,
    fineweb_smoke_cfg,
    small_cfg,
)
from paths import DATASET
from tokenizer import ENDOFTEXT, tokenizer_for
from train import train
from train_config import (  # noqa: F401
    TrainConfig,
    big_train,
    fineweb_smoke_train,
    fineweb_train,
    small_train,
)

tok = tokenizer_for()  # follows VIDEO_DATASET, like BinDataset does

# TF32 tensor cores for fp32 matmuls: ~1.26x on this model, measured.
torch.set_float32_matmul_precision("high")

gpt_cfg = GPTConfig(
    vocab_size=meta["vocab_size"],
    block_size=128,
    n_embed=192,
    n_head=6,
    n_layer=4,
    attention="sdpa",
    position="learned",
)


# needs VIDEO_DATASET=fineweb_edu; train() asserts the vocab matches the shards
gpt_cfg = fineweb_cfg
train_cfg = fineweb_train

# gpt_cfg, train_cfg = fineweb_smoke_cfg, fineweb_smoke_train  # 20 min on a 4060
# gpt_cfg, train_cfg = big_cfg, big_train  # tinystories


PROMPT = ENDOFTEXT if DATASET.startswith("fineweb") else "\n"


def sample(model: GPT, prompt: str = PROMPT, max_new_tokens: int = 300, **kw) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = generate(model, idx, max_new_tokens, **kw)
    return tok.decode(out[0].tolist())


if __name__ == "__main__":
    run_training = True
    # a specific file, or None to take the newest run named train_cfg.name.
    # main.py's own checkpoints load here now -- they only need `attention`
    # named, since main.py never stored it
    ckpt = CKPT_DIR / "fineweb_smoke_2026-09-16_18-00-23.pt"
    ckpt = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}   run_training {run_training}")

    if run_training:
        print(gpt_cfg)
        print(train_cfg)
        torch.manual_seed(train_cfg.seed)
        model = GPT(**asdict(gpt_cfg)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{n_params / 1e6:.2f}M params ({n_params:,})")

        train_ds, val_ds = BinDataset("train"), BinDataset("val")
        ckpt = generate_ckpt_path(train_cfg.name)
        train(
            model, train_cfg, gpt_cfg, train_ds, val_ds, device=device, ckpt_path=ckpt
        )
        train_ds.close()
        val_ds.close()
    else:
        ckpt = ckpt or latest_ckpt(train_cfg.name)

    # always sample from the file, never the in-memory model: if the checkpoint
    # is wrong, this is where it shows
    print(f"\nloading {ckpt.name}")
    model, m = load_checkpoint(ckpt, device)
    print(f"best step {m['step']}   val {m['val_loss']:.4f}   bpc {m['bpc']:.3f}")

    print("\n--- greedy ---")
    print(sample(model, temperature=0.0, use_cache=True))
    print("\n--- temperature 0.8, top_p 0.95 ---")
    print(sample(model, temperature=0.8, top_p=0.95, use_cache=True))
