"""The whole thing, assembled. Every import below is a file we wrote:
torch.nn, torch.nn.functional and torch.optim appear nowhere in the model path."""

import sys
from dataclasses import asdict

import torch
from checkpoint import generate_ckpt_path, latest_ckpt, load_checkpoint
from generate import generate
from gpt import GPT
from gpt_config import GPTConfig
from paths import ROOT
from train import train
from train_config import TrainConfig

from data import BinDataset, meta

sys.path.append(str(ROOT / "src"))  # BPE is its own topic -- reuse the tokenizer
from tokenizer import BPETokenizer

tok = BPETokenizer.load(str(ROOT / "artifacts" / "tokenizer" / "bpe_ts_4096.json"))

gpt_cfg = GPTConfig(
    vocab_size=meta["vocab_size"],
    block_size=128,
    n_embed=192,
    n_head=6,
    n_layer=4,
)
train_cfg = TrainConfig(
    batch_size=32,
    max_steps=1500,
    lr=1e-3,
    min_lr=1e-4,
    warmup_steps=100,
    eval_interval=250,
    eval_iters=50,
    name="video",
)


def sample(model: GPT, prompt: str = "\n", max_new_tokens: int = 300, **kw) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = generate(model, idx, max_new_tokens, **kw)
    return tok.decode(out[0].tolist())


if __name__ == "__main__":
    run_training = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}   run_training {run_training}")

    if run_training:
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
        ckpt = latest_ckpt(train_cfg.name)

    # always sample from the file, never the in-memory model: if the checkpoint
    # is wrong, this is where it shows
    print(f"\nloading {ckpt.name}")
    model, m = load_checkpoint(ckpt, device)
    print(f"best step {m['step']}   val {m['val_loss']:.4f}   bpc {m['bpc']:.3f}")

    print("\n--- greedy ---")
    print(sample(model, temperature=0.0))
    print("\n--- temperature 0.8, top_p 0.95 ---")
    print(sample(model, temperature=0.8, top_p=0.95))
