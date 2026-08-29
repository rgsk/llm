import math

import torch
from amp import autocast
from cross_entropy import cross_entropy
from dataset import BinDataset, get_batch, meta
from gpt import GPT


def batch_loss(
    model: GPT, x: torch.Tensor, y: torch.Tensor, amp: bool = True
) -> torch.Tensor:
    B, T = x.shape
    with autocast(str(x.device), amp):
        logits = model(x)  # [B, T, V]
        return cross_entropy(logits.reshape(B * T, -1), y.reshape(B * T))


@torch.no_grad()
def estimate_loss(
    model: GPT,
    ds: BinDataset,
    batch_size: int,
    block_size: int,
    iters: int = 200,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> float:
    """Cheap sampled estimate: the mean loss over `iters` random batches."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(ds, batch_size, block_size, generator)
        losses.append(batch_loss(model, x.to(device), y.to(device)))
    if was_training:
        model.train()  # never leave the caller's model in eval
    return torch.stack(losses).mean().item()


@torch.no_grad()
def full_loss(
    model: GPT,
    ds: BinDataset,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    max_windows: int | None = None,
) -> float:
    """Deterministic sweep over non-overlapping windows of the whole split."""
    was_training = model.training
    model.eval()
    n_windows = (ds.n_tokens - 1) // block_size
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    total = 0.0
    count = 0
    for start in range(0, n_windows, batch_size):
        stop = min(start + batch_size, n_windows)
        chunks = [ds.tokens(w * block_size, block_size + 1) for w in range(start, stop)]
        x = torch.stack([c[:-1] for c in chunks]).to(device)
        y = torch.stack([c[1:] for c in chunks]).to(device)
        loss = batch_loss(model, x, y)
        # weight by token count so the mean is right even if the last chunk is short
        total += loss.item() * y.numel()
        count += y.numel()
    if was_training:
        model.train()
    return total / count


def bits_per_char(loss: float, split: str = "val") -> float:
    """nats/token -> bits/char, the tokenizer-independent way to compare models."""
    return loss / math.log(2) / meta[split]["chars_per_token"]


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V, BS = meta["vocab_size"], 64
    torch.manual_seed(1337)
    m = GPT(V, BS, n_embed=128, n_head=4, n_layer=3).to(dev)
    val = BinDataset("val")

    # 1. an untrained model sits at ln(V), by either measure
    est = estimate_loss(m, val, 32, BS, iters=50, device=dev)
    full = full_loss(m, val, 64, BS, device=dev, max_windows=4000)
    print(f"estimate {est:.4f}   full {full:.4f}   ln(V) {math.log(V):.4f}")
    assert abs(est - math.log(V)) < 0.05
    assert abs(full - math.log(V)) < 0.05

    # 2. full_loss is deterministic, and independent of batch_size
    assert full_loss(m, val, 64, BS, device=dev, max_windows=4000) == full
    b2 = full_loss(m, val, 32, BS, device=dev, max_windows=4000)
    assert abs(full - b2) < 1e-6  # token-weighting is what makes this true

    # 3. the model's mode is restored, and no gradients are built
    m.train()
    m.zero_grad()
    estimate_loss(m, val, 8, BS, iters=2, device=dev)
    assert m.training and m.blocks[0].training
    m.eval()
    estimate_loss(m, val, 8, BS, iters=2, device=dev)
    assert not m.training
    assert all(p.grad is None for p in m.parameters())

    # 4. the sampled estimate gets less noisy as ~1/sqrt(iters)
    m.eval()
    for iters in (5, 50, 200):
        vals = [
            estimate_loss(m, val, 32, BS, iters=iters, device=dev) for _ in range(5)
        ]
        print(f"iters={iters:>3}  spread {max(vals) - min(vals):.4f}")

    # 5. the real thing, over every window in val
    full_all = full_loss(m, val, 64, BS, device=dev)
    print(f"full val loss {full_all:.4f}   bpc {bits_per_char(full_all):.4f}")
    val.close()

    print("ok")
