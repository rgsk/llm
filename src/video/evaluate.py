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
def estimate_agreement(
    model: GPT,
    reference: GPT,
    ds: BinDataset,
    batch_size: int,
    block_size: int,
    iters: int = 50,
    generator: torch.Generator | None = None,
    device: str = "cpu",
    amp: bool = True,
) -> tuple[float, float]:
    """How much of `model` would survive speculative decoding against `reference`.

    Returns (top1, accept).

    top1 is argmax agreement, which is exactly the acceptance rate under greedy
    decoding: a drafted token is kept iff the target would have picked it too.

    accept is sum_x min(p_model(x), p_ref(x)), averaged over positions. Under
    the standard speculative rule -- draw x from the draft, keep it with
    probability min(1, p_ref(x) / p_model(x)) -- that sum IS the expected
    fraction of drafted tokens accepted, at temperature 1. It also equals
    1 - TV(p_model, p_ref), so "make the draft accepted more often" and "move
    the draft's distribution onto the target's" are the same instruction.

    This is a validation number, measured teacher-forced on real text. At
    generation time the context is the model's own output, so live acceptance
    runs a little lower -- treat this as the ceiling, and a fair one for
    comparing checkpoints, which is what it is for.

    Cost: a full reference forward per batch. Pass a smaller batch_size than
    training uses -- two [B, T, V] logit tensors is the memory here, not the
    weights.
    """
    modes = model.training, reference.training
    model.eval()
    reference.eval()
    top1s, accepts = [], []
    for _ in range(iters):
        x, _ = get_batch(ds, batch_size, block_size, generator)
        x = x.to(device)
        with autocast(str(device), amp):
            draft_logits = model(x)  # [B, T, V]
            ref_logits = reference(x)
        top1s.append((draft_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean())
        # chunked over positions: two [B*T, V] softmaxes at once is the peak
        flat_d = draft_logits.reshape(-1, draft_logits.size(-1))
        flat_r = ref_logits.reshape(-1, ref_logits.size(-1))
        overlap = []
        for i in range(0, flat_d.size(0), 4096):
            p = flat_d[i : i + 4096].float().softmax(-1)
            q = flat_r[i : i + 4096].float().softmax(-1)
            overlap.append(torch.minimum(p, q).sum(-1))
        accepts.append(torch.cat(overlap).mean())
    if modes[0]:
        model.train()
    if modes[1]:
        reference.train()
    return torch.stack(top1s).mean().item(), torch.stack(accepts).mean().item()


def tokens_per_pass(accept: float, k: int) -> float:
    """Expected tokens per target forward when drafting k of them.

    Each of the k drafted tokens survives with probability `accept`, and the
    target's own token is free at the end of the run -- so the expectation is
    1 + a + a^2 + ... + a^k. At a=0 that is 1 (the cache-only baseline, no win);
    at a=1 it is k+1.
    """
    if accept >= 1.0:
        return float(k + 1)
    return (1 - accept ** (k + 1)) / (1 - accept)


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
