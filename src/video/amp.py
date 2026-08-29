import contextlib

import torch


def autocast(device: str, enabled: bool = True):
    """Run the matmuls in bfloat16 while the weights stay float32.

    autocast is selective, and that is the whole design: matmuls run in bf16
    because they are the FLOPs and tolerate 8 mantissa bits, while reductions,
    softmax, normalisation and the loss stay fp32 because summing thousands of
    terms at that precision loses everything.

    The stored parameters are NOT cast, so opt.step() is a full-precision
    update. model.to(bfloat16) casts them, and then `w - lr * g` rounds straight
    back to `w` for a small lr and the model quietly stops learning.

    bf16 needs no GradScaler: it gives up mantissa bits, not exponent range, so
    gradients that fp32 can represent do not underflow. fp16 is the one that
    needs a scaler.

    Only ever a speed/memory choice -- a no-op on CPU, and off by request.
    """
    if not enabled or not device.startswith("cuda"):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


if __name__ == "__main__":
    from dataclasses import asdict

    from cross_entropy import cross_entropy
    from gpt import GPT
    from gpt_config import GPTConfig

    torch.manual_seed(0)
    V = 256
    cfg = GPTConfig(vocab_size=V, block_size=32, n_embed=64, n_head=4, n_layer=2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = GPT(**asdict(cfg)).to(dev)
    x = torch.randint(0, V, (4, 16), device=dev)

    # 1. disabled, and on cpu, it is a plain no-op
    assert isinstance(autocast(dev, enabled=False), contextlib.nullcontext)
    assert isinstance(autocast("cpu"), contextlib.nullcontext)
    with autocast(dev, enabled=False):
        assert m(x).dtype == torch.float32

    if dev == "cpu":
        print("ok (cpu -- nothing to autocast)")
        raise SystemExit

    # 2. inside, matmul output is bf16 but the loss is promoted back to fp32
    with autocast(dev):
        logits = m(x)
        loss = cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    print(f"logits {logits.dtype}   loss {loss.dtype}")
    assert logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32  # the reduction is kept in fp32

    # 3. the weights are untouched -- this is what keeps opt.step() exact
    assert all(p.dtype == torch.float32 for p in m.parameters())
    loss.backward()
    assert all(p.grad.dtype == torch.float32 for p in m.parameters())

    # 4. why the weights must stay fp32: a realistic update vanishes in bf16
    w = torch.tensor([1.0])
    step = 3e-4 * 0.01  # lr * a small gradient
    assert (w - step).item() != 1.0  # fp32 keeps it
    wb = w.bfloat16()
    assert (wb - torch.tensor([step]).bfloat16()).float().item() == 1.0  # bf16 loses it
    print(f"fp32 keeps a {step:.1e} update; bf16 rounds it away entirely")

    # 5. same answer, to bf16 precision -- it is a speed knob, not a model change
    m.eval()
    with torch.no_grad():
        full = m(x)
        with autocast(dev):
            half = m(x)
    diff = ((half.float() - full).abs().max()).item()
    print(f"difference vs fp32: {diff:.2e}")
    assert diff < 0.05

    print("ok")
