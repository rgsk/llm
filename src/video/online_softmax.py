"""Softmax, and then attention, computed in one pass over chunks.

The two-pass softmax needs the max of the whole row before it can exponentiate
anything. The online version keeps a running max and rescales what it has
already accumulated whenever that max grows, which lets attention consume the
scores in tiles and never hold the [T, T] matrix that sdpa_attention.py
measures at 512 MB.
"""

import torch
from torch import Tensor


def online_softmax(x: Tensor, chunk: int, dim: int = -1) -> Tensor:
    """softmax(x) accumulated over chunks of `dim`, rescaling as the max grows."""
    x = x.transpose(dim, -1)
    m = x.new_full(x.shape[:-1] + (1,), -torch.inf)
    l = torch.zeros_like(m)
    parts = []
    for c in x.split(chunk, dim=-1):
        m_new = torch.maximum(m, c.max(dim=-1, keepdim=True).values)
        # a chunk that is entirely -inf leaves m_new at -inf, and -inf - -inf is
        # nan. substituting 0 there is safe: every term involved is -inf anyway,
        # so both exps below are 0 and nothing is accumulated
        m_safe = torch.where(m_new == -torch.inf, 0.0, m_new)
        alpha = (m - m_safe).exp()  # 0 on the first chunk, where m is -inf
        e = (c - m_safe).exp()
        l = alpha * l + e.sum(dim=-1, keepdim=True)
        # returning every probability is what makes this rescale quadratic in
        # the chunk count. flash_attention rescales one accumulator instead
        parts = [p * alpha for p in parts]
        parts.append(e)
        m = m_new
    return (torch.cat(parts, dim=-1) / l).transpose(dim, -1)


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_q: int,
    chunk_k: int,
    is_causal: bool = True,
) -> Tensor:
    """Causal attention over [B, nh, T, hs], tiled both ways.

    Same recurrence as online_softmax, accumulating the output instead of the
    probabilities: at most [chunk_q, chunk_k] scores exist at once. Queries are
    the last T_q keys, so row r sits at position T_kv - T_q + r.
    """
    T_q, T_kv = q.size(2), k.size(2)
    scale = q.size(-1) ** -0.5
    offset = T_kv - T_q
    arange = torch.arange(max(T_q, T_kv), device=q.device)
    out = []

    for qi in range(0, T_q, chunk_q):
        q_b = q[:, :, qi : qi + chunk_q]
        q_pos = offset + qi + arange[: q_b.size(2), None]
        q_last = offset + qi + q_b.size(2) - 1  # the same bound as a host int
        m = q.new_full(q_b.shape[:-1] + (1,), -torch.inf)
        l = torch.zeros_like(m)
        o = torch.zeros_like(q_b)

        for ki in range(0, T_kv, chunk_k):
            # every key in this block is in the future for every query in this
            # tile -- the block skip that makes causal tiling ~2x, not a mask.
            # comparing host ints keeps it off the device: reading the bound out
            # of q_pos would sync once per block
            if is_causal and ki > q_last:
                break
            k_b, v_b = k[:, :, ki : ki + chunk_k], v[:, :, ki : ki + chunk_k]
            s = (q_b @ k_b.transpose(-2, -1)) * scale
            if is_causal:
                k_pos = ki + arange[None, : k_b.size(2)]
                s = s.masked_fill(k_pos > q_pos, -torch.inf)

            m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            m_safe = torch.where(m_new == -torch.inf, 0.0, m_new)
            alpha = (m - m_safe).exp()
            p = (s - m_safe).exp()
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            o = alpha * o + p @ v_b
            m = m_new

        out.append(o / l)
    return torch.cat(out, dim=2)


if __name__ == "__main__":
    import time

    import torch.nn.functional as F
    from sliding_window import sliding_window_mask
    from softmax import softmax

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. same answer as the two-pass softmax, at every chunk size. 5 and 7 do
    # not divide 16, so the last chunk is short -- the case the recurrence has
    # to handle without a special case
    x = torch.randn(4, 8, 16)
    for chunk in (1, 3, 5, 7, 8, 16, 100):
        assert (online_softmax(x, chunk) - softmax(x)).abs().max() < 1e-7, chunk
    assert (online_softmax(x, 3, dim=1) - F.softmax(x, dim=1)).abs().max() < 1e-7

    # 2. chunk size is a knob on the memory, not on the answer -- two chunkings
    # differ by one fp32 ulp, from summing the same terms in a different order
    assert (online_softmax(x, 2) - online_softmax(x, 13)).abs().max() < 2e-7

    # 3. the running max carries the stability across chunks. the max here
    # lives in the LAST chunk, so every earlier chunk was accumulated against a
    # smaller max and had to be rescaled down
    big = torch.tensor([[1.0, 2.0, 1000.0, 1002.0]])
    assert torch.allclose(online_softmax(big, 2), F.softmax(big, dim=-1))
    assert not online_softmax(big, 2).isnan().any()

    # 4. a chunk that is entirely masked contributes nothing and poisons
    # nothing -- the -inf - -inf guard. this is the causal-tiling case
    masked = torch.tensor([[float("-inf"), float("-inf"), 1.0, 2.0]])
    p = online_softmax(masked, 2)
    assert torch.equal(p, F.softmax(masked, dim=-1))
    assert p[0, 0] == 0 and abs(p.sum().item() - 1.0) < 1e-6

    # 5. autograd flows through the loop unchanged -- the rescale is ordinary
    # differentiable arithmetic, not a custom backward
    xm = torch.randn(4, 16, requires_grad=True)
    xr = xm.detach().clone().requires_grad_(True)
    online_softmax(xm, 3).square().sum().backward()
    F.softmax(xr, dim=-1).square().sum().backward()
    assert (xm.grad - xr.grad).abs().max() < 1e-6

    # 6. flash_attention matches the kernel it is a model of. T_q == T_kv, so
    # SDPA's top-left triangle and our bottom-right one are the same mask
    B, nh, T, hs = 2, 4, 48, 32
    q, k, v = (torch.randn(B, nh, T, hs, device=dev) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    for cq, ck in ((8, 8), (16, 4), (5, 7), (T, T), (1, T)):
        got = flash_attention(q, k, v, cq, ck)
        assert (got - ref).abs().max() < 1e-5, (cq, ck)

    # 7. the cache case: 4 new queries against 48 keys. is_causal on SDPA is
    # WRONG here (it aligns top-left), so the oracle is the explicit mask
    q_new = q[:, :, -4:]
    mask = sliding_window_mask(4, T, None, dev)
    ref_cache = F.scaled_dot_product_attention(q_new, k, v, attn_mask=mask)
    got_cache = flash_attention(q_new, k, v, 2, 8)
    assert (got_cache - ref_cache).abs().max() < 1e-5

    # 8. grads match too, which is the part flash's real kernel has to earn
    # with a custom backward and we get for free by being slow
    qs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    qr = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    flash_attention(*qs, 8, 8).square().sum().backward()
    F.scaled_dot_product_attention(*qr, is_causal=True).square().sum().backward()
    for a, b in zip(qs, qr):
        assert (a.grad - b.grad).abs().max() < 1e-4

    if dev == "cpu":
        print("ok (cpu -- skipping the memory table)")
        raise SystemExit

    def naive(q, k, v):
        s = (q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5
        s = s.masked_fill(
            ~sliding_window_mask(q.size(2), k.size(2), None, dev), -torch.inf
        )
        return softmax(s) @ v

    def peak(fn, *a):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            fn(*a)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / 3
        return torch.cuda.max_memory_allocated() / 2**20, ms

    # 9. the table sdpa_attention.py's test 13 left as the kernel's business.
    # naive doubles T and quadruples memory; tiled doubles it
    print(f"{'T':>6}  {'naive':>18}  {'flash (128x128)':>18}  {'SDPA':>18}")
    with torch.no_grad():
        for T in (512, 1024, 2048, 4096):
            q, k, v = (torch.randn(2, 4, T, 64, device=dev) for _ in range(3))
            n_mb, n_ms = peak(naive, q, k, v)
            f_mb, f_ms = peak(lambda *a: flash_attention(*a, 128, 128), q, k, v)
            s_mb, s_ms = peak(
                lambda *a: F.scaled_dot_product_attention(*a, is_causal=True), q, k, v
            )
            print(
                f"{T:>6}  {n_mb:>8.0f} MB {n_ms:>6.1f} ms"
                f"  {f_mb:>8.0f} MB {f_ms:>6.1f} ms"
                f"  {s_mb:>8.0f} MB {s_ms:>6.1f} ms"
            )
            assert f_mb < n_mb

    # 10. and the honest part: eager tiling is SLOWER than the naive version it
    # fixes. the recurrence is the idea; the win needs the loop fused into one
    # kernel that keeps a tile in SRAM, which torch-level ops cannot express
    print(
        "\nthe recurrence is right and the loop is not the point -- see SDPA's column"
    )
    print("ok")
