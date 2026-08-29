import torch.nn.functional as F
from dropout import Dropout
from linear import Linear
from module import Module
from residual_proj import ResidualProj
from torch import Tensor


class SDPAttention(Module):
    """FusedQKVAttention with the mask, the softmax and both matmuls handed to
    torch's fused kernel.

    This is the one file in the series that does not rebuild what it calls, and
    the reason is worth being precise about. SDPA's advantage is not arithmetic
    -- it computes exactly what the previous file computes. It is that it never
    materialises the [B, nh, T, T] score matrix: it walks the sequence in tiles
    that stay in on-chip SRAM, so memory grows with T instead of T^2. That is a
    memory-access pattern, not a formula, and eager tensor ops cannot express it.

    People call this layer "FlashAttention". The name overpromises. SDPA is a
    dispatcher: it picks a backend that fits the inputs, and the flash kernel
    needs fp16/bf16. Ask for it in fp32 and you get the memory-efficient backend
    instead, silently.
    """

    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.0):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        self.dropout_p = dropout
        self.resid_dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, E = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(E, dim=-1)

        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # is_causal builds the same lower-triangular mask, inside the kernel.
        # dropout_p must be zeroed by hand at eval: the kernel is a function,
        # it cannot see that the Module around it is in eval mode
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, E)
        return self.resid_dropout(self.proj(out))


if __name__ == "__main__":
    import time

    import torch
    from fused_qkv_attention import FusedQKVAttention

    torch.manual_seed(0)
    B, T, E, NH = 2, 8, 32, 4
    x = torch.randn(B, T, E)
    sdpa, fused = SDPAttention(E, NH), FusedQKVAttention(E, NH, T)

    # 1. a drop-in replacement: same parameters, same names, same shapes
    assert sdpa.state_dict().keys() == fused.state_dict().keys()
    assert sum(p.numel() for p in sdpa.parameters()) == sum(
        p.numel() for p in fused.parameters()
    )

    # 2. same weights in, same answer out -- the kernel is not an approximation
    sdpa.load_state_dict(fused.state_dict())
    print("sdpa vs hand-written:", (sdpa(x) - fused(x)).abs().max().item())
    assert (sdpa(x) - fused(x)).abs().max() < 1e-6

    # 3. ... and the same gradients
    xs = x.clone().requires_grad_(True)
    xf = x.clone().requires_grad_(True)
    sdpa(xs).square().sum().backward()
    fused(xf).square().sum().backward()
    assert (xs.grad - xf.grad).abs().max() < 1e-5
    assert all(
        (a.grad - b.grad).abs().max() < 1e-5
        for a, b in zip(sdpa.parameters(), fused.parameters())
    )

    # 4. is_causal=True really is our tril: nothing leaks backwards
    out = sdpa(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        out2 = sdpa(x2)
        assert (out2[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"
        assert (out2[:, t] - out[:, t]).abs().max() > 1e-3
    print("causality holds for all t")

    # dropout: the kernel cannot see self.training, so the guard is load-bearing
    d = SDPAttention(E, NH, dropout=0.5)
    d.train()
    assert not torch.equal(d(x), d(x))
    d.eval()
    assert torch.equal(d(x), d(x))

    if not torch.cuda.is_available():
        print("ok (cpu -- skipped the memory and backend tests)")
        raise SystemExit

    # 5. the point of the whole file: memory. Ours allocates [B, nh, T, T] and
    #    several copies of it; SDPA allocates none, so it scales with T not T^2
    print(
        f"\npeak memory for one forward, B=4 nh=8 E=512  ({torch.cuda.get_device_name(0)})"
    )
    print(f"  {'T':>6}  {'hand-written':>14}  {'sdpa':>10}  {'B*nh*T*T fp32':>14}")
    for Tb in (512, 1024, 2048):
        xb = torch.randn(4, Tb, 512, device="cuda")
        peak = {}
        for name, m in (
            ("fused", FusedQKVAttention(512, 8, Tb).to("cuda")),
            ("sdpa", SDPAttention(512, 8).to("cuda")),
        ):
            m(xb)  # warm up allocator
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            m(xb)
            peak[name] = (torch.cuda.max_memory_allocated() - before) / 2**20
        scores_mb = 4 * 8 * Tb * Tb * 4 / 2**20
        print(
            f"  {Tb:>6}  {peak['fused']:>11.0f} MB  {peak['sdpa']:>7.0f} MB  {scores_mb:>11.0f} MB"
        )
        assert peak["sdpa"] < peak["fused"]
    """
    Test 5 is the table to hold on screen. Doubling T roughly quadruples the hand-written column (133 → 458 → 1689 MB) and exactly doubles SDPA's (24 → 48 → 96 MB). That's T² versus T, measured, not asserted. The last column shows why: one copy of the score matrix at T=2048 is 512 MB, and we hold several at once — scores, the masked version, the softmax output. At T=2048 this layer alone is 1.7 GB on an 8 GB card, which is the actual reason long-context models were impossible before this kernel.
    """

    # 6. and it is faster, which is a consequence of the same thing -- fewer
    #    round trips to HBM, not fewer floating point operations
    xb = torch.randn(4, 1024, 512, device="cuda")
    t = {}
    for name, m in (
        ("fused", FusedQKVAttention(512, 8, Tb).to("cuda")),
        ("sdpa", SDPAttention(512, 8).to("cuda")),
    ):
        for _ in range(3):
            m(xb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            m(xb)
        torch.cuda.synchronize()
        t[name] = (time.perf_counter() - t0) / 20 * 1e3
    print(
        f"\nforward at T=1024 -- hand-written {t['fused']:.2f} ms   "
        f"sdpa {t['sdpa']:.2f} ms   {t['fused'] / t['sdpa']:.1f}x"
    )
    assert t["sdpa"] < t["fused"]

    """
    Test 6's 4.4x is worth framing correctly: SDPA does the same number of FLOPs. It wins by not shipping 512 MB out to HBM and back between each step. That's the whole of FlashAttention's idea — it's an IO story, not an arithmetic one.
    """

    # 7. "flash" is a backend, not this layer. In fp32 it is not even available
    import warnings

    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(4, 8, 512, 64, device="cuda")
    print("\nwhich backends accept these inputs:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch narrates every backend it rejects
        """
        Not suppressing warnings prints this Error when float32 tries to use FLASH_ATTENTION
        
            UserWarning: Expected query, key and value to all be of dtype: {Half, BFloat16}. Got Query dtype: float, Key dtype: float, and Value dtype: float instead.
        """
        for dt in (torch.float32, torch.bfloat16):
            ok = []
            for be in (
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ):
                try:
                    with sdpa_kernel(be):
                        F.scaled_dot_product_attention(
                            q.to(dt), q.to(dt), q.to(dt), is_causal=True
                        )
                    ok.append(be.name)
                except RuntimeError:
                    pass
            print(f"  {dt!s:<16} {', '.join(ok)}")

    # 8. and the rest of the speedup is behind the DTYPE, not behind this file.
    #    flash needs fp16/bf16, so in fp32 it is unavailable on every card --
    #    including this one, which supports bf16 perfectly well. The model just
    #    is not running in it. Mixed precision is a separate change.
    print(f"\nbf16 supported on this GPU: {torch.cuda.is_bf16_supported()}")
    m = SDPAttention(512, 8).to("cuda")
    xb = torch.randn(4, 1024, 512, device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dt in (torch.float32, torch.bfloat16):
            xd = xb.to(dt)
            m.to(dt)
            for _ in range(3):
                m(xd)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            for _ in range(20):
                m(xd)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1e3
            mem = (torch.cuda.max_memory_allocated() - base) / 2**20
            try:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    m(xd)
                flash = "flash RUNS"
            except RuntimeError:
                flash = "flash unavailable"
            print(f"  {dt!s:<16} {ms:5.2f} ms   {mem:>3.0f} MB peak   {flash}")

    print("\nok")
