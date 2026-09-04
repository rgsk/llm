"""Sliding window attention: the mask, and nothing else.

Three things get tangled together whenever this comes up, and they are three
independent rungs:

    1. a sliding window is a MASK. this file.
    2. a ring buffer is STORAGE that wraps. kv_cache.py's next episode.
    3. attention sinks are a POLICY about which slots never get evicted.

A window needs neither of the other two. Mask query i to the `window` keys
ending at i, and every position past that is simply not attended -- the cache
still holds them, the kernel still walks them, they just contribute nothing.
So this rung buys a change in what the model *computes*, and buys nothing at
all in memory. Saying otherwise is the usual confusion.

It is only correct because scores already depend on `i - j`. RoPE bakes each
key's rotation in at write time, by the absolute position that token had, and
`R(m)^T R(n) = R(n - m)` -- so a key sitting 500 positions back scores exactly
as a key 500 positions back should, whether or not anything in between is
still visible. Dropping columns from the mask cannot desynchronise anything.

The interesting fact is that a window does not cap what a *model* can see.
Stack L layers of window W and information travels W-1 positions per layer:
the receptive field is L*(W-1) + 1, the same argument as a stack of
convolutions. Mistral-7B ships W=4096 over 32 layers -- a 131k token
theoretical reach out of a 4096-wide mask. Test 4 measures exactly this.

What it costs is in test 6, and it is not free: an [T_q, T_kv] mask is the
T^2 object SDPA exists not to materialise, and handing SDPA any mask at all
rules out the flash backend. A real windowed kernel takes the window as an
integer and skips whole blocks. This is the honest version, not the fast one.
"""

"""
The band, T=8 W=3 (rows are queries, columns are keys):

pos 0  X . . . . . . .
    1  X X . . . . . .
    2  X X X . . . . .
    3  . X X X . . . .      each row sees at most 3 keys,
    4  . . X X X . . .      no matter how long the sequence gets
    5  . . . X X X . .
    6  . . . . X X X .
    7  . . . . . X X X
"""

import torch
from torch import Tensor


def sliding_window_mask(
    T_q: int,
    T_kv: int,
    window: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """[T_q, T_kv] bool, True = visible. The queries are the LAST T_q keys.

    Query row r sits at absolute position T_kv - T_q + r, because with a cache
    the new tokens are the tail of the sequence and not the start of it. It
    sees every key at or before itself, and -- given a window -- no key more
    than window - 1 positions back. window=None is plain causal attention.
    """
    assert window is None or window >= 1, "a window has to include the query itself"
    T_past = T_kv - T_q
    i = torch.arange(T_past, T_kv, device=device).unsqueeze(1)  # query positions
    j = torch.arange(T_kv, device=device)  # key positions
    visible = j <= i
    if window is not None:
        visible &= j > i - window
    return visible


if __name__ == "__main__":
    torch.manual_seed(0)
    T, W = 8, 3

    # 1. the band. window=None has to be exactly the tril every attention file
    #    in this repo has been building by hand, and a window narrows it from
    #    below without touching the diagonal -- a query always sees itself
    causal = sliding_window_mask(T, T)
    assert torch.equal(causal, torch.ones(T, T, dtype=torch.bool).tril())
    band = sliding_window_mask(T, T, W)
    assert band.shape == (T, T) and band.dtype == torch.bool
    assert torch.equal(band, causal & ~torch.ones(T, T, dtype=torch.bool).tril(-W))
    assert band.diagonal().all()
    #    row i sees exactly {i-W+1 .. i}, clipped at 0. spelled out, because
    #    every off-by-one in this file is an off-by-one in the model
    for i in range(T):
        assert band[i].nonzero().flatten().tolist() == list(
            range(max(0, i - W + 1), i + 1)
        )
    assert band.sum(dim=1).tolist() == [1, 2, 3, 3, 3, 3, 3, 3]
    #    W=1 is the degenerate end: attend to yourself and nobody else
    assert torch.equal(sliding_window_mask(T, T, 1), torch.eye(T, dtype=torch.bool))
    #    and W >= T is a window that never bites -- the plain causal mask back
    assert torch.equal(sliding_window_mask(T, T, T), causal)
    assert torch.equal(sliding_window_mask(T, T, 99), causal)
    print("the band is causal from above and window-wide from below")

    # 2. the cached shift, which is the half that is easy to get wrong. When
    #    T_q < T_kv the queries are the TAIL, so row 0 of the mask is not
    #    position 0 -- it is position T_kv - T_q. Decoding one token at a time
    #    must therefore reproduce the rows of the full mask, one per step
    full = sliding_window_mask(T, T, W)
    for t in range(T):
        step = sliding_window_mask(1, t + 1, W)  # one query, t+1 keys cached
        assert torch.equal(step[0], full[t, : t + 1])
    #    and so must a prefill-then-continue split, which is the case is_causal
    #    cannot express: three tokens, then the other five at once
    assert torch.equal(sliding_window_mask(3, 3, W), full[:3, :3])
    assert torch.equal(sliding_window_mask(T - 3, T, W), full[3:])
    print("mask rows are the same whether built at once or one step at a time")

    # 3. the oracle. A mask is only worth trusting if it means what it says, so
    #    compute attention through the band and then compute it AGAIN over the
    #    literal slice k[i-W+1 : i+1] -- no mask involved, a shorter sequence.
    #    If they agree, the masked-out columns genuinely contributed nothing
    B, NH, HS = 2, 4, 16
    q, k, v = (torch.randn(B, NH, T, HS) for _ in range(3))
    scores = q @ k.transpose(-2, -1) * HS**-0.5
    out = scores.masked_fill(~band, float("-inf")).softmax(dim=-1) @ v
    for i in range(T):
        lo = max(0, i - W + 1)
        s = q[:, :, i : i + 1] @ k[:, :, lo : i + 1].transpose(-2, -1) * HS**-0.5
        ref = s.softmax(dim=-1) @ v[:, :, lo : i + 1]
        assert (out[:, :, i] - ref[:, :, 0]).abs().max() < 1e-6, f"row {i}"
    #    ...and it is a different answer from full causal attention, which is
    #    the point: this rung changes what the model computes
    full_out = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1) @ v
    assert (out - full_out).abs().max() > 1e-2
    print("each row is attention over its slice alone, and differs from full causal")

    # 4. the fact that makes windows usable at all: depth. Reachability through
    #    L stacked layers is the boolean matrix power of the mask -- output i
    #    of layer 2 saw r, and r saw j -- so information walks W-1 positions
    #    per layer and the receptive field is L*(W-1) + 1, not W
    reach = sliding_window_mask(32, 32, W)
    M = reach.clone()
    for L in range(2, 6):
        reach = (reach.float() @ M.float()) > 0
        i = torch.arange(32).unsqueeze(1)
        j = torch.arange(32)
        assert torch.equal(reach, (j <= i) & (i - j <= L * (W - 1)))
    print(f"W={W} over 5 layers reaches {5 * (W - 1) + 1} positions back")
    #    which is why Mistral-7B's 4096-token window is not a 4096-token model
    print(f"  Mistral-7B geometry: W=4096, 32 layers -> {32 * 4095 + 1:,} tokens")

    # 5. and the same claim in a real stack, since a matrix power proves a
    #    statement about the mask and not about the layer that uses it. Three
    #    windowed attentions, no FFN and no residual to muddy it: perturb
    #    position 0 and watch how far forward the change actually travels
    from sdpa_attention import SDPAttention

    E, L_, W_, T_ = 32, 3, 4, 16
    x = torch.randn(1, T_, E)
    layers = [SDPAttention(E, NH, window=W_) for _ in range(L_)]

    def stack(z: Tensor) -> Tensor:
        for layer in layers:
            z = layer(z)
        return z

    x2 = x.clone()
    x2[:, 0] += 10.0
    moved = (stack(x2) - stack(x)).abs().amax(dim=(0, 2))
    span = L_ * (W_ - 1) + 1
    assert (moved[:span] > 1e-4).all(), moved
    assert (moved[span:] == 0).all(), moved  # structurally zero, not merely small
    print(
        f"{L_} layers of W={W_}: position 0 moves positions 0..{span - 1}, then nothing"
    )

    # 6. what it costs, stated honestly. The window changes the mask; it does
    #    not change the cache, so nothing here is smaller yet. Worse, the mask
    #    IS an [T_q, T_kv] tensor -- the T^2 object SDPA exists not to build --
    #    so at long context the thing describing the saving costs more than a
    #    step of the saving is worth. A real windowed kernel takes W as an
    #    integer and skips blocks; that is rung 2's and the kernel's problem
    print("\n  one decode step, W=4096, cache [1, 8, T, 128] bf16")
    print(f"  {'T_kv':>7}  {'kv cache':>10}  {'prefill mask':>13}")
    for T_kv in (4096, 16384, 65536):
        cache = 2 * 1 * 8 * T_kv * 128 * 2
        mask = T_kv * T_kv  # bool, one byte, if the whole thing were prefilled
        print(f"  {T_kv:>7}  {cache / 2**20:>7.0f} MB  {mask / 2**30:>10.1f} GB")

    #    and the mask is not free in the kernel either: flash takes is_causal
    #    but not an arbitrary attn_mask, so a window silently drops SDPA to the
    #    memory-efficient backend
    if torch.cuda.is_available():
        import warnings

        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        qc = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
        m512 = sliding_window_mask(512, 512, 128, device="cuda")
        print()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, kw in (
                ("is_causal", {"is_causal": True}),
                ("attn_mask", {"attn_mask": m512}),
            ):
                try:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        F.scaled_dot_product_attention(qc, qc, qc, **kw)
                    print(f"  bf16 with {name:<9} flash RUNS")
                except RuntimeError:
                    print(f"  bf16 with {name:<9} flash unavailable")

    print("\nok")
