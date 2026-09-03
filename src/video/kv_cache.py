import torch
from torch import Tensor


class KVCache:
    """One layer's k and v for every position attended so far, [B, n_kv, T_kv, hs].

    This was a type alias in common.py, and for one of the two modes below it
    still says exactly what happens:

        type KVCache = tuple[Tensor, Tensor]

    That version grew by copying. Every step allocated a whole new cache and
    torch.cat the old one into it: O(T) bytes moved per token, so O(T^2) over a
    generation, and at real context lengths the copy costs more than the
    attention it feeds (measured in fused_gqa_attention.py, test 9 -- 133 MB of
    pure reallocation per decode step on an 8-head 512-wide layer).

    So the class has two modes, and the naive one is kept alive on purpose:

        KVCache()         grow by copying -- exactly the old tuple, unchanged
        KVCache(shape)    preallocate [B, n_kv, block_size, hs] once, then write
                          each step in place at a cursor

    Both hand back (k, v) views over everything written so far, and both still
    unpack and index like the tuple they replaced: `past_k, past_v = cache` and
    `cache[0]` are what the older attention files do, and they keep working
    against either mode with not a line changed.

    Buffers are allocated on the first write, not in __init__, so device and
    dtype follow the data -- autocast, .to(bf16), and a cache that is never
    written all do the right thing without being told about it.

    Preallocation has a real cost, not only a benefit: writing in place means
    the cached path stops being differentiable, and a second write would clobber
    a tensor the first one's graph still needs. append refuses tensors that
    require grad rather than let that fail later, somewhere else. Decoding runs
    under no_grad so nothing real is lost -- but it is a trade, not a free win.

    What this does NOT do yet is wrap around. Capacity is block_size, and the
    cursor only moves forward, so a run still has to fit. Evicting the oldest
    entries and reusing their slots is a ring buffer, and it is only correct
    once positions are relative -- which is RoPE's episode, not this one.
    """

    def __init__(self, shape: tuple[int, int, int, int] | None = None):
        """shape is [B, n_kv_head, block_size, head_size]. None grows by copying."""
        self.shape = shape
        self.k: Tensor | None = None
        self.v: Tensor | None = None
        self.pos = 0  # positions written so far

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Add this step's k and v, and return everything cached so far."""
        T = k.size(2)
        if self.shape is None:  # naive mode: a fresh, bigger tensor every step
            self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
            self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
            self.pos += T
            return self.k, self.v

        # is_grad_enabled, not just requires_grad: inside no_grad the flag on
        # the tensor is still True, what changes is whether the write records a
        # graph -- and it is the graph, not the flag, that the next write breaks
        assert not (torch.is_grad_enabled() and (k.requires_grad or v.requires_grad)), (
            "a preallocated KVCache writes in place, which autograd cannot "
            "follow across steps -- decode under torch.no_grad()"
        )
        if self.k is None:  # the first write decides device and dtype
            self.k = torch.empty(self.shape, device=k.device, dtype=k.dtype)
            self.v = torch.empty_like(self.k)
        end = self.pos + T
        assert end <= self.shape[2], (
            f"sequence of {end} outgrew the cache's block_size {self.shape[2]}"
        )
        self.k[:, :, self.pos : end] = k
        self.v[:, :, self.pos : end] = v
        self.pos = end
        # views, not copies: the caller attends over the live buffer
        return self.k[:, :, :end], self.v[:, :, :end]

    # what the tuple gave for free, so nothing downstream has to change
    def __getitem__(self, i: int) -> Tensor:
        assert self.k is not None, "nothing written to this cache yet"
        return (self.k, self.v)[i][:, :, : self.pos]

    def __iter__(self):
        return iter((self[0], self[1]))

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        kind = "grow" if self.shape is None else f"buffer{tuple(self.shape)}"
        return f"KVCache({kind}, pos={self.pos})"


if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    B, NKV, T, HS = 2, 2, 8, 4
    steps = [(torch.randn(B, NKV, 1, HS), torch.randn(B, NKV, 1, HS)) for _ in range(T)]
    k_all = torch.cat([k for k, _ in steps], dim=2)  # what a full forward computes
    v_all = torch.cat([v for _, v in steps], dim=2)

    # 1. both modes are the same cache. THE test: after every step, each mode
    #    holds exactly the prefix a full forward would have computed
    grow, buf = KVCache(), KVCache((B, NKV, T, HS))
    for t, (k, v) in enumerate(steps, start=1):
        gk, gv = grow.append(k, v)
        bk, bv = buf.append(k, v)
        assert torch.equal(gk, k_all[:, :, :t]) and torch.equal(gv, v_all[:, :, :t])
        assert torch.equal(bk, gk) and torch.equal(bv, gv)
        assert grow.pos == buf.pos == t
    print("grow and buffer agree with the full forward at every step")

    # 2. and this is the difference, the only one that matters: the buffer is
    #    allocated once. Track the storage each mode returns -- grow hands back
    #    a new allocation every step, the buffer hands back a view of the same
    #    one it started with
    def storages(cache: KVCache) -> int:
        c = KVCache(cache.shape)
        seen = set()
        for k, v in steps:
            ck, _ = c.append(k, v)
            seen.add(ck.data_ptr())
        return len(seen)

    assert storages(KVCache()) == T  # a fresh tensor per token
    assert storages(KVCache((B, NKV, T, HS))) == 1  # one, for the whole run
    print(f"allocations over {T} steps -- grow: {T}   buffer: 1")

    # 3. the returned tensors really are views into that buffer, not copies:
    #    writing through the buffer shows up in a view handed out earlier
    c = KVCache((B, NKV, T, HS))
    view, _ = c.append(*steps[0])
    c.k[:, :, 0].fill_(1.5)
    assert (view == 1.5).all()
    assert view.data_ptr() == c.k.data_ptr()

    # 4. nothing is allocated until something is written, so device and dtype
    #    come from the data rather than from a constructor argument
    c = KVCache((B, NKV, T, HS))
    assert c.k is None and c.v is None and c.pos == 0
    kd = torch.randn(B, NKV, 1, HS, dtype=torch.float64)
    c.append(kd, kd)
    assert c.k.dtype == torch.float64  # not float32, and nobody said so
    assert c.k.shape == (B, NKV, T, HS)  # full capacity, only pos of it used
    #    a cache that is never written costs nothing at all
    assert KVCache((1024, 32, 8192, 128)).k is None

    # 5. the tuple interface the older attention files still use. this is the
    #    whole compatibility claim -- fused_qkv_attention.py and
    #    sdpa_attention.py cat onto `past_k, past_v = kv_cache` and never learn
    #    that the object changed underneath them
    for c in (KVCache(), KVCache((B, NKV, T, HS))):
        c.append(*steps[0])
        c.append(*steps[1])
        past_k, past_v = c  # unpacks
        assert past_k.shape == past_v.shape == (B, NKV, 2, HS)
        assert torch.equal(c[0], past_k) and torch.equal(c[1], past_v)  # indexes
        assert len(c) == 2
        assert torch.equal(torch.cat([past_k, steps[2][0]], dim=2), k_all[:, :, :3])
    print("unpacks and indexes exactly like the tuple it replaced")

    # 6. capacity is block_size, and outgrowing it is an error, not a silent
    #    overwrite. (Wrapping around instead is the ring buffer, and it needs
    #    RoPE before it is correct.)
    c = KVCache((B, NKV, 3, HS))
    c.append(k_all[:, :, :3], v_all[:, :, :3])
    try:
        c.append(*steps[0])
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "block_size" in str(e)

    # 7. the trade. grow mode is differentiable -- cat builds a graph like any
    #    other op -- and the buffer is not, because the next in-place write
    #    would clobber what this backward needs. It refuses up front instead
    kg = torch.randn(B, NKV, 1, HS, requires_grad=True)
    g = KVCache()
    g.append(kg, kg)[0].square().sum().backward()
    assert kg.grad is not None
    try:
        KVCache((B, NKV, T, HS)).append(kg, kg)
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "no_grad" in str(e)
    #    note kg.requires_grad is still True in here -- what no_grad changes
    #    is whether the write records a graph, and that is what append tests
    with torch.no_grad():  # which is how decoding runs anyway
        assert kg.requires_grad
        KVCache((B, NKV, T, HS)).append(kg, kg)
    print(
        "grow mode is differentiable; the buffer refuses grad instead of failing later"
    )

    # 8. what it costs, at a size worth measuring: decoding N tokens by copying
    #    moves sum(t for t in 1..N) rows, which is O(N^2). The buffer moves N
    Bb, NKVb, Nb, HSb = 4, 8, 512, 64
    tok = [
        (torch.randn(Bb, NKVb, 1, HSb), torch.randn(Bb, NKVb, 1, HSb))
        for _ in range(Nb)
    ]

    def decode(shape) -> float:
        c = KVCache(shape)
        c.append(*tok[0])  # warm
        c = KVCache(shape)
        t0 = time.perf_counter()
        for k, v in tok:
            c.append(k, v)
        return (time.perf_counter() - t0) * 1e3

    grow_ms = decode(None)
    buf_ms = decode((Bb, NKVb, Nb, HSb))
    row = Bb * NKVb * HSb * 4 * 2  # bytes of k+v for one position, fp32
    copied = row * Nb * (Nb + 1) // 2  # grow: the whole cache, every step
    print(f"\n  decoding {Nb} tokens, cache [{Bb}, {NKVb}, {Nb}, {HSb}] fp32")
    print(f"    grow    {grow_ms:7.2f} ms   {copied / 2**20:8.1f} MB copied")
    print(f"    buffer  {buf_ms:7.2f} ms   {row * Nb / 2**20:8.1f} MB copied")
    print(f"    {grow_ms / buf_ms:.1f}x, and the gap widens linearly with context")
    assert buf_ms < grow_ms

    print("\nok")
