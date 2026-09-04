"""Rung 2: storage that wraps.

sliding_window.py stopped the model LOOKING at old keys. It did not stop the
cache holding them -- capacity was still block_size and a long run still had to
fit. This closes that gap: capacity becomes the window, the cursor wraps, and
the entry for position p lives in slot p % C forever. Memory per layer goes
from growing with the run to flat, which is the half of preallocation that
actually pays.

Two things change, and only one of them is the wrapping.

The wrapping is easy. The consequence is not: once the cursor has been round
once, self.k[:, :, :fill] is no longer in position order, so the mask can no
longer come from arange. The cache has to say which absolute position lives in
each slot, and attention has to build its mask from that vector. That is
`positions`, and it is the real content of this file.

What does NOT have to happen is un-permuting the ring. Softmax is
permutation-equivariant over the key axis: permute the columns of the scores
and the rows of v by the same permutation and w @ v is unchanged. Every key
already carries its own position -- rope baked the rotation in at write time --
so slot order is not information. Test 4 measures this rather than asserting
it, because it is the load-bearing simplification.

The cost is that a wrapped ring cannot go backwards. rollback() exists for
speculative decoding, and the entries it wants back are precisely the ones
eviction overwrote. That is a genuine incompatibility, not an implementation
gap, so append refuses it rather than returning quietly wrong keys.
"""

import torch
from torch import Tensor

from kv_cache import KVCache


class RingKVCache(KVCache):
    """A KVCache whose cursor wraps. Capacity is the window, not block_size.

    Slot invariant: the entry for absolute position p lives in slot p % C. So
    the buffer always holds the most recent min(pos, C) positions, and
    `positions[s]` is the absolute position of slot s.
    """

    def __init__(self, shape: tuple[int, int, int, int]):
        """shape is [B, n_kv_head, capacity, head_size]. capacity IS the window."""
        assert shape is not None, "a ring buffer has to know its capacity"
        super().__init__(shape)
        self.positions: Tensor | None = None  # [C] absolute position per slot
        self.key_positions: Tensor | None = None  # positions of the last append

    @property
    def capacity(self) -> int:
        return self.shape[2]

    @property
    def fill(self) -> int:
        """Slots holding a live entry. Saturates at capacity; pos does not."""
        return min(self.pos, self.capacity)

    def _write(self, k: Tensor, v: Tensor, new_pos: Tensor) -> None:
        T, C = k.size(2), self.capacity
        # rows older than the last C are overwritten within this very call, so
        # writing them would be pure waste -- and, worse, would need the slots
        # to be visited in order to end up with the right survivor
        keep = min(T, C)
        slots = new_pos[T - keep :] % C
        self.k[:, :, slots] = k[:, :, T - keep :]
        self.v[:, :, slots] = v[:, :, T - keep :]
        self.positions[slots] = new_pos[T - keep :]
        self.pos += T

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        T, C = k.size(2), self.capacity
        assert not (torch.is_grad_enabled() and (k.requires_grad or v.requires_grad)), (
            "a RingKVCache writes in place, which autograd cannot follow across "
            "steps -- decode under torch.no_grad()"
        )
        if self.k is None:  # the first write decides device and dtype
            self.k = torch.empty(self.shape, device=k.device, dtype=k.dtype)
            self.v = torch.empty_like(self.k)
            self.positions = torch.empty(C, dtype=torch.long, device=k.device)
        new_pos = torch.arange(self.pos, self.pos + T, device=k.device)

        if T == 1:
            # the decode path, and the only one that costs nothing. the single
            # query sits at self.pos and may see the C positions ending there.
            # writing FIRST evicts exactly the one that just fell out of its
            # window, so afterwards the ring is precisely the attend set
            self._write(k, v, new_pos)
            self.key_positions = self.positions[: self.fill]
            return self.k[:, :, : self.fill], self.v[:, :, : self.fill]

        # T > 1: the early queries of this chunk need keys the late ones evict,
        # so the ring cannot be both the storage and the attend set. cat, which
        # is what grow mode always did -- prefill and speculative verification
        # are not the hot path, and this keeps the hot path free
        f = self.fill
        if f:
            k_att = torch.cat([self.k[:, :, :f], k], dim=2)
            v_att = torch.cat([self.v[:, :, :f], v], dim=2)
            self.key_positions = torch.cat([self.positions[:f], new_pos])
        else:
            k_att, v_att, self.key_positions = k, v, new_pos
        self._write(k, v, new_pos)
        return k_att, v_att

    def rollback(self, pos: int) -> None:
        """Forget everything after `pos` -- only while the ring has not wrapped.

        Once it has, the entries a rollback wants back were overwritten by the
        very writes it is undoing. Speculative decoding and eviction are
        mutually exclusive at that point, and saying so is the honest answer:
        the alternative is a cache that returns stale keys and a generation
        that is quietly wrong.
        """
        assert 0 <= pos <= self.pos, f"cannot roll back to {pos} from {self.pos}"
        assert self.pos <= self.capacity or pos == self.pos, (
            f"a wrapped ring cannot roll back: positions "
            f"{max(0, pos - self.capacity)}..{self.pos - self.capacity - 1} were "
            f"already overwritten. eviction and rollback are mutually constrained"
        )
        self.pos = pos

    def __getitem__(self, i: int) -> Tensor:
        assert self.k is not None, "nothing written to this cache yet"
        return (self.k, self.v)[i][:, :, : self.fill]

    def __repr__(self) -> str:
        return f"RingKVCache(capacity={self.capacity}, pos={self.pos})"


if __name__ == "__main__":
    from sliding_window import window_mask_from_positions

    torch.manual_seed(0)
    B, NKV, HS, C = 2, 2, 4, 4
    N = 11  # more tokens than the ring holds, so it goes round twice and a bit
    steps = [(torch.randn(B, NKV, 1, HS), torch.randn(B, NKV, 1, HS)) for _ in range(N)]
    k_all = torch.cat([k for k, _ in steps], dim=2)  # every position, in order
    v_all = torch.cat([v for _, v in steps], dim=2)

    # 1. until it wraps, a ring IS the preallocated buffer -- same entries, same
    #    order, same everything. The first C tokens must not be able to tell
    #    which class they were written into, or nothing below is a fair test
    ring, buf = RingKVCache((B, NKV, C, HS)), KVCache((B, NKV, C, HS))
    for t in range(C):
        rk, rv = ring.append(*steps[t])
        bk, bv = buf.append(*steps[t])
        assert torch.equal(rk, bk) and torch.equal(rv, bv)
        assert torch.equal(rk, k_all[:, :, : t + 1])
        assert ring.pos == buf.pos == t + 1
    print(f"the first {C} tokens are byte-identical to a plain buffer")

    # 2. and then it wraps, which the buffer cannot do at all -- it raises. the
    #    ring keeps going, and what it holds is exactly the last C positions
    try:
        buf.append(*steps[C])
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "block_size" in str(e)

    for t in range(C, N):
        rk, rv = ring.append(*steps[t])
        assert rk.shape == (B, NKV, C, HS)  # flat, forever
        want = list(range(t - C + 1, t + 1))  # the C positions ending at t
        assert sorted(ring.key_positions.tolist()) == want
        #    the slot invariant: position p lives in slot p % C, which is what
        #    makes the write a single indexed store and not a shift
        for p in want:
            assert ring.positions[p % C].item() == p
            assert torch.equal(ring.k[:, :, p % C], k_all[:, :, p])
    assert ring.pos == N and ring.fill == C  # pos counts tokens, fill saturates
    print(
        f"after {N} tokens the ring holds positions {sorted(ring.key_positions.tolist())}"
    )

    # 3. the point of the whole file: memory stops growing. one allocation, and
    #    a returned tensor that never exceeds the capacity no matter how long
    #    the run is. compare with grow mode, which is O(N) per step and O(N^2)
    #    over a generation
    ptrs, widths = set(), set()
    r = RingKVCache((B, NKV, C, HS))
    g = KVCache()
    for step in steps:
        rk, _ = r.append(*step)
        gk, _ = g.append(*step)
        ptrs.add(rk.data_ptr())
        widths.add((rk.size(2), gk.size(2)))
    assert len(ptrs) == 1, "the ring should be allocated once, not per step"
    assert max(w for w, _ in widths) == C and max(w for _, w in widths) == N
    print(f"over {N} tokens -- ring: 1 allocation, width <= {C}   grow: width {N}")

    # 4. THE simplification, measured rather than asserted: slot order is not
    #    information. Attention is a weighted sum over keys, and softmax is
    #    permutation-equivariant over that axis -- permute the score columns and
    #    the v rows by the same permutation and w @ v is unchanged. So the ring
    #    never has to be un-permuted, and this is why. If it were false, every
    #    step would owe a gather
    q = torch.randn(B, NKV, 1, HS)
    kk, vv = ring[0], ring[1]
    order = ring.key_positions.argsort()  # what un-permuting would cost

    def attend(k_: Tensor, v_: Tensor, k_pos: Tensor) -> Tensor:
        m = window_mask_from_positions(torch.tensor([ring.pos - 1]), k_pos, C)
        w = (q @ k_.transpose(-2, -1) * HS**-0.5).masked_fill(~m, -torch.inf)
        return w.softmax(dim=-1) @ v_

    scrambled = attend(kk, vv, ring.key_positions)
    sorted_ = attend(kk[:, :, order], vv[:, :, order], ring.key_positions[order])
    assert (scrambled - sorted_).abs().max() < 1e-6
    print("attending over the ring in slot order == in position order")

    # 5. the prefill case, where what is RETURNED and what is STORED come apart.
    #    hand it more tokens than it can hold: every query in that chunk still
    #    needs the keys the later ones will evict, so append hands back all of
    #    them and retains only the tail. the buffer's old contract -- "what you
    #    get is what I keep" -- is the thing rung 2 breaks
    r = RingKVCache((B, NKV, C, HS))
    pk, pv = r.append(k_all[:, :, :N], v_all[:, :, :N])
    assert pk.shape == (B, NKV, N, HS)  # returned: the whole chunk
    assert torch.equal(pk, k_all[:, :, :N]) and torch.equal(pv, v_all[:, :, :N])
    assert r.fill == C and r.pos == N  # retained: the last C only
    assert sorted(r.positions.tolist()) == list(range(N - C, N))
    #    and a chunk landing on top of an existing ring cats the two, so the
    #    early queries still see what they are entitled to. speculative
    #    decoding's verify step is exactly this shape
    r2 = RingKVCache((B, NKV, C, HS))
    for t in range(3):
        r2.append(*steps[t])
    ck, _ = r2.append(k_all[:, :, 3:6], v_all[:, :, 3:6])
    assert ck.shape == (B, NKV, 6, HS)
    assert torch.equal(ck, k_all[:, :, :6])
    assert sorted(r2.key_positions.tolist()) == list(range(6))
    print("a chunk gets back every key it needs, and the ring keeps the tail")

    # 6. rollback, and the constraint the roadmap wanted pinned. While the ring
    #    has not wrapped it behaves like the buffer -- one integer, no copy.
    #    Once it has, the entries a rollback wants back are the ones eviction
    #    overwrote, so it refuses. This is a real incompatibility between
    #    speculative decoding and eviction, not a missing feature
    r = RingKVCache((B, NKV, C, HS))
    for t in range(3):  # 3 < C, so nothing has been evicted
        r.append(*steps[t])
    r.rollback(1)
    assert r.pos == 1 and r.fill == 1
    rk, _ = r.append(*steps[7])  # the next write lands where the undone one was
    assert torch.equal(rk[:, :, 1], steps[7][0][:, :, 0])
    assert torch.equal(rk[:, :, 0], k_all[:, :, 0])

    r = RingKVCache((B, NKV, C, HS))
    for step in steps:  # N > C, so it has been round
        r.append(*step)
    r.rollback(r.pos)  # a no-op rollback is still fine
    try:
        r.rollback(r.pos - 1)
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "mutually constrained" in str(e)
    print("rollback works until the ring wraps, then refuses instead of lying")

    # 7. the oracle, and the reason rung 1 had to ship first. Decode through a
    #    ring of capacity C, and separately through a cache that keeps
    #    EVERYTHING with a window-C mask over it. Same window, same answers --
    #    so eviction is not an approximation of the windowed model, it is the
    #    windowed model with the unreachable keys not stored
    torch.manual_seed(1)
    qs = [torch.randn(B, NKV, 1, HS) for _ in range(N)]

    def decode(cache: KVCache, window: int | None) -> Tensor:
        outs = []
        for t, (k, v) in enumerate(steps):
            kk_, vv_ = cache.append(k, v)
            k_pos = getattr(cache, "key_positions", None)
            if k_pos is None:
                k_pos = torch.arange(kk_.size(2))
            m = window_mask_from_positions(torch.tensor([t]), k_pos, window)
            w = (qs[t] @ kk_.transpose(-2, -1) * HS**-0.5).masked_fill(~m, -torch.inf)
            outs.append(w.softmax(dim=-1) @ vv_)
        return torch.cat(outs, dim=2)

    evicted = decode(RingKVCache((B, NKV, C, HS)), C)
    kept = decode(KVCache(), C)
    assert (evicted - kept).abs().max() < 1e-6
    #    ...and both differ from the same decode without a window, which is the
    #    check that stops this passing for the boring reason
    unwindowed = decode(KVCache(), None)
    assert (kept - unwindowed).abs().max() > 1e-2
    print("evicting == masking: the ring is the windowed model, stored honestly")

    # 8. what it is worth, at a size that matters. A 32k-token chat with a 4096
    #    window: the cache stops tracking the conversation and starts tracking
    #    the window, and the saving is the ratio of the two
    L, NKVH, HSL = 32, 8, 128
    print(f"\n  KV cache, {L}L / {NKVH} kv heads / hs={HSL}, B=1, bf16")
    print(f"  {'context':>8}  {'full':>9}  {'ring W=4096':>12}")
    for ctx in (4096, 16384, 65536, 262144):
        full = 2 * L * ctx * NKVH * HSL * 2
        ring_b = 2 * L * min(ctx, 4096) * NKVH * HSL * 2
        print(f"  {ctx:>8}  {full / 2**30:>6.2f} GB  {ring_b / 2**30:>9.2f} GB")
    print(
        "\n  and it is FLAT past 4096 -- the run length stops being a memory question"
    )

    print("\nok")
