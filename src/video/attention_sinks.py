"""Rung 3: a policy about which slots never get evicted.

Rung 1 stopped the model looking at old keys, rung 2 stopped storing them. Both
throw away position 0, and StreamingLLM (Xiao et al., 2024) measured what that
costs: the first few tokens absorb attention mass that has nowhere else to go,
and evicting them redistributes it onto real keys. The fix is a policy -- pin
the first S slots, ring the rest.

Pinning is four lines. What it drags in is positions. A run that never ends
runs out of rope table, so a cached key's position becomes its RANK among what
is cached rather than where it sat in the text. Ranks never exceed capacity, so
the table stops being the thing that bounds a run.

The price is that keys can no longer be rotated once on the way in, because the
position they are rotated by changes every time something is evicted. They go
in raw and are rotated on the way out -- the whole window, every step. That is
rung 3's cost, and it is the point.

Why this file does not show the quality win: an untrained model has no sink
behaviour to lose. The mass measurement belongs on a checkpoint, in the
notebook. What is testable here is the mechanism -- that the pinned slots
survive, that re-indexing preserves distances inside the window and only lies
about the gap, and that a sink decode matches an oracle that keeps everything.
"""

import torch
from torch import Tensor

from ring_kv_cache import RingKVCache
from sliding_window import window_mask_from_positions


def sink_mask_from_positions(
    q_pos: Tensor, k_pos: Tensor, window: int | None = None, sinks: int = 0
) -> Tensor:
    """window_mask_from_positions with the first `sinks` positions always visible.

    Still causal: a sink is only visible to queries at or after it, which is
    what keeps a prefill's early rows honest.
    """
    visible = window_mask_from_positions(q_pos, k_pos, window)
    if sinks:
        visible = visible | ((k_pos < sinks) & (k_pos <= q_pos.unsqueeze(1)))
    return visible


class SinkKVCache(RingKVCache):
    """A ring whose first `sinks` slots are pinned; the cursor wraps over the rest.

    Capacity is sinks + window. Position p < S lives in slot p forever; p >= S
    lives in slot S + (p - S) % window.
    """

    def __init__(self, shape: tuple[int, int, int, int], sinks: int):
        super().__init__(shape)
        assert 0 <= sinks < shape[2], "sinks must leave room for a window to ring over"
        self.sinks = sinks

    @property
    def window(self) -> int:
        """Slots the cursor may use. Capacity minus the pinned ones."""
        return self.capacity - self.sinks

    def _slots(self, pos: Tensor) -> Tensor:
        return torch.where(
            pos < self.sinks, pos, self.sinks + (pos - self.sinks) % self.window
        )

    def _write(self, k: Tensor, v: Tensor, new_pos: Tensor) -> None:
        n, S = new_pos.size(0), self.sinks
        n_sink = int((new_pos < S).sum())  # a prefix: new_pos is increasing
        keep = min(n - n_sink, self.window)  # the rest overwrite each other
        sel = torch.cat(
            [
                torch.arange(n_sink, device=new_pos.device),
                torch.arange(n - keep, n, device=new_pos.device),
            ]
        )
        slots = self._slots(new_pos[sel])
        self.k[:, :, slots] = k[:, :, sel]
        self.v[:, :, slots] = v[:, :, sel]
        self.positions[slots] = new_pos[sel]
        self.pos += n

    def read_positions(self, T_q: int) -> tuple[Tensor, Tensor]:
        """Re-indexed (query, key) positions to rotate by, as rope table rows.

        A key's position becomes its rank in absolute-position order, so the
        highest row ever touched is capacity-1 however long the run is.
        Distances inside the window survive intact; the gap eviction left is
        what closes, which is the same approximation the mask already made.
        """
        kp = self.key_positions
        n = kp.size(0)
        # queries are the T_q newest positions, so they take the top ranks
        return torch.arange(n - T_q, n, device=kp.device), kp.argsort().argsort()

    def __repr__(self) -> str:
        return f"SinkKVCache(sinks={self.sinks}, window={self.window}, pos={self.pos})"


if __name__ == "__main__":
    from rope import apply_rope, rope_tables

    torch.manual_seed(0)
    B, NKV, HS = 2, 2, 4
    S, W = 2, 4
    C = S + W
    N = 17  # well past capacity, so the ring goes round three times
    steps = [(torch.randn(B, NKV, 1, HS), torch.randn(B, NKV, 1, HS)) for _ in range(N)]
    k_all = torch.cat([k for k, _ in steps], dim=2)
    v_all = torch.cat([v for _, v in steps], dim=2)

    def held(pos: int) -> list[int]:
        """The positions a sink cache should hold after `pos` tokens."""
        return sorted(set(range(min(S, pos))) | set(range(max(0, pos - W), pos)))

    # 1. sinks=0 is exactly a RingKVCache -- the policy is the only difference,
    #    so this pins that nothing else drifted
    sink0, ring = SinkKVCache((B, NKV, W, HS), 0), RingKVCache((B, NKV, W, HS))
    for step in steps:
        a, _ = sink0.append(*step)
        b, _ = ring.append(*step)
        assert torch.equal(a, b) and sink0.pos == ring.pos
    print("sinks=0 is byte-identical to the plain ring")

    # 2. the policy itself: the first S positions are still there after N tokens,
    #    in their original slots, and the rest is the last W
    c = SinkKVCache((B, NKV, C, HS), S)
    for t, step in enumerate(steps, start=1):
        c.append(*step)
        assert sorted(c.key_positions.tolist()) == held(t), t
    for p in range(S):
        assert c.positions[p].item() == p
        assert torch.equal(c.k[:, :, p], k_all[:, :, p])
    assert c.pos == N and c.fill == C
    print(f"after {N} tokens: sinks {list(range(S))} + window {held(N)[S:]}")

    #    and a plain ring of the same capacity has long since lost them
    r = RingKVCache((B, NKV, C, HS))
    for step in steps:
        r.append(*step)
    assert 0 not in r.key_positions.tolist()

    # 3. re-indexing. ranks are a permutation of 0..fill-1, so the highest rope
    #    row touched is capacity-1, however far into the run the cache is
    q_pos, k_pos = c.read_positions(1)
    assert sorted(k_pos.tolist()) == list(range(C))
    assert int(k_pos.max()) == C - 1 < c.pos
    assert q_pos.tolist() == [C - 1]  # the query is the newest key

    # 4. what re-indexing preserves and what it fakes. Inside the window every
    #    query-key distance is exact; across the gap it is compressed, and that
    #    compression IS the approximation
    abs_pos = c.key_positions
    order = abs_pos.argsort()
    for s in order.tolist():
        d_abs, d_rank = c.pos - 1 - int(abs_pos[s]), C - 1 - int(k_pos[s])
        if int(abs_pos[s]) >= S:  # a window key
            assert d_abs == d_rank
        else:  # a sink: told it is nearby when it is not
            assert d_rank < d_abs
    far = c.pos - 1 - 0
    print(
        f"sink at distance {far} is rotated as if it were {C - 1 - int(k_pos[order[0]])}"
    )

    # 5. the mask. sink columns are visible from everywhere past them, the band
    #    is the window, and an early query still cannot see a later sink
    q_abs = torch.arange(N)
    m = sink_mask_from_positions(q_abs, torch.arange(N), W, S)
    assert m[:, :S].tril(diagonal=0).equal(m[:, :S])  # causal, sinks included
    assert m[N - 1].nonzero().flatten().tolist() == held(N)
    assert not m[0, 1]  # position 0 cannot see sink 1
    assert torch.equal(
        sink_mask_from_positions(q_abs, torch.arange(N), W, 0),
        window_mask_from_positions(q_abs, torch.arange(N), W),
    )

    # 6. prefill: a chunk bigger than the cache gets back every key it needs and
    #    leaves behind sinks + the tail, which is the rung-2 split plus a policy
    pre = SinkKVCache((B, NKV, C, HS), S)
    pk, _ = pre.append(k_all, v_all)
    assert pk.shape == (B, NKV, N, HS) and torch.equal(pk, k_all)
    assert sorted(pre.key_positions.tolist()) == list(range(N))  # returned
    assert sorted(pre.positions.tolist()) == held(N)  # retained
    #    and the same chunk fed one token at a time lands on the same cache
    assert sorted(pre.positions.tolist()) == sorted(c.positions.tolist())
    print("prefill returns the whole chunk and retains sinks + the tail")

    # 7. THE test: a sink decode reproduces an oracle that stores everything and
    #    does the re-indexing by hand. If read_positions or the slot arithmetic
    #    were wrong, only this would catch it
    torch.manual_seed(1)
    qs = [torch.randn(B, NKV, 1, HS) for _ in range(N)]
    cos_t, sin_t = rope_tables(C, HS)

    def attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        w = q @ k.transpose(-2, -1) * HS**-0.5
        return w.softmax(dim=-1) @ v

    def oracle(t: int) -> Tensor:
        """Keep every key; pick the visible ones; rotate by rank."""
        vis = held(t + 1)
        rank = torch.arange(len(vis))
        k = apply_rope(k_all[:, :, vis], cos_t[rank], sin_t[rank])
        q = apply_rope(qs[t], cos_t[rank[-1:]], sin_t[rank[-1:]])
        return attend(q, k, v_all[:, :, vis])

    cache = SinkKVCache((B, NKV, C, HS), S)
    for t in range(N):
        k, v = cache.append(*steps[t])  # raw keys in, rotated on the way out
        qp, kp_ = cache.read_positions(1)
        kr = apply_rope(k, cos_t[kp_], sin_t[kp_])
        qr = apply_rope(qs[t], cos_t[qp], sin_t[qp])
        assert (attend(qr, kr, v) - oracle(t)).abs().max() < 1e-6, t
    print("sink decode == the keep-everything oracle, every step")

    # 8. what it costs. write-time rotation touched T keys per step; read-time
    #    touches the whole cache, forever
    print(f"\n  rope work per decode step, cache = {S} sinks + {W} window")
    print(f"    rungs 1-2 (write time): {1:>4} key   rotated once")
    print(f"    rung 3   (read time):  {C:>4} keys  rotated every step")
    print(f"    ...and in exchange the rope table stops bounding the run at {C}")

    print("\nok")
