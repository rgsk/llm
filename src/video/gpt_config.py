from dataclasses import dataclass, fields

from block import FFN, Attention, Norm
from gpt import Position
from tokenizer import VOCAB_SIZE


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    """Exactly the arguments needed to rebuild a GPT. This is what a checkpoint stores."""

    vocab_size: int  # V
    block_size: int  # T
    n_embed: int  # E
    n_head: int  # nh
    n_layer: int
    dropout: float = 0.0  # 0 disables it entirely; 0.1-0.2 for a model that overfits
    attention: Attention = "mha"  # "fused" and "sdpa" share one set of keys
    n_kv_head: int | None = None  # "gqa" only; None means one k/v per query head
    norm: Norm = "layer"
    ffn: FFN = "dense"
    position: Position = "learned"
    window: int | None = None  # sliding window; None attends to the whole past
    ring: bool = False  # evict past the window instead of keeping everything
    sinks: int = 0  # ring slots pinned against eviction; needs rope

    def __post_init__(self):
        assert self.n_embed % self.n_head == 0, "n_embed must divide by n_head"
        if self.position == "sinusoidal":
            # the columns come in (sin, cos) pairs, so there has to be an even
            # number of them. Caught here rather than three frames deeper
            assert self.n_embed % 2 == 0, "sinusoidal positions need an even n_embed"
        if self.position == "rope":
            # rope rotates inside attention, so only an attention that knows
            # about it can carry it -- and it pairs dims per HEAD, not per model
            assert self.attention in ("sdpa", "gqa", "flex"), (
                "position='rope' needs attention='sdpa', 'gqa' or 'flex'"
            )
            assert (self.n_embed // self.n_head) % 2 == 0, (
                "rope needs an even head_size: the dims rotate in pairs"
            )
        if self.window is not None:
            # a window is a mask, and only the layers that build their own mask
            # from sliding_window_mask can narrow it
            assert self.attention in ("sdpa", "gqa"), (
                "window needs attention='sdpa' or 'gqa'"
            )
            assert self.window >= 1, "a window has to include the query itself"
        if self.ring:
            # only "gqa" holds a KVCache to wrap, and the ring's capacity is the
            # window, so there is no ring without one
            assert self.attention == "gqa", "ring needs attention='gqa'"
            assert self.window is not None, "a ring buffer's capacity IS the window"
        if self.sinks:
            # pinning slots only means something in a cache that evicts, and the
            # rank re-indexing that lets a run outlive block_size is a rope
            # operation -- a learned table has no rows past block_size to re-use
            assert self.ring, "sinks pin ring slots, so they need ring=True"
            assert self.position == "rope", (
                "sinks re-index positions inside the cache, which needs position='rope'"
            )
            assert self.sinks + self.window <= self.block_size, (
                f"sinks + window ({self.sinks} + {self.window}) is the cache, and "
                f"read-time rope indexes a block_size {self.block_size} table by rank"
            )
        if self.attention == "flex":
            # flex always rotates; a learned table on top would double-count
            # window/ring/sinks are already rejected above for anything but gqa
            assert self.position == "rope", "attention='flex' needs position='rope'"
        if self.n_kv_head is not None:
            assert self.attention in ("gqa", "flex"), (
                "n_kv_head needs attention='gqa' or 'flex'"
            )
            assert self.n_head % self.n_kv_head == 0, (
                "n_kv_head must divide n_head: every kv head serves the same group"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        """Build from a dict that may carry unrelated keys (e.g. a main.py checkpoint)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


small_cfg = GPTConfig(vocab_size=4096, block_size=32, n_embed=64, n_head=4, n_layer=3)

big_cfg = GPTConfig(
    vocab_size=4096,
    block_size=512,
    n_embed=512,
    n_head=8,
    n_layer=8,
    attention="sdpa",
    norm="rms",
    ffn="gated",
)


# --- TinyStories, its own 4096 vocab --------------------------------------
# big_cfg's shape, with rope and flex. Pretraining packs one continuous stream,
# so there is no block_mask and flex falls through to the same SDPA kernel --
# bit-identical output. What differs is the checkpoint: "sdpa" bakes a
# block_size rope table into every layer as buffers, flex threads the angles
# from GPT and stores none, so nothing caps a later extension.
# ~27M params, only 2.1M of it embedding -- the opposite balance to fineweb_smoke.
tinystories_cfg = GPTConfig(
    vocab_size=4097,  # 4096 merges + <|endoftext|>
    block_size=512,
    n_embed=512,
    n_head=8,
    n_layer=8,
    attention="flex",
    norm="rms",
    ffn="gated",
    position="rope",
)

# --- FineWeb-Edu, GPT-2's 50259 vocab -------------------------------------
# 20 minutes on a 4060 at B=8 x grad_accum 2: enough to watch loss fall and read
# the samples. 26.6M of its 29.9M params are the embedding -- the vocab IS the model
# at this size, which is the point the shape below fixes.
fineweb_smoke_cfg = GPTConfig(
    vocab_size=VOCAB_SIZE,
    block_size=512,
    n_embed=384,
    n_head=6,
    n_layer=6,
    attention="sdpa",
    norm="rms",
    ffn="gated",
    position="rope",
)

# GPT-2 small's shape. 123.6M params, embedding back down to 32%.
# 17-19 h per 1B tokens on a 4060 whatever the batch shape -- a RunPod job.
fineweb_cfg = GPTConfig(
    vocab_size=VOCAB_SIZE,
    block_size=1024,
    n_embed=768,
    n_head=12,
    n_layer=12,
    attention="sdpa",
    norm="rms",
    ffn="gated",
    position="rope",
)


if __name__ == "__main__":
    from dataclasses import asdict, replace

    from gpt import GPT

    # 1. field names match GPT's parameter names, so ** just works
    m = GPT(**asdict(small_cfg))
    assert m.block_size == small_cfg.block_size and m.n_layer == small_cfg.n_layer

    # 2. frozen: configs cannot drift after a model is built from one
    try:
        small_cfg.n_layer = 5
        raise SystemExit("should have failed")
    except Exception as e:
        assert "frozen" in str(e) or "cannot assign" in str(e)
    assert replace(small_cfg, n_layer=5).n_layer == 5  # this is how you vary one

    # 3. invalid geometry is rejected at construction, not at forward
    try:
        GPTConfig(vocab_size=10, block_size=8, n_embed=10, n_head=4, n_layer=1)
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "n_head" in str(e)

    # 3b. rope is the first position that constrains the attention: it rotates
    #     inside the layer, so only "sdpa" can carry it, and it pairs dims per
    #     head rather than per model -- an odd head_size has nothing to rotate
    assert replace(small_cfg, attention="sdpa", position="rope").position == "rope"
    for bad, msg in [
        ({"position": "rope"}, "attention='sdpa'"),  # default attention is mha
        (
            {"position": "rope", "attention": "sdpa", "n_embed": 6, "n_head": 2},
            "head_size",
        ),
    ]:
        try:
            replace(small_cfg, **bad)
            raise SystemExit(f"should have failed: {bad}")
        except AssertionError as e:
            assert msg in str(e), (bad, str(e))

    # 3c. a window is the second flag that constrains the attention, and for
    #     the same reason: it is a mask, and only "sdpa" and "gqa" build their
    #     own. W=0 is caught here too -- a query always attends to itself
    assert replace(small_cfg, attention="sdpa", window=8).window == 8
    for bad, msg in [
        ({"window": 8}, "attention='sdpa'"),  # default attention is mha
        ({"attention": "sdpa", "window": 0}, "include the query"),
    ]:
        try:
            replace(small_cfg, **bad)
            raise SystemExit(f"should have failed: {bad}")
        except AssertionError as e:
            assert msg in str(e), (bad, str(e))

    # 3d. sinks are the third, and the narrowest: they pin slots in a ring, and
    #     the re-indexing that makes them worth having is rope's
    assert (
        replace(
            small_cfg, attention="gqa", position="rope", window=8, ring=True, sinks=2
        ).sinks
        == 2
    )
    for bad, msg in [
        ({"sinks": 2}, "ring=True"),  # no ring to pin anything in
        (
            {"attention": "gqa", "window": 8, "ring": True, "sinks": 2},
            "position='rope'",
        ),
        (
            {
                "attention": "gqa",
                "position": "rope",
                "window": 31,
                "ring": True,
                "sinks": 2,
            },
            "by rank",  # 2 + 31 > block_size 32
        ),
    ]:
        try:
            replace(small_cfg, **bad)
            raise SystemExit(f"should have failed: {bad}")
        except AssertionError as e:
            assert msg in str(e), (bad, str(e))

    # 4. from_dict tolerates a main.py checkpoint's extra training keys
    saved = {
        "n_embed": 512,
        "n_head": 8,
        "n_layer": 8,
        "block_size": 512,
        "vocab_size": 4096,
        "dropout": 0.0,
        "batch_size": 64,
        "max_steps": 20000,
        "lr": 3e-4,
        "name": "big",
        "norm": "rms",
        "ffn": "gated",
        "activation": "silu",
        "use_wandb": True,
    }
    cfg = GPTConfig.from_dict(saved)
    assert "attention" not in saved
    assert cfg == replace(big_cfg, attention="mha")

    try:
        GPTConfig(**saved)
        raise SystemExit("should have failed")
    except TypeError as e:
        assert "unexpected keyword" in str(e)

    print("ok")
