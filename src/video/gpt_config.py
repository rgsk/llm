from dataclasses import dataclass, fields

from block import FFN, Attention, Norm
from gpt import Position


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

    def __post_init__(self):
        assert self.n_embed % self.n_head == 0, "n_embed must divide by n_head"
        if self.position == "sinusoidal":
            # the columns come in (sin, cos) pairs, so there has to be an even
            # number of them. Caught here rather than three frames deeper
            assert self.n_embed % 2 == 0, "sinusoidal positions need an even n_embed"
        if self.position == "rope":
            # rope rotates inside attention, so only an attention that knows
            # about it can carry it -- and it pairs dims per HEAD, not per model
            assert self.attention in ("sdpa", "gqa"), (
                "position='rope' needs attention='sdpa' or 'gqa'"
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
        if self.n_kv_head is not None:
            assert self.attention == "gqa", "n_kv_head needs attention='gqa'"
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
