from dataclasses import dataclass, fields

from block import FFN, Attention, Norm


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
    norm: Norm = "layer"
    ffn: FFN = "dense"

    def __post_init__(self):
        assert self.n_embed % self.n_head == 0, "n_embed must divide by n_head"

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        """Build from a dict that may carry unrelated keys (e.g. a main.py checkpoint)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


small_cfg = GPTConfig(vocab_size=4096, block_size=32, n_embed=64, n_head=4, n_layer=3)
big_cfg = GPTConfig(vocab_size=4096, block_size=512, n_embed=512, n_head=8, n_layer=8)


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
    assert cfg == replace(big_cfg, norm="rms", ffn="gated")

    try:
        GPTConfig(**saved)
        raise SystemExit("should have failed")
    except TypeError as e:
        assert "unexpected keyword" in str(e)

    print("ok")
