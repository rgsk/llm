from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Everything about *running* a training job. Never reaches the model."""

    batch_size: int  # B
    max_steps: int
    lr: float
    min_lr: float  # usually lr / 10
    warmup_steps: int  # ~2% of max_steps
    eval_interval: int
    eval_iters: int
    grad_accum_steps: int = 1
    amp: bool = True  # bf16 autocast on cuda; ~1.7x here, and no accuracy cost
    seed: int = 1337
    name: str = "video"


small_train = TrainConfig(
    batch_size=32,
    max_steps=200,
    lr=1e-2,
    min_lr=1e-3,
    warmup_steps=20,
    eval_interval=100,
    eval_iters=100,
    name="scratch",
)

big_train = TrainConfig(
    batch_size=64,
    max_steps=20000,
    lr=3e-4,
    min_lr=3e-5,
    warmup_steps=400,
    eval_interval=1000,
    eval_iters=100,
    name="big",
)


if __name__ == "__main__":
    from dataclasses import asdict, replace

    from gpt_config import GPTConfig

    # 1. no field overlaps GPTConfig -- the two configs are disjoint concerns
    assert not (
        set(asdict(small_train))
        & set(
            asdict(
                GPTConfig(vocab_size=1, block_size=1, n_embed=1, n_head=1, n_layer=1)
            )
        )
    )

    # 2. sane relationships hold in the presets
    for cfg in (small_train, big_train):
        assert cfg.min_lr < cfg.lr
        assert cfg.warmup_steps < cfg.max_steps
        assert cfg.eval_interval <= cfg.max_steps

    # 3. frozen, varied via replace
    assert replace(big_train, max_steps=100).max_steps == 100
    assert big_train.max_steps == 20000

    print("ok")
