"""Weights & Biases behind one small object, so the training loop never has to
ask whether anything is being recorded."""

from checkpoint import timestamp

import wandb


class Run:
    """One training run's log. `enabled=False` makes every method a no-op, so
    smoke tests and throwaway runs never touch the network."""

    def __init__(
        self,
        name: str,
        config: dict,
        project: str = "llm",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.name = f"{name}_{timestamp()}"
        if enabled:
            wandb.init(
                project=project,
                name=self.name,
                config=config,
                settings=wandb.Settings(silent=True),
            )

    def log(self, row: dict, step: int) -> None:
        """One row per eval. `step` is the x-axis, so it must never go backwards."""
        if self.enabled:
            wandb.log({k: v for k, v in row.items() if k != "step"}, step=step)

    def summary(self, **kw: object) -> None:
        """Single numbers for the run as a whole -- what the runs table sorts on."""
        if self.enabled:
            wandb.summary.update(kw)

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()


if __name__ == "__main__":
    # 1. disabled is a complete no-op: no init, no network, no run created
    off = Run("test", {"a": 1}, enabled=False)
    off.log({"loss": 1.0}, step=0)
    off.summary(best=1.0)
    off.finish()
    assert wandb.run is None

    # 2. enabled, offline: a real run object exists and carries the config
    import os

    os.environ["WANDB_MODE"] = "offline"
    with Run("test", {"lr": 3e-4, "name": "test"}) as run:
        assert wandb.run is not None
        assert wandb.run.config["lr"] == 3e-4
        assert run.name.startswith("test_2")
        for it in range(3):
            run.log({"step": it, "val": 1.0 / (it + 1)}, step=it)
        run.summary(total_time_s=12.5)
        assert wandb.summary["total_time_s"] == 12.5
        # "step" was stripped from every row: it is the axis, not a metric
        assert "step" not in wandb.summary.keys()

    # 3. the context manager closed it
    assert wandb.run is None
    print("ok")
