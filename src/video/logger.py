"""Weights & Biases behind one small object, so the training loop never has to
ask whether anything is being recorded."""

from pathlib import Path

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
        self.base = name  # artifact name: every run of "big" versions one artifact
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

    def log_checkpoint(self, path: Path) -> None:
        """Upload a file as a model artifact, so it outlives the machine that wrote
        it. finish() blocks until the upload is done."""
        if self.enabled:
            art = wandb.Artifact(self.base, type="model")
            art.add_file(str(path))
            wandb.log_artifact(art)

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
        # a checkpoint goes up as a model artifact named after the run's base name
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "test.pt"
            f.write_bytes(b"weights")
            run.log_checkpoint(f)

    # 3. the context manager closed it
    assert wandb.run is None
    print("ok")
