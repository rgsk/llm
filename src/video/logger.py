"""Weights & Biases behind one small object, so the training loop never has to
ask whether anything is being recorded.

`online=False` runs wandb in OFFLINE mode, not disabled: disabled writes nothing at
all (measured), offline writes a real run that `wandb sync` can upload later. An
offline run is a binary .wandb file though, so every run also appends
`artifacts/logs/<name>.jsonl` -- the copy you can read without syncing."""

import json
import math
from pathlib import Path

import wandb
from checkpoint import timestamp
from paths import LOG_DIR


class Run:
    """One training run's log. `online=False` still records everything, locally."""

    def __init__(
        self,
        name: str,
        config: dict,
        project: str = "llm",
        online: bool = True,
    ) -> None:
        self.online = online
        self.base = name  # artifact name: every run of "big" versions one artifact
        self.name = f"{name}_{timestamp()}"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{self.name}.jsonl"
        self._write({"event": "config", **config})
        wandb.init(
            project=project,
            name=self.name,
            config=config,
            settings=wandb.Settings(silent=True),
            mode=None if online else "offline",
        )

    def _write(self, obj: dict) -> None:
        # nan/inf are valid to python's json but not to jq, and a readable log that
        # jq chokes on is not readable
        clean = {
            k: None if isinstance(v, float) and not math.isfinite(v) else v
            for k, v in obj.items()
        }
        with self.path.open("a") as f:
            f.write(json.dumps(clean, default=str, allow_nan=False) + "\n")

    def log(self, row: dict, step: int) -> None:
        """One row per eval. `step` is the x-axis, so it must never go backwards."""
        self._write({"event": "log", "step": step, **row})
        wandb.log({k: v for k, v in row.items() if k != "step"}, step=step)

    def summary(self, **kw: object) -> None:
        """Single numbers for the run as a whole -- what the runs table sorts on."""
        self._write({"event": "summary", **kw})
        wandb.summary.update(kw)

    def log_checkpoint(self, path: Path) -> None:
        """Upload a file as a model artifact, so it outlives the machine that wrote
        it. finish() blocks until the upload is done. Online only: offline would
        copy the whole checkpoint into the run dir and upload it nowhere."""
        if self.online:
            art = wandb.Artifact(self.base, type="model")
            art.add_file(str(path))
            wandb.log_artifact(art)

    def finish(self) -> None:
        wandb.finish()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()


if __name__ == "__main__":
    # 1. online=False is offline, not disabled: a real wandb run exists, and the
    #    readable copy lands in artifacts/logs
    off = Run("test", {"a": 1}, online=False)
    assert wandb.run is not None and wandb.run.offline
    off.log({"loss": 1.0}, step=0)
    off.summary(best=1.0)
    off.finish()
    assert wandb.run is None
    rows = [json.loads(ln) for ln in off.path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["config", "log", "summary"]
    assert rows[1]["loss"] == 1.0 and rows[1]["step"] == 0
    assert rows[2]["best"] == 1.0
    off.path.unlink()

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
