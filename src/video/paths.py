from pathlib import Path


def repo_root() -> Path:
    """Walk up to the folder holding pyproject.toml, so paths work no matter
    where python is launched from."""
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").exists():
            return d
    raise FileNotFoundError("no pyproject.toml above this file")


ROOT = repo_root()
DATA_DIR = ROOT / "artifacts" / "data" / "tinystories"
CKPT_DIR = ROOT / "artifacts" / "checkpoints"


if __name__ == "__main__":
    print("ROOT     ", ROOT)
    print("DATA_DIR ", DATA_DIR, DATA_DIR.exists())
    print("CKPT_DIR ", CKPT_DIR, CKPT_DIR.exists())
