import os
from pathlib import Path


def repo_root() -> Path:
    """Walk up to the folder holding pyproject.toml, so paths work no matter
    where python is launched from."""
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").exists():
            return d
    raise FileNotFoundError("no pyproject.toml above this file")


ROOT = repo_root()
DATA_ROOT = ROOT / "artifacts" / "data"
# which prepared dataset train.py reads; VIDEO_DATASET=fineweb_edu switches it
DATASET = os.getenv("VIDEO_DATASET", "tinystories_eot")
DATA_DIR = DATA_ROOT / DATASET
CKPT_DIR = ROOT / "artifacts" / "checkpoints"
TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
LOG_DIR = ROOT / "artifacts" / "logs"


if __name__ == "__main__":
    print("ROOT      ", ROOT)
    print("DATASET   ", DATASET)
    print("DATA_DIR  ", DATA_DIR, DATA_DIR.exists())
    print("CKPT_DIR  ", CKPT_DIR, CKPT_DIR.exists())
    print("TOKENIZER ", TOKENIZER_DIR, TOKENIZER_DIR.exists())
    print("LOG_DIR   ", LOG_DIR, LOG_DIR.exists())
