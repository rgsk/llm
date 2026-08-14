import json

import numpy as np
from datasets import load_dataset

from tokenizer import BPETokenizer
from utils import repo_root

ROOT = repo_root()


def download_ts():
    items = [
        ("train", "tinystories_train.txt"),
        ("validation", "tinystories_validation.txt"),
    ]
    paths = [ROOT / "data" / fname for _, fname in items]
    if all(p.exists() for p in paths):
        return
    ds = load_dataset(
        "roneneldan/TinyStories"
    )  # ~2.1M stories, first run downloads ~1.9GB
    SEP = "\n<|endoftext|>\n"
    for split, fname in items:
        with open(ROOT / "data" / fname, "w", encoding="utf-8") as f:
            f.writelines(row["text"].strip() + SEP for row in ds[split])  # type: ignore


def load_or_build_bpe():
    vocab_size = 4096
    dir = ROOT / "artifacts" / "tokenizer"
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / f"bpe_ts_{vocab_size}.json"
    if path.exists():
        return BPETokenizer.load(str(path))
    with open(ROOT / "data" / "tinystories_train.txt", encoding="utf-8") as f:
        sample = f.read(20_000_000)
        tok = BPETokenizer()
        tok.train(sample, vocab_size)
        tok.save(str(path))
        return tok


def line_batches(path, size=10_000):
    batch = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            batch.append(line)
            if len(batch) == size:
                yield "".join(batch)
                batch = []
    if batch:
        yield "".join(batch)


def save_tokens():
    out = ROOT / "artifacts" / "data" / "tinystories"
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and all(
        (out / f"{s}.bin").exists() for s in ("train", "val")
    ):
        return json.loads(meta_path.read_text())

    train_path = ROOT / "data" / "tinystories_train.txt"
    validation_path = ROOT / "data" / "tinystories_validation.txt"
    tok = load_or_build_bpe()
    assert tok.vocab_size < 65536
    meta: dict = {"vocab_size": tok.vocab_size}
    for split, src in [("train", train_path), ("val", validation_path)]:
        n_chars = n_tokens = 0
        with open(out / f"{split}.bin", "wb") as f:
            for b in line_batches(src):
                ids = tok.encode(b)
                np.array(ids, dtype=np.uint16).tofile(f)
                n_chars += len(b)
                n_tokens += len(ids)
        meta[split] = {
            "n_chars": n_chars,
            "n_tokens": n_tokens,
            "chars_per_token": n_chars / n_tokens,
        }
        print(split, meta[split])

    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    download_ts()
    save_tokens()
