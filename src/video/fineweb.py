"""FineWeb-Edu -> uint16 token shards on disk.

Streamed and written incrementally: the corpus is far larger than RAM at every
stage, so nothing but one shard's buffer is ever held.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from paths import DATA_ROOT
from tokenizer import ENDOFTEXT, VOCAB_SIZE, load_tokenizer

HF_DATASET = "HuggingFaceFW/fineweb-edu"
HF_CONFIG = "sample-10BT"  # 10B tokens of the full 1.3T
OUT_DIR = DATA_ROOT / "fineweb_edu"
SHARD_TOKENS = 100_000_000  # 200 MB per shard as uint16
DOC_BATCH = 1024

assert VOCAB_SIZE <= 65536, "uint16 shards need the vocab to fit in 16 bits"


class ShardWriter:
    """Appends ids to `{split}_00000.bin`, `_00001.bin`, ... of fixed token count."""

    def __init__(self, out_dir: Path, split: str, shard_tokens: int = SHARD_TOKENS):
        self.out_dir = out_dir
        self.split = split
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.n = 0  # tokens buffered
        self.total = 0
        self.paths: list[Path] = []

    def add(self, ids: list[int]) -> None:
        arr = np.asarray(ids, dtype=np.int64)
        assert arr.size == 0 or (arr.min() >= 0 and arr.max() < 65536)
        i = 0
        while i < arr.size:
            take = min(arr.size - i, self.buf.size - self.n)
            self.buf[self.n : self.n + take] = arr[i : i + take]
            self.n += take
            self.total += take
            i += take
            if self.n == self.buf.size:
                self._flush()

    def _flush(self) -> None:
        path = self.out_dir / f"{self.split}_{len(self.paths):05d}.bin"
        self.buf[: self.n].tofile(path)
        self.paths.append(path)
        self.n = 0

    def close(self) -> None:
        if self.n:
            self._flush()


def build(
    target_tokens: int = 1_000_000_000,
    val_tokens: int = 10_000_000,
    out_dir: Path = OUT_DIR,
    shard_tokens: int = SHARD_TOKENS,
) -> dict:
    """Stream documents, tokenize, write shards. val is the first `val_tokens`."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    eot = tok.specials[ENDOFTEXT]
    stream = load_dataset(HF_DATASET, name=HF_CONFIG, split="train", streaming=True)

    writers = {s: ShardWriter(out_dir, s, shard_tokens) for s in ("val", "train")}
    chars = {"val": 0, "train": 0}
    t0 = last = time.time()
    batch: list[str] = []

    def drain() -> None:
        # encode_ordinary: a document containing the literal "<|endoftext|>" must
        # stay text, or the corpus can forge a document boundary.
        # num_threads=1 on purpose: the stream delivers 1.1M tok/s and one thread
        # encodes 5.3M, so the build is download-bound and the pool buys nothing
        for text, ids in zip(batch, tok.encode_batch(batch, num_threads=1)):
            split = "val" if writers["val"].total < val_tokens else "train"
            writers[split].add([eot] + ids)  # eot leads a document, GPT-2 style
            chars[split] += len(text)
        batch.clear()

    for row in stream:
        batch.append(row["text"])
        if len(batch) == DOC_BATCH:
            drain()
            done = writers["val"].total + writers["train"].total
            if done >= target_tokens:
                break
            if time.time() - last > 30:
                last = time.time()
                rate = done / (last - t0) / 1e6
                eta = (target_tokens - done) / (rate * 1e6) / 60
                print(
                    f"{done:>13,} / {target_tokens:,} tokens  "
                    f"{rate:.2f}M tok/s  {eta:.0f} min left",
                    flush=True,
                )
    drain()
    for w in writers.values():
        w.close()

    meta = {"vocab_size": VOCAB_SIZE, "source": HF_DATASET, "config": HF_CONFIG}
    for split, w in writers.items():
        meta[split] = {
            "n_chars": chars[split],
            "n_tokens": w.total,
            "chars_per_token": chars[split] / max(w.total, 1),
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{json.dumps(meta, indent=2)}\ndone in {(time.time() - t0) / 60:.1f} min")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="download and tokenize")
    ap.add_argument("--tokens", type=float, default=1e9)
    ap.add_argument("--val-tokens", type=float, default=1e7)
    ap.add_argument("--shard-tokens", type=float, default=SHARD_TOKENS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.build:
        build(int(args.tokens), int(args.val_tokens), args.out, int(args.shard_tokens))
        # meta.json is written, so every byte is on disk. _exit skips interpreter
        # finalization, where the streaming reader and tiktoken tear each other
        # down -- SIGABRT or a hang, at random, long after the data is safe.
        sys.stdout.flush()
        os._exit(0)

    import tempfile

    from dataset import BinDataset

    rng = np.random.default_rng(0)

    # 1. a shard boundary is a buffer boundary, not a document boundary: ids
    #    written across many add() calls come back as one contiguous stream
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w = ShardWriter(tmp, "train", shard_tokens=100)
        docs = [
            rng.integers(0, VOCAB_SIZE, size=int(n)).tolist()
            for n in rng.integers(1, 40, 30)
        ]
        for doc in docs:
            w.add(doc)
        w.close()
        flat = [i for doc in docs for i in doc]
        assert w.total == len(flat)
        assert len(w.paths) == -(-len(flat) // 100)  # ceil
        ds = BinDataset("train", tmp)
        assert ds.n_tokens == len(flat)
        assert ds.tokens(0, len(flat)).tolist() == flat
        ds.close()
    print(
        f"{len(docs)} docs -> {len(w.paths)} shards -> one stream of {w.total} tokens"
    )

    # 2. an add() larger than a whole shard still lands exactly once
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        w = ShardWriter(tmp, "val", shard_tokens=10)
        w.add(list(range(35)))
        w.close()
        assert w.total == 35 and len(w.paths) == 4
        assert BinDataset("val", tmp).tokens(0, 35).tolist() == list(range(35))
    print("an add() spanning four shards splits cleanly")

    # 3. a token that does not fit uint16 fails loudly rather than wrapping
    with tempfile.TemporaryDirectory() as d:
        try:
            ShardWriter(Path(d), "train", 10).add([65536])
            raise SystemExit("should have failed")
        except AssertionError:
            pass

    # 4. the real shards, if built: meta matches what is on disk
    if (OUT_DIR / "meta.json").exists():
        from dataset import load_meta

        meta = load_meta(OUT_DIR)
        for split in ("train", "val"):
            ds = BinDataset(split, OUT_DIR)
            assert ds.n_tokens == meta[split]["n_tokens"]
            print(
                f"{split}: {ds.n_tokens:,} tokens over {len(ds.paths)} shards, "
                f"{meta[split]['chars_per_token']:.2f} chars/token"
            )
            ds.close()
    else:
        print(f"\nno shards at {OUT_DIR}; build them with")
        print("  uv run python fineweb.py --build --tokens 1e9")

    print("ok")
