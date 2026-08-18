from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import ExitStack
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from huggingface_hub import hf_hub_download

from tokenizer import BPETokenizer
from utils import repo_root

ROOT = repo_root()
CACHE = ROOT / "artifacts" / "data" / "tinystories_instruct_full"

DATASET = "roneneldan/TinyStoriesInstruct"
FILES = {
    "train": "TinyStories-Instruct-train.txt",  # 2.66 GB, 2,476,334 records
    "valid": "TinyStories-Instruct-valid.txt",  # 26.9 MB, 25,026 records
}
FIELDS = ("Random sentence:", "Features:", "Words:", "Summary:", "Story:")
SEP = "<|endoftext|>"


def source_path(split: Literal["train", "valid"]) -> Path:
    """Download-once, cached in ~/.cache/huggingface."""
    return Path(hf_hub_download(DATASET, FILES[split], repo_type="dataset"))


def parse(rec: str) -> dict[str, str]:
    """One record's text -> {field name: value}, in source order."""
    fields: dict[str, list[str]] = {}
    cur = None
    for line in rec.split("\n"):
        hit = next((f for f in FIELDS if line.startswith(f)), None)
        if hit:
            cur = hit[:-1]
            fields[cur] = [line[len(hit) :].strip()]
        elif cur:
            fields[cur].append(line.strip())
    return {k: "\n".join(v).strip() for k, v in fields.items()}


def usable(f: dict[str, str]) -> bool:
    """Need a story to learn, and at least one field to condition on."""
    return bool(f.get("Story")) and any(f[k] for k in f if k != "Story")


def iter_records(path: Path, chunk_bytes: int = 1 << 26) -> Iterator[dict[str, str]]:
    """Stream a source file -> parsed records, without holding the file in RAM.

    Reading it whole is what stops working at 2.66 GB: the str, the list the
    split produces, and 2.35M dicts are ~20 GB together. Here only one 64 MB
    chunk plus the trailing partial record are ever resident.

    The remainder carry is the whole trick -- a chunk boundary lands mid-record
    99% of the time, so the last piece of every split is held back and prefixed
    onto the next chunk instead of being parsed as if it were complete.
    """
    rest = ""
    with open(path, encoding="utf-8") as f:
        while chunk := f.read(chunk_bytes):
            parts = (rest + chunk).split(SEP)
            rest = parts.pop()
            for p in parts:
                if (rec := parse(p.strip())) and usable(rec):
                    yield rec
    if (rec := parse(rest.strip())) and usable(rec):
        yield rec


@cache
def tokenizer() -> BPETokenizer:
    """Lazy so parsing-only work (and the notebook) never pays for it."""
    return BPETokenizer.load(str(ROOT / "artifacts" / "tokenizer" / "bpe_ts_4096.json"))


def render(f: dict[str, str]) -> tuple[str, str]:
    """Fields -> (prompt, completion). Story is forced last; other fields keep
    their source order, which varies per record on purpose -- the corpus shows
    every permutation (Words/Features/Summary and Features/Words/Summary are
    8.1% each), so preserving order is what teaches order-independence."""
    prompt = "".join(f"{k}: {f[k]}\n" for k in f if k != "Story") + "Story.\n\n"
    return prompt, f["Story"] + "\n<|endoftext|>\n"


def encode_example(f: dict[str, str]) -> tuple[list[int], list[bool]]:
    """-> (token ids, is_completion flag per token). Tokenized separately so the
    prompt/completion boundary is exact, not inferred."""
    tok = tokenizer()
    p, c = render(f)
    ip, ic = tok.encode(p), tok.encode(c)
    return ip + ic, [False] * len(ip) + [True] * len(ic)


def shuffled(
    it: Iterator[dict[str, str]], size: int, rng: np.random.Generator
) -> Iterator[dict[str, str]]:
    """Near-global shuffle in bounded memory: hold `size` records, emit a random
    one, refill. A true permutation would need all 2.5M records resident.

    Only affects which story sits in a window as *context* for another -- batches
    already sample example starts uniformly across the whole stream. It is here
    as cheap insurance against any hidden ordering in the source file.
    """
    buf: list[dict[str, str]] = []
    for rec in it:
        buf.append(rec)
        if len(buf) >= size:
            j = int(rng.integers(size))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()
    for j in rng.permutation(len(buf)):
        yield buf[j]


def _pack_split(
    recs: Iterator[dict[str, str]], split: str, flush_every: int = 1 << 20
) -> dict:
    """Records -> three .bin files, written as we go.

    The old in-memory pack cannot survive this scale: 621M ids as a Python list
    is ~25 GB (values above 255 are not interned, so each is a 28-byte object
    plus an 8-byte pointer). Buffering ~1M tokens and calling .tofile keeps the
    resident set flat while producing the identical byte stream.

    `starts` is written here rather than derived from the mask at load time.
    We know each example's offset exactly at pack time; recovering it later
    would mean a 621 MB scan on every import.
    """
    paths = {k: CACHE / f"{split}.{k}.bin" for k in ("ids", "mask", "starts")}
    ids_buf: list[int] = []
    mask_buf: list[bool] = []
    start_buf: list[int] = []
    n_tokens = n_recs = n_comp = 0
    t0 = time.perf_counter()

    with ExitStack() as stack:
        fh = {k: stack.enter_context(open(p, "wb")) for k, p in paths.items()}

        def flush() -> None:
            np.array(ids_buf, dtype=np.uint16).tofile(fh["ids"])
            np.array(mask_buf, dtype=bool).tofile(fh["mask"])
            # int64, not uint64: these get added to Python ints when slicing
            # windows, and signed keeps that arithmetic boring.
            np.array(start_buf, dtype=np.int64).tofile(fh["starts"])
            ids_buf.clear()
            mask_buf.clear()
            start_buf.clear()

        for rec in recs:
            ids, is_c = encode_example(rec)
            start_buf.append(n_tokens)
            ids_buf += ids
            mask_buf += is_c
            n_tokens += len(ids)
            n_comp += sum(is_c)
            n_recs += 1
            if len(ids_buf) >= flush_every:
                flush()
                if n_recs % 200_000 < 5_000:
                    dt = time.perf_counter() - t0
                    print(
                        f"  {split}: {n_recs:>9,} recs  {n_tokens / 1e6:>6.1f}M tok"
                        f"  {n_tokens / dt / 1e6:.2f}M tok/s  {dt:.0f}s"
                    )
        flush()

    assert tokenizer().vocab_size < 65536, "ids no longer fit in uint16"
    return {
        "n_records": n_recs,
        "n_tokens": n_tokens,
        "completion_frac": n_comp / n_tokens,
        "pack_seconds": round(time.perf_counter() - t0, 1),
    }


def build_or_load(
    seed: int = 1337, shuffle_buffer: int = 100_000, rebuild: bool = False
) -> dict[str, np.ndarray]:
    """Tokenize + pack once, cache to disk, memmap thereafter.

    Splits come from the dataset's own two files -- train from the train file,
    val from the valid file. The previous tail-slice of a single file existed
    only because the train file had not been downloaded; using the real split
    means no window can straddle the boundary by construction, and the two are
    distributionally matched (field-combination frequencies agree within 0.7pp).
    """
    names = {
        f"{s}_{k}": CACHE / f"{s}.{k}.bin"
        for s in ("train", "val")
        for k in ("ids", "mask", "starts")
    }
    meta_path = CACHE / "meta.json"

    if rebuild or not (meta_path.exists() and all(p.exists() for p in names.values())):
        CACHE.mkdir(parents=True, exist_ok=True)
        meta: dict = {
            "seed": seed,
            "shuffle_buffer": shuffle_buffer,
            "vocab_size": tokenizer().vocab_size,
        }
        for split, source in (("train", "train"), ("val", "valid")):
            path = source_path(source)  # type: ignore[arg-type]
            rng = np.random.default_rng(seed)
            recs = shuffled(iter_records(path), shuffle_buffer, rng)
            meta[split] = {"source": path.name} | _pack_split(recs, split)
            print(split, meta[split])
        meta_path.write_text(json.dumps(meta, indent=2))

    dtypes = {"ids": np.uint16, "mask": bool, "starts": np.int64}
    return {
        k: np.memmap(p, dtype=dtypes[k.split("_")[1]], mode="r")
        for k, p in names.items()
    }


if __name__ == "__main__":
    packed = build_or_load()
    for k, v in packed.items():
        print(f"{k:>13}  {v.dtype}  {v.shape[0]:,}")
