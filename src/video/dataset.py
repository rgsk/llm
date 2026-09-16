import json
import mmap
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor

from paths import DATA_DIR


class SplitMeta(TypedDict):
    n_chars: int
    n_tokens: int
    chars_per_token: float


class Meta(TypedDict):
    vocab_size: int
    train: SplitMeta
    val: SplitMeta


def load_meta(data_dir: Path = DATA_DIR) -> Meta:
    return json.loads((data_dir / "meta.json").read_text())


def shard_paths(data_dir: Path, split: str) -> list[Path]:
    """`train.bin`, or `train_00000.bin`, `train_00001.bin`, ... in name order."""
    one = data_dir / f"{split}.bin"
    if one.exists():
        return [one]
    paths = sorted(data_dir.glob(f"{split}_*.bin"))
    assert paths, f"no shards for split {split!r} in {data_dir}"
    return paths


meta: Meta = load_meta()


class BinDataset:
    """One or more .bin files of uint16 token ids, memory-mapped and addressed as
    a single stream: the OS pages them in on demand, so a 2 GB split costs no RAM
    until it is read."""

    def __init__(self, split: str, data_dir: Path = DATA_DIR):
        self.paths = shard_paths(data_dir, split)
        self.files = [p.open("rb") for p in self.paths]
        self.mms = [
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) for f in self.files
        ]
        self.sizes = [len(m) // 2 for m in self.mms]  # uint16 -> 2 bytes per token
        self.offsets = list(accumulate(self.sizes, initial=0))  # len == n_shards + 1
        self.n_tokens = self.offsets[-1]

    def tokens(self, start: int, count: int) -> Tensor:
        assert 0 <= start and start + count <= self.n_tokens
        raw = bytearray()
        i = bisect_right(self.offsets, start) - 1
        while len(raw) < count * 2:
            local = start - self.offsets[i]
            take = min(count - len(raw) // 2, self.sizes[i] - local)
            raw += self.mms[i][local * 2 : (local + take) * 2]
            start += take
            i += 1
        # frombuffer wants a writable buffer, which the bytearray already is
        return torch.frombuffer(raw, dtype=torch.uint16).to(torch.int64)

    def close(self) -> None:
        for m, f in zip(self.mms, self.files):
            m.close()
            f.close()


def get_batch(
    ds: BinDataset,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Random windows into the split. y is x shifted one token left: predicting
    token t+1 from tokens <= t, for every t at once."""
    high = ds.n_tokens - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)
    chunks = [ds.tokens(i, block_size + 1) for i in ix.tolist()]
    x = torch.stack([c[:-1] for c in chunks])
    y = torch.stack([c[1:] for c in chunks])
    return x, y  # [B, T], [B, T]


if __name__ == "__main__":
    import tempfile

    import numpy as np

    V = meta["vocab_size"]
    val = BinDataset("val")

    # 1. token count matches what the prepare step recorded
    print(f"val: {val.n_tokens:,} tokens   meta says {meta['val']['n_tokens']:,}")
    assert val.n_tokens == meta["val"]["n_tokens"]

    # 2. our stdlib mmap reads the same bytes numpy does
    ref = np.memmap(val.paths[0], dtype=np.uint16, mode="r")
    for start in (0, 1, 12345, val.sizes[0] - 10):
        assert val.tokens(start, 9).tolist() == ref[start : start + 9].tolist(), start
    print("matches np.memmap at every probe")

    # 3. batch shape, dtype, range
    g = torch.Generator().manual_seed(1337)
    x, y = get_batch(val, batch_size=4, block_size=16, generator=g)
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == torch.int64  # Embedding needs long indices
    assert x.min() >= 0 and x.max() < V

    # 4. y IS x shifted by one -- the whole supervision signal
    assert torch.equal(x[:, 1:], y[:, :-1])
    print("x[0][:8] =", x[0, :8].tolist())
    print("y[0][:8] =", y[0, :8].tolist())

    # 5. seeded generator is reproducible; a different seed is not
    a = get_batch(val, 4, 16, torch.Generator().manual_seed(0))[0]
    b = get_batch(val, 4, 16, torch.Generator().manual_seed(0))[0]
    assert torch.equal(a, b)
    assert not torch.equal(
        a, get_batch(val, 4, 16, torch.Generator().manual_seed(1))[0]
    )

    # 6. never reads past the end
    for _ in range(200):
        get_batch(val, 8, 512)

    # 7. many shards read as one stream, including windows that straddle a boundary
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        whole = np.arange(3000, dtype=np.uint16)
        for i in range(3):
            whole[i * 1000 : (i + 1) * 1000].tofile(tmp / f"train_{i:05d}.bin")
        ds = BinDataset("train", tmp)
        assert len(ds.paths) == 3 and ds.n_tokens == 3000
        assert ds.tokens(0, 3000).tolist() == whole.tolist()
        for start, count in [(995, 10), (0, 1001), (1999, 2), (990, 1020), (2990, 10)]:
            got = ds.tokens(start, count).tolist()
            assert got == whole[start : start + count].tolist(), (start, count)
        ds.close()
    print("3 shards read as one contiguous stream")

    # 8. the big split opens without loading its bytes
    train = BinDataset("train")
    mb = sum(p.stat().st_size for p in train.paths) / 1e6
    print(f"train: {train.n_tokens:,} tokens, {len(train.paths)} shard(s), {mb:.0f} MB")
    assert train.n_tokens == meta["train"]["n_tokens"]
    train.close()
    val.close()

    print("ok")
