import json
import mmap

import torch
from paths import DATA_DIR
from torch import Tensor

meta = json.loads((DATA_DIR / "meta.json").read_text())


class BinDataset:
    """A .bin of uint16 token ids, memory-mapped: the OS pages it in on demand,
    so a 945 MB train split costs no RAM until it is actually read."""

    def __init__(self, split: str):
        self.path = DATA_DIR / f"{split}.bin"
        self.file = self.path.open("rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.n_tokens = len(self.mm) // 2  # uint16 -> 2 bytes per token

    def tokens(self, start: int, count: int) -> Tensor:
        raw = self.mm[start * 2 : (start + count) * 2]
        # frombuffer wants a writable buffer; bytearray copies these few bytes
        return torch.frombuffer(bytearray(raw), dtype=torch.uint16).to(torch.int64)

    def close(self) -> None:
        self.mm.close()
        self.file.close()


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
    import numpy as np

    V = meta["vocab_size"]
    val = BinDataset("val")

    # 1. token count matches what prepare_data recorded
    print(f"val: {val.n_tokens:,} tokens   meta says {meta['val']['n_tokens']:,}")
    assert val.n_tokens == meta["val"]["n_tokens"]

    # 2. our stdlib mmap reads the same bytes numpy does
    ref = np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r")
    assert len(ref) == val.n_tokens
    for start in (0, 1, 12345, val.n_tokens - 10):
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

    # 7. the big split opens without loading 945 MB
    train = BinDataset("train")
    print(
        f"train: {train.n_tokens:,} tokens, file {train.path.stat().st_size / 1e6:.0f} MB"
    )
    assert train.n_tokens == meta["train"]["n_tokens"]
    train.close()
    val.close()

    print("ok")
