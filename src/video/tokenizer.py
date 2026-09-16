"""Byte-level BPE: regex pre-split, count adjacent pairs, merge the most frequent, repeat.

Two backends, same ids. `OurBPETokenizer` is the pure-Python one; `FastBPETokenizer`
wraps tiktoken and is what tokenizes billions of tokens. They agree because both are
built from the same merge table -- ours can adopt GPT-2's (`from_mergeable_ranks`) and
hand its own back (`mergeable_ranks`).
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import regex

from backend import USE_TORCH
from paths import DATASET, TOKENIZER_DIR

# GPT-2's pre-tokenizer. Merges never cross a chunk boundary, which is what stops
# "dog." and " the" from becoming single tokens.
GPT2_PAT = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
# GPT-4's: case-insensitive contractions, digits capped at 3, punctuation keeps its
# trailing newlines.
GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

ENDOFTEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """[1,2,1,2,3] -> {(1,2):2, (2,1):1, (2,3):1}."""
    counts: dict[tuple[int, int], int] = {}
    for a, b in pairwise(ids):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of `pair` with `idx`.

    The `i += 2` is what makes [1,1,1] merge to [99,1] and not [99,99]: a token
    consumed by one merge cannot also feed the next.
    """
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class OurBPETokenizer:
    def __init__(self, pat: str = GPT2_PAT):
        self.pat = pat
        self._re = regex.compile(pat)
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> new id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.byte_ids: list[int] = list(range(256))  # byte value -> id
        self.specials: dict[str, int] = {}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return max([*self.vocab, *self.specials.values()], default=-1) + 1

    def chunks(self, text: str) -> list[str]:
        return self._re.findall(text)

    # --- training -------------------------------------------------------------

    def train(self, text: str, vocab_size: int) -> None:
        """Learn vocab_size - 256 merges, most frequent pair first.

        Counting unique chunks once and carrying their multiplicity is the only
        reason this finishes: a corpus of N chunks collapses to its distinct ones.
        """
        assert vocab_size >= 256
        self.merges.clear()
        self._cache.clear()
        self.byte_ids = list(range(256))
        corpus = [(list(c.encode()), n) for c, n in Counter(self.chunks(text)).items()]
        for i in range(vocab_size - 256):
            stats: dict[tuple[int, int], int] = {}
            for ids, n in corpus:
                for a, b in pairwise(ids):
                    stats[(a, b)] = stats.get((a, b), 0) + n
            if not stats:
                break
            pair = max(stats, key=lambda p: stats[p])
            idx = 256 + i
            corpus = [(merge(ids, pair, idx), n) for ids, n in corpus]
            self.merges[pair] = idx
        self._rebuild_vocab()

    def _rebuild_vocab(self) -> None:
        # merges are in learned order, so both halves of an id always exist first
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    # --- adopting an external table -------------------------------------------

    @classmethod
    def from_mergeable_ranks(
        cls,
        ranks: dict[bytes, int],
        pat: str = GPT2_PAT,
        specials: dict[str, int] | None = None,
    ) -> OurBPETokenizer:
        """Recover the merge list from a rank table (tiktoken's format).

        A rank table says what each id expands to but not which pair built it.
        BPE's own invariant recovers that: encoding a token's bytes with every
        lower rank has to leave exactly two pieces, and those are the pair.
        """
        tok = cls(pat)
        tok.vocab = {r: b for b, r in ranks.items()}
        assert len(tok.vocab) == len(ranks), "ranks are not unique"
        tok.byte_ids = [ranks[bytes([i])] for i in range(256)]
        tok.merges = {}
        for b, r in sorted(ranks.items(), key=lambda kv: kv[1]):
            if len(b) == 1:
                continue
            parts = tok._encode_bytes(b)
            assert len(parts) == 2, f"rank {r} ({b!r}) split into {len(parts)}"
            tok.merges[(parts[0], parts[1])] = r
        tok.specials = dict(specials or {})
        return tok

    def mergeable_ranks(self) -> dict[bytes, int]:
        """The inverse, so tiktoken can be handed a vocab we trained."""
        return {b: i for i, b in self.vocab.items()}

    def add_special_tokens(self, tokens: list[str]) -> None:
        nxt = self.vocab_size
        for i, t in enumerate(tokens):
            self.specials.setdefault(t, nxt + i)

    # --- encode / decode ------------------------------------------------------

    def _encode_bytes(self, raw: bytes) -> list[int]:
        ids = [self.byte_ids[b] for b in raw]
        while len(ids) >= 2:
            stats = get_stats(ids)
            # lowest merge index = learned earliest; inf keeps unlearned pairs last
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """No special tokens: "<|endoftext|>" comes back as its literal characters."""
        out: list[int] = []
        for chunk in self.chunks(text):
            hit = self._cache.get(chunk)
            if hit is None:
                hit = self._cache[chunk] = self._encode_bytes(chunk.encode())
            out.extend(hit)
        return out

    def encode(self, text: str) -> list[int]:
        if not self.specials:
            return self.encode_ordinary(text)
        parts = regex.split(
            "(" + "|".join(regex.escape(s) for s in self.specials) + ")", text
        )
        out: list[int] = []
        for part in parts:
            if part in self.specials:
                out.append(self.specials[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def encode_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        return [self.encode_ordinary(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        rev = {i: s for s, i in self.specials.items()}
        pieces: list[str] = []
        run: list[bytes] = []
        for i in ids:
            if i in rev:
                pieces.append(self._flush(run))
                pieces.append(rev[i])
            else:
                run.append(self.vocab[i])
        pieces.append(self._flush(run))
        return "".join(pieces)

    @staticmethod
    def _flush(run: list[bytes]) -> str:
        # errors="replace": a slice of ids can cut a multi-byte character in half
        s = b"".join(run).decode("utf-8", errors="replace")
        run.clear()
        return s

    # --- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        # merges carry explicit ids because an adopted table does not number them 256, 257, ...
        Path(path).write_text(
            json.dumps(
                {
                    "pat": self.pat,
                    "byte_ids": self.byte_ids,
                    "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
                    "specials": self.specials,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> OurBPETokenizer:
        obj = json.loads(Path(path).read_text())
        if (
            "byte_ids" not in obj
        ):  # src/tokenizer.py's format: ids numbered 256, 257, ...
            obj = {
                "pat": GPT4_PAT,
                "byte_ids": list(range(256)),
                "merges": [[a, b, 256 + i] for i, (a, b) in enumerate(obj["merges"])],
                "specials": {},
            }
        tok = cls(obj["pat"])
        tok.byte_ids = obj["byte_ids"]
        tok.merges = {(a, b): idx for a, b, idx in obj["merges"]}
        tok.specials = obj["specials"]
        tok.vocab = {i: bytes([b]) for b, i in enumerate(tok.byte_ids)}
        for (a, b), idx in tok.merges.items():
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        return tok


class FastBPETokenizer:
    """tiktoken over the same merge table. Rust, and it releases the GIL, so
    encode_batch threads."""

    def __init__(
        self,
        ranks: dict[bytes, int],
        pat: str = GPT2_PAT,
        specials: dict[str, int] | None = None,
    ):
        import tiktoken

        self.enc = tiktoken.Encoding(
            name="video",
            pat_str=pat,
            mergeable_ranks=ranks,
            special_tokens=specials or {},
        )

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    @property
    def specials(self) -> dict[str, int]:
        return self.enc._special_tokens

    def encode_ordinary(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special="all")

    def encode_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        return self.enc.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def gpt2_ranks() -> dict[bytes, int]:
    import tiktoken

    return tiktoken.get_encoding("gpt2")._mergeable_ranks


# GPT-2's 50256 merges, then the tokens SFT needs. 50259 still fits uint16.
SPECIALS = {ENDOFTEXT: 50256, IM_START: 50257, IM_END: 50258}
VOCAB_SIZE = 50259

Tokenizer = FastBPETokenizer if USE_TORCH else OurBPETokenizer


def load_tokenizer():
    """What the data pipeline uses. VIDEO_BACKEND picks the implementation, not the ids."""
    ranks = gpt2_ranks()
    if USE_TORCH:
        return FastBPETokenizer(ranks, GPT2_PAT, SPECIALS)
    return OurBPETokenizer.from_mergeable_ranks(ranks, GPT2_PAT, SPECIALS)


def tokenizer_for(dataset: str = DATASET):
    """The tokenizer a prepared dataset was written with -- ids are meaningless otherwise."""
    if dataset.startswith("tinystories"):
        return OurBPETokenizer.load(TOKENIZER_DIR / "bpe_ts_4096.json")
    return load_tokenizer()


if __name__ == "__main__":
    import tempfile
    import time

    # 1. merge() does not reuse a token it just consumed
    assert merge([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3]
    assert merge([1, 1, 1], (1, 1), 99) == [99, 1]
    assert get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}

    text = (Path(__file__).parent / "gpt.py").read_text() * 4

    # 2. training adds one id per merge, and every id decodes back to bytes
    ours = OurBPETokenizer()
    ours.train(text, 512)
    assert len(ours.merges) == 256 and ours.vocab_size == 512
    assert all(
        ours.vocab[a] + ours.vocab[b] == ours.vocab[i]
        for (a, b), i in ours.merges.items()
    )

    # 3. roundtrip is exact, and a bigger vocab is strictly fewer tokens
    assert ours.decode(ours.encode(text)) == text
    small = OurBPETokenizer()
    small.train(text, 320)
    assert len(small.encode(text)) > len(ours.encode(text)) > len(text.encode()) // 8
    print(f"vocab 512: {len(text) / len(ours.encode(text)):.2f} chars/token")

    # 4. the oracle: tiktoken given OUR table encodes identically
    oracle = FastBPETokenizer(ours.mergeable_ranks(), GPT2_PAT)
    probe = "def forward(self, x: Tensor) -> Tensor:\n    return self.ln(x) # fin\n"
    assert ours.encode(probe) == oracle.encode(probe)
    assert ours.encode(text) == oracle.encode(text)

    # 5. GPT-2's table round-trips through the merge list -- 50000 merges recovered,
    #    and its byte ids are a permutation of 0..255, not the identity
    t0 = time.time()
    g2 = OurBPETokenizer.from_mergeable_ranks(gpt2_ranks(), GPT2_PAT, SPECIALS)
    assert len(g2.merges) == 50000 and g2.vocab_size == VOCAB_SIZE
    assert sorted(g2.byte_ids) == list(range(256)) and g2.byte_ids != list(range(256))
    print(f"recovered GPT-2's merges in {time.time() - t0:.2f}s")

    # 6. ours == tiktoken's own gpt2, token for token, on text neither was tuned on
    import tiktoken

    ref = tiktoken.get_encoding("gpt2")
    for s in [
        text[:20000],
        "café naïve — dash, emoji 🚀 and 漢字テスト",
        "   leading\n\n\ntrailing   \tit's DON'T 12345678",
        "",
    ]:
        assert g2.encode_ordinary(s) == ref.encode_ordinary(s), repr(s[:40])
    print("ours matches tiktoken's gpt2 on every probe")

    # 7. specials are atomic in encode() and literal text in encode_ordinary()
    chat = f"{IM_START}user\nhi{IM_END}\n{ENDOFTEXT}"
    ids = g2.encode(chat)
    assert ids[0] == 50257 and 50258 in ids and ids[-1] == 50256
    assert all(i < 50256 for i in g2.encode_ordinary(chat))
    assert g2.decode(ids) == chat
    fast = (
        load_tokenizer()
        if USE_TORCH
        else FastBPETokenizer(gpt2_ranks(), GPT2_PAT, SPECIALS)
    )
    assert fast.encode(chat) == ids and fast.decode(ids) == chat

    # 8. decode never raises on a cut multi-byte character
    assert g2.decode(g2.encode("🚀")[:1]).endswith("�")

    # 9. save/load preserves ids exactly, for both the trained and adopted tables
    with tempfile.TemporaryDirectory() as d:
        for tok in (ours, g2):
            p = Path(d) / "t.json"
            tok.save(p)
            back = OurBPETokenizer.load(p)
            assert back.vocab == tok.vocab and back.specials == tok.specials
            assert back.encode(probe) == tok.encode(probe)

    # 10. src/tokenizer.py's older save format still loads, GPT-4 pattern and all,
    #     so the tinystories artifact and every checkpoint trained on it keep working
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "legacy.json"
        p.write_text(
            json.dumps({"vocab_size": 512, "merges": [list(k) for k in ours.merges]})
        )
        legacy = OurBPETokenizer.load(p)
        assert legacy.pat == GPT4_PAT and legacy.byte_ids == list(range(256))
        assert legacy.vocab == ours.vocab
    old_bpe = TOKENIZER_DIR / "bpe_ts_4096.json"
    if old_bpe.exists():
        ts = OurBPETokenizer.load(old_bpe)
        assert ts.vocab_size == 4096
        assert ts.decode(ts.encode("Once upon a time, Lily saw a big red ball.")) == (
            "Once upon a time, Lily saw a big red ball."
        )

    # 11. load_tokenizer() gives the same ids whichever backend is selected
    assert load_tokenizer().encode(chat) == ids
    print(f"backend={'torch' if USE_TORCH else 'ours'}  vocab_size={VOCAB_SIZE}")
    print("ok")
