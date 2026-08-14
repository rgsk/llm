from __future__ import annotations

import json
from collections import Counter

import regex


class CharTokenizer:
    def __init__(self, text: str):
        self.itoc = sorted(set(text))
        self.ctoi = {c: i for i, c in enumerate(self.itoc)}
        self.vocab_size = len(self.itoc)

    def encode(self, s: str) -> list[int]:
        return [self.ctoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itoc[i] for i in ids)


PAT = regex.compile(
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)


def split_chunks(text: str) -> list[str]:
    return PAT.findall(text)


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair appears: [1,2,1,2,3] -> {(1,2):2,(2,1):1,(2,3):1}."""
    counts = {}
    for i in range(len(ids) - 1):
        counts[(ids[i], ids[i + 1])] = counts.get((ids[i], ids[i + 1]), 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of `pair` with `idx`: merge([1,2,1,2,3],(1,2),99) -> [99,99,3].

    The i += 2 on a hit is what handles overlaps correctly: [1,1,1] merging
    (1,1) gives [99,1], not [99,99] — the middle element isn't reused.
    """
    new_ids: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def get_stats_weighted(corpus) -> dict[tuple[int, int], int]:
    counts = {}
    for ids, n in corpus:
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + n
    return counts


class BPETokenizer:
    def __init__(self):
        # learned during train(); merges records the ORDER pairs were merged in.
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> new_id
        self.vocab: dict[int, bytes] = {}  # id -> the bytes it expands to
        self.vocab_size = 0
        self._cache: dict[str, list[int]] = {}

    def train(self, text: str, vocab_size: int) -> None:
        """Learn (vocab_size - 256) merges: greedily merge the most frequent pair, repeat.

        The new ids count up 256, 257, ... and self.merges values ARE that order,
        which encode() must replay (earliest first) so a later merge only fires
        after the earlier one it builds on (e.g. t+h->th before th+e->the).
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (the byte base)"
        self.merges.clear()
        self._cache.clear()

        num_merges = vocab_size - 256
        counts = Counter(split_chunks(text))
        corpus = [(list(ch.encode("utf-8")), n) for ch, n in counts.items()]

        for i in range(num_merges):
            stats = get_stats_weighted(corpus)
            if not stats:
                break  # nothing left to merge
            pair = max(stats, key=stats.get)  # type: ignore
            idx = 256 + i
            corpus = [(merge(c, pair, idx), n) for c, n in corpus]
            self.merges[pair] = idx

        # id -> bytes, so decode can expand. Built in learned order so each
        # merged id's two halves already exist when we concatenate them.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.vocab_size = len(self.vocab)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no remaining pair is mergeable
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, s: str) -> list[int]:
        """str -> list[int], replaying merges in LEARNED order (not by frequency).

        Each pass merges the pair with the lowest merge index = learned earliest;
        the inf default makes never-learned pairs sort last so they're never picked.
        (len >= 2 guarantees stats is non-empty, so min() is safe.)
        """
        out = []
        for chunk in split_chunks(s):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            out.extend(self._cache[chunk])
        return out

    def decode(self, ids: list[int]) -> str:
        # errors="replace": ids can form invalid UTF-8 mid-character
        # (e.g. a model mid-generation); replace emits U+FFFD (�) instead of crashing.
        # eg. decode([195]) will cause error without errors="replace"
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # --- persistence: training is slow (~minutes on 1MB), so cache the result ---
    def save(self, path: str) -> None:
        # only the merges (in learned order) + vocab_size need saving; vocab is
        # derived from them. JSON-friendly: tuple keys -> [a, b] pairs.

        with open(path, "w") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "merges": [[a, b] for (a, b) in self.merges],
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> BPETokenizer:

        with open(path) as f:
            obj = json.load(f)
        tok = cls()
        tok.vocab_size = obj["vocab_size"]
        # file order IS learned order, so ids count up 256, 257, ... again
        tok.merges = {(a, b): 256 + i for i, (a, b) in enumerate(obj["merges"])}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in tok.merges.items():
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        return tok
