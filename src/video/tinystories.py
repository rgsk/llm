"""TinyStories with a real end-of-text token.

The prepared corpus was written by a tokenizer with no specials, so the story
boundary sits in the stream as the three tokens that spell "<|endoftext|>\\n".
Packed, doc_ids and every Task want one id. This rewrites the boundary as a
single new token at the end of the vocab -- no re-download and no
re-tokenization, since ids 0..4095 keep their meaning either way.

Why bother: the base then learns an actual stop token during pretraining, which
is the thing SFT's stop_rate measures and generation needs to terminate.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from paths import DATA_ROOT, TOKENIZER_DIR
from tokenizer import ENDOFTEXT, OurBPETokenizer

SRC = DATA_ROOT / "tinystories"
OUT = DATA_ROOT / "tinystories_eot"
OLD_TOK = TOKENIZER_DIR / "bpe_ts_4096.json"
NEW_TOK = TOKENIZER_DIR / "bpe_ts_4097.json"
# "<|", "endoftext", then either "|>\n" or "|>" -- 2.1M of the first and 230 of
# the second, and the trailing newline rides along inside "|>\n"
OPEN, MID, CLOSE = 379, 381, {383: len(ENDOFTEXT + "\n"), 382: len(ENDOFTEXT)}


def build_tokenizer() -> OurBPETokenizer:
    """The same 4096 merges, plus <|endoftext|> as id 4096."""
    tok = OurBPETokenizer.load(OLD_TOK)
    assert tok.vocab_size == 4096 and not tok.specials
    tok.specials = {ENDOFTEXT: 4096}
    tok.save(NEW_TOK)
    return tok


def collapse(ids: np.ndarray, eot_id: int) -> tuple[np.ndarray, int, int]:
    """Replace each boundary triple with one eot id -> (ids, n_docs, chars_dropped)."""
    third = np.isin(ids[2:], list(CLOSE))
    hit = np.flatnonzero((ids[:-2] == OPEN) & (ids[1:-1] == MID) & third)
    # every "<|" must be a boundary, or documents would split where the corpus
    # did not -- this is the assumption the whole rewrite rests on
    assert (ids == OPEN).sum() == len(hit), "a '<|' outside a boundary"

    out = ids.copy()
    out[hit] = eot_id
    drop = np.zeros(len(ids), bool)
    drop[hit + 1] = True
    drop[hit + 2] = True
    chars = sum(CLOSE[int(ids[i + 2])] for i in hit)
    return out[~drop], len(hit), chars


if __name__ == "__main__":
    # --build to rewrite, like fineweb.py: test_legacy_mains sweeps this file,
    # and the rewrite reads 945 MB
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="rewrite the boundary token")
    args = ap.parse_args()

    if not args.build:
        if not (OUT / "meta.json").exists():
            print(f"{OUT} not built -- run with --build")
            raise SystemExit
        meta = json.loads((OUT / "meta.json").read_text())
        tok = OurBPETokenizer.load(NEW_TOK)
        eot = tok.specials[ENDOFTEXT]
        val = np.fromfile(OUT / "val.bin", dtype=np.uint16)
        i = int(np.flatnonzero(val == eot)[0])
        print(f"vocab {meta['vocab_size']}   eot id {eot}")
        for split in ("train", "val"):
            m = meta[split]
            print(f"  {split}: {m['n_tokens']:,} tokens, {m['n_docs']:,} documents")
        print("around the first boundary:")
        print([tok.decode([int(t)]) for t in val[i - 2 : i + 3]])
        raise SystemExit

    tok = build_tokenizer()
    eot = tok.specials[ENDOFTEXT]
    OUT.mkdir(parents=True, exist_ok=True)
    old_meta = json.loads((SRC / "meta.json").read_text())
    meta = {"vocab_size": tok.vocab_size}

    for split in ("train", "val"):
        ids = np.fromfile(SRC / f"{split}.bin", dtype=np.uint16)
        out, n_docs, chars = collapse(ids, eot)
        out.tofile(OUT / f"{split}.bin")
        # the separator's characters leave the text with it
        n_chars = old_meta[split]["n_chars"] - chars
        meta[split] = {
            "n_chars": n_chars,
            "n_tokens": len(out),
            "chars_per_token": n_chars / len(out),
            "n_docs": n_docs,
        }
        print(
            f"{split}: {len(ids):,} -> {len(out):,} tokens, {n_docs:,} documents, "
            f"{n_chars / len(out):.3f} chars/token"
        )

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT}")
