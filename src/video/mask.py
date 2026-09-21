"""Document masking: stop attention crossing an example boundary inside a window.

windows() packs examples back to back, so a window's tail can already hold the
next example. The loss excludes those positions; attention does not.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def doc_ids(x: Tensor, eot_id: int, eot_leads: bool = False) -> Tensor:
    """[B, T] int -- which document each position belongs to, numbered from 0.

    eot_leads picks the convention: fineweb.py writes `[eot] + ids` so the eot
    opens its document, while Packed ends each example with one.
    """
    is_eot = x == eot_id
    n = is_eot.cumsum(-1)
    return n if eot_leads else n - is_eot.long()


def doc_causal_mask(docs: Tensor) -> Tensor:
    """[B, 1, T, T] bool -- causal, and only within one document."""
    T = docs.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=docs.device).tril()
    same = docs[:, :, None] == docs[:, None, :]
    return (causal & same)[:, None]


def doc_block_mask(docs: Tensor) -> BlockMask:
    """The same mask as a BlockMask: flex_attention skips whole blocks instead
    of materialising T x T."""

    def keep(b, h, q, k):
        return (q >= k) & (docs[b, q] == docs[b, k])

    B, T = docs.shape
    return create_block_mask(keep, B, None, T, T, device=str(docs.device))


if __name__ == "__main__":
    import numpy as np

    from task import Packed, windows
    from tokenizer import load_tokenizer

    tok = load_tokenizer()
    eot = tok.specials["<|endoftext|>"]

    # "ab" packs to "ab>" + "ba<eot>", so an 8-wide window over ["ab", "cd"]
    # straddles the boundary
    ids, mask, starts = [], [], []
    for w in ["ab", "cd"]:
        prompt = [tok.encode(ch)[0] for ch in w] + [tok.encode(">")[0]]
        comp = [tok.encode(ch)[0] for ch in reversed(w)] + [eot]
        starts.append(len(ids))
        ids += prompt + comp
        mask += [False] * len(prompt) + [True] * len(comp)

    packed = Packed(
        ids=np.array(ids, np.uint16),
        mask=np.array(mask, bool),
        starts=np.array(starts, np.int64),
        vocab_size=tok.vocab_size,
        eot_id=eot,
    )
    x, _, _ = windows(packed, 1, 8, np.random.default_rng(0))
    docs = doc_ids(x, eot)
    chars = [tok.decode([int(t)]).replace("<|endoftext|>", "@") for t in x[0]]

    print("window:", " ".join(f"{c:>3}" for c in chars))
    print("doc:   ", " ".join(f"{int(d):>3}" for d in docs[0]))

    m = doc_causal_mask(docs)[0, 0]
    causal = torch.ones(8, 8, dtype=torch.bool).tril()
    print("\nX attended   . causal but another document   - future\n")
    print("      " + " ".join(f"{c:>3}" for c in chars))
    for i in range(8):
        row = ["X" if m[i, j] else ("." if causal[i, j] else "-") for j in range(8)]
        print(f"{chars[i]:>4}  " + " ".join(f"{c:>3}" for c in row))
