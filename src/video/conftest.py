"""Shared fixtures for src/video tests.

conftest, not a helpers module: this folder is flat with no __init__.py, so
test files cannot import each other and two test_x.py anywhere in the repo
collide.
"""

import numpy as np
import pytest
import torch

from task import Packed
from tokenizer import load_tokenizer

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session")
def tok():
    return load_tokenizer()


@pytest.fixture
def build(tok):
    """words -> Packed of reverse-string examples: "cat" becomes the prompt
    "cat>" and the completion "tac<eot>", one char per token."""
    eot = tok.specials["<|endoftext|>"]

    def make(words: list[str]) -> Packed:
        ids: list[int] = []
        mask: list[bool] = []
        starts: list[int] = []
        for w in words:
            prompt = [tok.encode(c)[0] for c in w] + [tok.encode(">")[0]]
            comp = [tok.encode(c)[0] for c in reversed(w)] + [eot]
            starts.append(len(ids))
            ids += prompt + comp
            mask += [False] * len(prompt) + [True] * len(comp)
        return Packed(
            ids=np.array(ids, np.uint16),
            mask=np.array(mask, bool),
            starts=np.array(starts, np.int64),
            vocab_size=tok.vocab_size,
            eot_id=eot,
        )

    return make


@pytest.fixture
def chars(tok):
    """ids -> one string per token, so an assert shows the data not the numbers."""

    def d(ids) -> list[str]:
        return [tok.decode([int(t)]).replace("<|endoftext|>", "<eot>") for t in ids]

    return d
