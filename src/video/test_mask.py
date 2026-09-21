import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from conftest import requires_cuda
from mask import doc_block_mask, doc_causal_mask, doc_ids
from task import windows


@pytest.fixture
def straddling(build, tok):
    """One 8-wide window over ["ab", "cd"] -- "ab>ba<eot>" is 6 tokens, so the
    window's tail already holds the next example."""
    x, _, _ = windows(build(["ab", "cd"]), 1, 8, np.random.default_rng(0))
    return x, tok.specials["<|endoftext|>"]


def test_window_straddles_two_examples(straddling, chars):
    x, _ = straddling
    assert chars(x[0]) == [*"ab>ba", "<eot>", "c", "d"]


def test_doc_ids_split_after_a_trailing_eot(straddling):
    x, eot = straddling
    # Packed ends each example with eot, so the eot closes document 0
    assert doc_ids(x, eot)[0].tolist() == [0] * 6 + [1, 1]


def test_doc_ids_split_before_a_leading_eot(tok):
    # fineweb.py writes [eot] + ids, so the eot opens its document instead
    eot = tok.specials["<|endoftext|>"]
    x = torch.tensor([[eot, 1, 2, eot, 3, 4]])
    assert doc_ids(x, eot, eot_leads=True)[0].tolist() == [1, 1, 1, 2, 2, 2]
    assert doc_ids(x, eot)[0].tolist() == [0, 1, 1, 1, 2, 2]


def test_mask_is_causal_inside_one_document(straddling):
    x, eot = straddling
    m = doc_causal_mask(doc_ids(x, eot))[0, 0]
    assert m[:6, :6].tolist() == torch.ones(6, 6).tril().bool().tolist()


def test_mask_blocks_the_overhang(straddling):
    x, eot = straddling
    m = doc_causal_mask(doc_ids(x, eot))[0, 0]
    assert not m[6:, :6].any()  # "c"/"d" cannot see "ab>ba<eot>"
    assert m[6:, 6:].tolist() == [[True, False], [True, True]]  # still causal


def test_one_document_leaves_the_causal_mask_alone(build, tok):
    x, _, _ = windows(build(["cat", "dog"]), 1, 8, np.random.default_rng(0))
    m = doc_causal_mask(doc_ids(x, tok.specials["<|endoftext|>"]))[0, 0]
    assert m.tolist() == torch.ones(8, 8).tril().bool().tolist()


@pytest.mark.slow
@requires_cuda
def test_block_mask_matches_the_materialized_mask():
    # compiled: uncompiled flex_attention materializes the scores matrix, which
    # is the thing a BlockMask exists to avoid
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 256, 32
    eot = 0
    x = torch.randint(1, 100, (B, T), device="cuda")
    x[:, ::64] = eot  # a document boundary every 64 positions
    docs = doc_ids(x, eot)
    q, k, v = (torch.randn(B, H, T, D, device="cuda") for _ in range(3))

    fused = torch.compile(flex_attention)
    flex = fused(q, k, v, block_mask=doc_block_mask(docs))
    sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=doc_causal_mask(docs))
    assert torch.allclose(flex, sdpa, atol=1e-5)
