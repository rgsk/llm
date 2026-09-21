import pytest
import torch

from conftest import requires_cuda
from flex_attention import FlexAttention
from mask import doc_block_mask, doc_ids
from rope import rope_angles

B, T, E, NH = 2, 128, 64, 4
HS = E // NH


def rope(n: int, offset: int = 0):
    """cos/sin for n positions -- GPT's job in real use, spelled out here."""
    return rope_angles(offset, n, HS)


@pytest.fixture
def layer():
    torch.manual_seed(0)
    return FlexAttention(E, NH, block_size=256, n_kv_head=2)


def test_shape_is_preserved(layer):
    assert layer(torch.randn(B, T, E), *rope(T)).shape == (B, T, E)


def test_gqa_shrinks_only_the_kv_projection(layer):
    # q keeps E columns, k and v get n_kv_head * head_size each
    assert layer.qkv.weight.shape == (E + 2 * 2 * (E // NH), E)
    assert FlexAttention(E, NH, 256).qkv.weight.shape == (3 * E, E)  # MHA


def test_default_path_is_causal(layer):
    x = torch.randn(B, T, E)
    out = layer(x, *rope(T))
    x[:, T // 2 :] = torch.randn(B, T - T // 2, E)  # change only the future
    assert torch.allclose(out[:, : T // 2], layer(x, *rope(T))[:, : T // 2], atol=1e-6)


def test_block_mask_changes_the_output(layer):
    x = torch.randn(B, T, E)
    ids = torch.ones(B, T, dtype=torch.long)
    ids[:, ::32] = 0
    bm = doc_block_mask(doc_ids(ids, 0))
    with torch.no_grad():  # flex has no CPU backward, so no graph on cpu
        assert not torch.allclose(layer(x, *rope(T)), layer(x, *rope(T), block_mask=bm))


def test_flex_on_cpu_with_grad_is_refused(layer):
    ids = torch.ones(B, T, dtype=torch.long)
    bm = doc_block_mask(doc_ids(ids, 0))
    with pytest.raises(AssertionError, match="no CPU backward"):
        layer(torch.randn(B, T, E), *rope(T), block_mask=bm)


def test_cache_returns_the_same_logits_as_a_full_pass(layer):
    torch.manual_seed(0)
    x = torch.randn(B, T, E)
    full = layer(x, *rope(T))
    with torch.no_grad():  # a preallocated cache writes in place
        prefill, cache = layer(x[:, :-1], *rope(T - 1), use_cache=True)
        step, _ = layer(x[:, -1:], *rope(1, offset=T - 1), cache, use_cache=True)
    assert torch.allclose(full[:, :-1], prefill, atol=1e-5)
    assert torch.allclose(full[:, -1:], step, atol=1e-5)


def test_rope_is_computed_on_demand_not_tabled(layer):
    # no buffers to run out of, so training length stops being a ceiling --
    # this is what PI/NTK/YaRN need
    assert not [n for n, _ in layer.named_buffers()]
    assert layer(torch.randn(B, 512, E), *rope(512)).shape == (
        B,
        512,
        E,
    )  # block_size is 256


def test_rope_must_be_threaded_not_defaulted():
    """A layer computing its own angles would use base/scale defaults, so a
    PI/NTK/YaRN run would come out silently un-extended."""
    from block import Block

    blk = Block(E, NH, block_size=128, attention="flex", n_kv_head=2)
    with pytest.raises(AssertionError, match="threaded from GPT"):
        blk(torch.randn(B, 32, E))


def test_gpt_threads_one_set_of_angles_to_every_layer():
    from dataclasses import asdict

    from gpt import GPT
    from gpt_config import GPTConfig

    cfg = GPTConfig(
        vocab_size=64,
        block_size=128,
        n_embed=E,
        n_head=NH,
        n_layer=2,
        attention="flex",
        n_kv_head=2,
        position="rope",
    )
    torch.manual_seed(0)
    m = GPT(**asdict(cfg))
    assert m(torch.randint(0, 64, (B, 32))).shape == (B, 32, 64)
    assert not [n for n, _ in m.named_buffers() if "rope" in n]


def test_cache_grows_past_block_size(layer):
    """block_size sizes the first buffer, it does not cap the run -- rope has no
    ceiling now, so the cache must not reintroduce one."""
    with torch.no_grad():
        _, cache = layer(torch.randn(B, 256, E), *rope(256), use_cache=True)
        assert cache.k.size(2) == 256
        out, cache = layer(
            torch.randn(B, 1, E), *rope(1, offset=256), cache, use_cache=True
        )
    assert out.shape == (B, 1, E)
    assert cache.pos == 257 and cache.k.size(2) == 512  # doubled, not refused


def test_cache_without_grow_still_refuses():
    from kv_cache import KVCache

    c = KVCache((1, 1, 4, 8))  # grow defaults to False, as the older rungs expect
    with torch.no_grad():
        c.append(torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8))
        with pytest.raises(AssertionError, match="outgrew the cache"):
            c.append(torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8))


@pytest.mark.slow
@requires_cuda
def test_flex_matches_sdpa_on_the_same_mask(layer):
    """The BlockMask path and a plain causal pass agree when every position is
    one document -- so the speedup is not coming from dropping real work."""
    layer = layer.cuda()
    x = torch.randn(B, T, E, device="cuda")
    ids = torch.ones(B, T, dtype=torch.long, device="cuda")  # no eot: one document
    bm = doc_block_mask(doc_ids(ids, 0))
    cos, sin = (t.cuda() for t in rope(T))
    assert torch.allclose(
        layer(x, cos, sin), layer(x, cos, sin, block_mask=bm), atol=1e-4
    )


def test_gpt_threads_angles_through_the_cached_path_too():
    # the decode branch had no cos/sin, so flex + use_cache raised on the first
    # step -- exercised only once a task generated with a cache
    from dataclasses import asdict

    from generate import generate
    from gpt import GPT
    from gpt_config import GPTConfig

    cfg = GPTConfig(
        vocab_size=64, block_size=64, n_embed=E, n_head=NH, n_layer=2,
        attention="flex", n_kv_head=2, position="rope",
    )  # fmt: skip
    torch.manual_seed(0)
    m = GPT(**asdict(cfg))
    prompt = torch.randint(0, 64, (2, 5))
    assert torch.equal(
        generate(m, prompt, 8, temperature=0.0),
        generate(m, prompt, 8, temperature=0.0, use_cache=True),
    )
