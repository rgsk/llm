"""A pytest walkthrough over the modules in this folder.

Nothing here is new machinery -- it is the same `assert` you already write in
each module's test(), handed to a runner that isolates, names and reports each
one. Run it:

    uv run --with pytest python -m pytest src/minimal_experiments/testing.py -v

    -v                          one line per test
    -x                          stop at the first failure
    -q                          quiet
    -k rope                     only tests whose name matches "rope"
    -m "not slow"               skip tests marked slow
    --lf                        re-run only last-run failures
    -s                          don't swallow print()

The filename is `testing.py`, not `test_*.py`, so pytest won't auto-collect it
in a bare `pytest` run -- pass the path, as above. The `slow` marker prints an
"unknown mark" warning until it's registered in pyproject.toml; harmless.
"""

from dataclasses import asdict
from typing import get_args

import pytest
import torch
from rope2 import (
    GPT,
    FlashAttention,
    GPTConfig,
    GQAttention,
    apply_rope,
    batch_loss,
    rope_inv_freq,
    rope_tables,
)

import kv_cache as kvc

# ----------------------------------------------------------------------------
# fixtures: setup that several tests share.
#
# A fixture is a function whose NAME a test asks for as an argument; pytest
# calls it and passes the result in. The point is not saving keystrokes -- it is
# that the fixture re-runs for every test, so no test can be polluted by what a
# previous one did to the object. (Module-level `model = GPT(...)` would be
# shared mutable state; one test calling .backward() would leak grads into the
# next.)
# ----------------------------------------------------------------------------


@pytest.fixture
def cfg() -> GPTConfig:
    """Deliberately tiny: these tests check correctness, not learning."""
    return GPTConfig(
        vocab_size=65, block_size=16, n_embed=32, n_head=4, n_layer=2, n_kv_head=2
    )


@pytest.fixture
def model(cfg: GPTConfig) -> GPT:
    # a fixture can request another fixture by name, same rule
    torch.manual_seed(0)
    return GPT(**asdict(cfg))


@pytest.fixture
def float64():
    """Everything built inside a test using this fixture is float64.

    A fixture that yields is setup/teardown: pytest runs up to the yield, hands
    control to the test, then runs the rest -- even if the test failed. Without
    that restore a single failure here would silently make every later test
    run in float64.

    Why bother: float32 gives ~1e-7 of headroom, so an exact-equivalence check
    has to be written as `< 1e-5` and a real bug worth 1e-6 hides under the
    tolerance. float64 pushes the floor to ~1e-16 and the same check becomes
    `< 1e-12`, which noise cannot reach.

    It does NOT reach the rope tables: rope_inv_freq() and rope_tables() pass
    dtype=torch.float32 explicitly, so cos/sin carry float32 rounding no matter
    what the default is, and upcasting afterwards cannot recover the lost bits.
    Every check downstream of the tables is therefore written at 1e-6, not
    1e-12 -- the tests below say so where it bites. (Found by writing 1e-12
    first and watching it fail at 3.9e-07, which is how a tolerance should be
    arrived at: measured, not guessed.)
    """
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


# ----------------------------------------------------------------------------
# 1. plain asserts -- pytest's only real syntax
#
# pytest rewrites the bytecode of `assert` inside test files so a failure shows
# both operands. `assert freq[0] == 1.0` failing prints the actual value, which
# is the single biggest thing it buys over a hand-rolled runner.
# ----------------------------------------------------------------------------


def test_rope_inv_freq_decays():
    hs = 16
    freq = rope_inv_freq(hs)
    assert freq.shape == (hs // 2,)  # one angular frequency per rotating pair
    assert freq[0] == 1.0  # pair 0 turns a full radian per position
    assert (freq.diff() < 0).all()  # later pairs turn slower: long-range context
    assert freq[-1] == pytest.approx(10000.0 ** (-(hs - 2) / hs))


# ----------------------------------------------------------------------------
# 2. parametrize -- one test body, N cases, N separate results
#
# Reported as test_rope_tables_shape[8], [...16], [...64]. A loop inside one
# test would stop at the first bad size and hide the rest; these are independent
# runs, and `-k "rope_tables and 64"` selects just the one that broke.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("head_size", [8, 16, 64])
@pytest.mark.parametrize("block_size", [1, 32])
def test_rope_tables_shape(head_size: int, block_size: int):
    # stacked parametrize = cartesian product, so this is 6 tests
    cos, sin = rope_tables(block_size, head_size)
    assert cos.shape == sin.shape == (block_size, head_size // 2)
    # it is a rotation. 1e-6, not 1e-12: the tables are built in float32
    assert (cos**2 + sin**2 - 1).abs().max() < 1e-6
    assert torch.equal(cos[0], torch.ones(head_size // 2))  # position 0 = identity
    assert torch.equal(sin[0], torch.zeros(head_size // 2))


# ----------------------------------------------------------------------------
# 3. pytest.raises -- assert that bad input is rejected
#
# Half of what an assert in library code is worth is that it fires. `match` is a
# regex against the message, so the test fails if a DIFFERENT AssertionError
# happens to be raised on the way -- without it, a typo'd attribute access would
# also "pass".
# ----------------------------------------------------------------------------


def test_rope_rejects_odd_head_size():
    with pytest.raises(AssertionError, match="rotate in pairs"):
        rope_inv_freq(15)


def test_gpt_rejects_sequence_longer_than_block_size(model: GPT, cfg: GPTConfig):
    idx = torch.randint(cfg.vocab_size, (2, cfg.block_size + 1))
    with pytest.raises(AssertionError, match="outgrew block_size"):
        model(idx)


def test_make_attention_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown attention"):
        kvc.make_attention("linear", 32, 4, 16)  # type: ignore[arg-type]


def test_mha_only_impls_reject_kv_groups():
    # "flash" has no grouping, so asking for 2 kv heads must be refused rather
    # than silently ignored -- silently ignored is how you train the wrong model
    with pytest.raises(AssertionError, match="MHA only"):
        kvc.make_attention("flash", 32, 4, 16, n_kv_head=2)


# ----------------------------------------------------------------------------
# 4. property tests -- the interesting case, where there is no oracle
#
# For LayerNorm you diff against nn.LayerNorm. For RoPE there is no reference
# impl in torch, so instead assert the mathematical property the thing exists
# for: <R_m q, R_n k> depends only on m - n. Rotate one fixed q into every
# position, same for k, and the [T, T] matrix of dot products must be constant
# along each diagonal.
#
# This is the test to write for a custom kernel, a sampler, a policy -- anything
# where "what should it equal" has no answer but "what must stay true" does.
# ----------------------------------------------------------------------------


def _rope_dot_matrix(hs: int, T: int, rope=apply_rope) -> torch.Tensor:
    """dots[m, n] = <rope(q, at m), rope(k, at n)> for one fixed q, k."""
    cos, sin = rope_tables(T, hs)
    q, k = torch.randn(hs), torch.randn(hs)
    # every row is the same vector, so row m comes out rotated by position m
    Q = rope(q.expand(T, hs).contiguous(), cos, sin)
    K = rope(k.expand(T, hs).contiguous(), cos, sin)
    return Q @ K.T


def test_rope_dot_product_depends_only_on_distance(float64):
    torch.manual_seed(0)
    dots = _rope_dot_matrix(hs=16, T=12)
    for offset in range(-11, 12):
        d = dots.diagonal(offset=offset)
        # float32 tables set the floor (see the float64 fixture); measured
        # worst case is ~4e-7, and the mutant below misses by ~1e-1, so there
        # is still four orders of separation between noise and a real break
        assert (d - d[0]).abs().max() < 1e-6, f"offset {offset} is not constant"


def test_rope_property_catches_a_sign_flip(float64):
    """A test that cannot fail is worth nothing -- so break the code and watch.

    Flipping one sign turns the 2x2 rotation into a reflection. Shapes still
    match, values still look plausible, training would still descend. Only the
    relative-position property notices: a reflection's dot product carries an
    m + n term.
    """

    def reflect(x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos + x2 * sin  # sign flipped
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    torch.manual_seed(0)
    dots = _rope_dot_matrix(hs=16, T=12, rope=reflect)
    spread = max((dots.diagonal(o) - dots.diagonal(o)[0]).abs().max() for o in (1, 3))
    assert spread > 1e-3, "the mutant was not distinguishable"


def test_rope_property_does_not_pin_down_the_pairing(float64):
    """And the flip side: know what your test does NOT constrain.

    Llama pairs dim j with j + hs/2 instead of the interleaved 2j, 2j+1 here.
    That is a different function -- swap one impl for the other under trained
    weights and the model breaks -- but it is still a rotation, so the property
    above passes for both. Interop with a checkpoint needs a golden-value test
    against that checkpoint; no invariant can substitute.
    """

    def split_half(x, cos, sin):
        hs = x.shape[-1]
        x1, x2 = x[..., : hs // 2], x[..., hs // 2 :]
        out = torch.empty_like(x)
        out[..., : hs // 2] = x1 * cos - x2 * sin
        out[..., hs // 2 :] = x1 * sin + x2 * cos
        return out

    torch.manual_seed(0)
    dots = _rope_dot_matrix(hs=16, T=12, rope=split_half)
    for offset in range(-11, 12):
        d = dots.diagonal(offset=offset)
        assert (d - d[0]).abs().max() < 1e-6


# ----------------------------------------------------------------------------
# 5. differential testing -- two impls of one thing must agree
#
# The strongest test available when you do have a second implementation, and
# the reason to keep the old ones around instead of deleting them. Note
# load_state_dict is doing double duty: it makes the comparison fair, and it
# asserts the two impls expose the same parameter names and shapes.
# ----------------------------------------------------------------------------


def test_gqa_reduces_to_mha_when_every_head_has_its_own_kv(float64):
    """n_kv_head == n_head is the degenerate case; SDPA's plain kernel is truth."""
    torch.manual_seed(0)
    E, NH, T = 32, 4, 8
    mine, ref = GQAttention(E, NH, NH), FlashAttention(E, NH)
    mine.load_state_dict(ref.state_dict())

    cos, sin = rope_tables(T, E // NH)
    x = torch.randn(2, T, E)
    # assert_close beats `(a - b).abs().max() < tol`: on failure it prints the
    # mismatch count, the worst element and its index, not just False
    torch.testing.assert_close(mine(x, cos, sin), ref(x, cos, sin), atol=1e-12, rtol=0)


@pytest.mark.parametrize("kind", get_args(kvc.Attention))
def test_all_attention_impls_agree(kind: kvc.Attention, float64):
    """Every impl in kv_cache.py, against the one that is hardest to get wrong.

    get_args() reads the cases off the Literal, so adding a sixth attention to
    that type adds a test here for free -- and a new impl arrives already
    covered rather than trusted.
    """
    torch.manual_seed(0)
    base = {
        "vocab_size": 65,
        "block_size": 16,
        "n_embed": 32,
        "n_head": 4,
        "n_layer": 2,
    }
    ref = kvc.GPT(**base, n_kv_head=4, attention="flash")
    got = kvc.GPT(**base, n_kv_head=4, attention=kind)
    got.load_state_dict(ref.state_dict())  # also checks the keys line up

    idx = torch.randint(65, (2, 12))
    torch.testing.assert_close(got(idx), ref(idx), atol=1e-12, rtol=0)


def test_kv_cache_generate_matches_recompute():
    """Incremental decode must produce exactly what full recompute produces.

    The bug this catches is the whole reason a KV cache is scary: the cached
    path takes a different code branch per step (T=1 queries, a shifted causal
    mask, a position offset) and a mistake there shows up as slightly worse
    samples, never as a crash. Seeding before each call makes the multinomial
    draws line up, so any difference is the cache and nothing else.
    """
    torch.manual_seed(0)
    model = kvc.GPT(vocab_size=65, block_size=16, n_embed=32, n_head=4, n_layer=2)
    prompt = torch.randint(65, (2, 4))

    torch.manual_seed(1)
    slow = kvc.generate(model, prompt, max_new_tokens=8, use_cache=False)
    torch.manual_seed(1)
    fast = kvc.generate(model, prompt, max_new_tokens=8, use_cache=True)
    assert torch.equal(slow, fast)


def test_kv_cache_generate_refuses_to_overflow_the_context():
    torch.manual_seed(0)
    model = kvc.GPT(vocab_size=65, block_size=16, n_embed=32, n_head=4, n_layer=2)
    prompt = torch.randint(65, (1, 10))
    with pytest.raises(AssertionError, match="use_cache needs"):
        kvc.generate(model, prompt, max_new_tokens=20, use_cache=True)


# ----------------------------------------------------------------------------
# 6. the cheap structural tests -- boring, and they catch real refactors
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_gqa_shrinks_kv_without_touching_the_output_shape(n_kv_head: int):
    E, NH, T = 32, 4, 8
    attn = GQAttention(E, NH, n_kv_head)
    cos, sin = rope_tables(T, E // NH)
    x = torch.randn(2, T, E)
    assert attn(x, cos, sin).shape == x.shape
    # the whole point of GQA: the qkv projection narrows as kv heads drop
    assert attn.qkv.weight.shape[0] == E + 2 * n_kv_head * (E // NH)


def test_forward_returns_logits_per_position(model: GPT, cfg: GPTConfig):
    B, T = 2, 12
    logits = model(torch.randint(cfg.vocab_size, (B, T)))
    assert logits.shape == (B, T, cfg.vocab_size)
    assert logits.isfinite().all()


def test_weight_tying_is_one_tensor(model: GPT):
    # `is`, not equality: a copy would pass an allclose check and then drift
    # apart on the first optimizer step
    assert model.lm_head.weight is model.token_embedding_table.weight


def test_backward_reaches_every_parameter(model: GPT, cfg: GPTConfig):
    """Catches the dead branch: a module built, registered, and never called.

    grad is None for exactly the parameters the loss did not depend on, so this
    is a one-line check that the graph really is wired end to end.
    """
    x = torch.randint(cfg.vocab_size, (2, 8))
    y = torch.randint(cfg.vocab_size, (2, 8))
    batch_loss(model, x, y).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.isfinite().all(), f"{name} gradient has nan/inf"
        assert p.grad.abs().max() > 0, f"{name} gradient is all zero"


def test_untrained_loss_is_close_to_uniform(model: GPT, cfg: GPTConfig):
    """The one number worth eyeballing at init: -ln(1/V).

    Off by a lot and the init scale or the logit scaling is wrong -- and you
    want to know that before burning a training run, not after.
    """
    torch.manual_seed(0)
    x = torch.randint(cfg.vocab_size, (8, cfg.block_size))
    y = torch.randint(cfg.vocab_size, (8, cfg.block_size))
    loss = batch_loss(model, x, y).item()
    assert loss == pytest.approx(torch.tensor(cfg.vocab_size).log().item(), abs=0.15)


# ----------------------------------------------------------------------------
# 7. the slow one -- can it learn at all
#
# Marked so `-m "not slow"` skips it: the tests above run in a second and belong
# in the edit loop, this one takes ~10s and belongs in the run-before-you-commit
# pass. Overfitting a single batch is the canonical ML smoke test -- it is the
# weakest possible learning claim, which is exactly why a failure is unambiguous
# (a model that cannot memorise 4x16 tokens is broken, not undertrained).
# ----------------------------------------------------------------------------


@pytest.mark.slow
def test_overfits_a_single_batch(model: GPT, cfg: GPTConfig):
    torch.manual_seed(0)
    x = torch.randint(cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(cfg.vocab_size, (4, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = batch_loss(model, x, y).item()
    for _ in range(300):
        loss = batch_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert loss.item() < first / 10, f"{first:.3f} -> {loss.item():.3f}, not learning"
