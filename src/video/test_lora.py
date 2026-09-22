from dataclasses import asdict

import pytest
import torch

from adamw import AdamW, decay_groups
from checkpoint import save_checkpoint
from gpt import GPT
from gpt_config import GPTConfig
from linear import Linear
from lora import (
    ATTN_TARGETS,
    LoRALinear,
    apply_lora,
    load_adapter,
    load_lora,
    lora_state_dict,
    merge_lora,
    param_counts,
    save_lora,
    walk,
)
from residual_proj import ResidualProj


@pytest.fixture
def gpt_cfg():
    return GPTConfig(
        vocab_size=128,
        block_size=32,
        n_embed=64,
        n_head=4,
        n_layer=2,
        attention="sdpa",
        norm="rms",
        ffn="gated",  # gate_up/down, so "attention only" has something to exclude
        position="rope",
    )


@pytest.fixture
def model(gpt_cfg):
    torch.manual_seed(0)
    return GPT(**asdict(gpt_cfg))


@pytest.fixture
def x():
    return torch.randint(0, 128, (2, 16))


def lora_layers(model):
    return [c for m in walk(model) for _, c in m.named_children()
            if isinstance(c, LoRALinear)]  # fmt: skip


def test_a_fresh_adapter_leaves_the_model_bit_identical(model, x):
    before = model(x)
    apply_lora(model, r=4)
    assert (model(x) - before).abs().max() == 0  # B is zeros, so the delta is zeros


def test_the_factors_are_r_by_in_and_out_by_r(model):
    apply_lora(model, r=4, alpha=8.0)
    qkv = model.blocks[0].attn.qkv
    assert qkv.base.in_features == 64 and qkv.base.out_features == 192  # 3 * n_embed
    assert tuple(qkv.A.shape) == (4, 64)
    assert tuple(qkv.B.shape) == (192, 4)
    assert qkv.scale == 2.0  # alpha / r
    assert (qkv.B == 0).all()
    assert qkv.A.abs().max() > 0


def test_alpha_defaults_to_twice_r(model):
    apply_lora(model, r=8)
    assert model.blocks[0].attn.qkv.scale == 2.0


def test_only_qkv_and_proj_are_wrapped(model):
    apply_lora(model, r=4)
    wrapped = [
        name
        for m in walk(model)
        for name, c in m.named_children()
        if isinstance(c, LoRALinear)
    ]
    assert sorted(wrapped) == ["proj", "proj", "qkv", "qkv"]  # 2 blocks x 2 targets
    ffn = model.blocks[0].ffwd
    assert isinstance(ffn.gate_up, Linear) and not isinstance(ffn.gate_up, LoRALinear)
    assert isinstance(ffn.down, Linear) and not isinstance(ffn.down, LoRALinear)


def test_proj_is_wrapped_even_though_it_is_a_residualproj(model):
    assert isinstance(model.blocks[0].attn.proj, ResidualProj)
    apply_lora(model, r=4)
    assert isinstance(model.blocks[0].attn.proj, LoRALinear)


def test_a_narrower_target_list_wraps_less(model):
    apply_lora(model, r=4, targets=("qkv",))
    assert isinstance(model.blocks[0].attn.qkv, LoRALinear)
    assert not isinstance(model.blocks[0].attn.proj, LoRALinear)


def test_trainable_is_exactly_a_and_b(model):
    apply_lora(model, r=4)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(names) == 8  # 4 wrapped layers x (A, B)
    assert all(n.endswith((".A", ".B")) for n in names)
    assert not model.token_embedding_table.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    assert not model.blocks[0].attn.qkv.base.weight.requires_grad


def test_the_trainable_count_is_the_rank_arithmetic(model, gpt_cfg):
    apply_lora(model, r=4)
    trainable, total = param_counts(model)
    # per block: qkv 4*64 + 192*4 = 1024, proj 4*64 + 64*4 = 512
    assert trainable == 2 * (1024 + 512) == 3072
    assert total == sum(p.numel() for p in GPT(**asdict(gpt_cfg)).parameters()) + 3072


def test_b_moves_first_and_a_only_once_b_is_nonzero(model, x):
    apply_lora(model, r=4)
    qkv = model.blocks[0].attn.qkv
    model(x).sum().backward()
    assert qkv.B.grad.abs().max() > 0  # A is random, so B has a gradient at step 0
    assert qkv.A.grad.abs().max() == 0  # B is still zero, so A has none yet
    assert qkv.base.weight.grad is None

    opt = AdamW(decay_groups(model, 0.1), lr=1e-2)
    opt.step()
    opt.zero_grad()
    model(x).sum().backward()
    assert qkv.A.grad.abs().max() > 0  # B is nonzero now, so A trains too


def test_a_step_moves_the_adapter_and_nothing_else(model, x):
    apply_lora(model, r=4)
    qkv = model.blocks[0].attn.qkv
    base_before = qkv.base.weight.detach().clone()
    emb_before = model.token_embedding_table.weight.detach().clone()
    b_before = qkv.B.detach().clone()

    opt = AdamW(decay_groups(model, 0.1), lr=1e-2)  # the groups sft.py builds
    model(x).sum().backward()
    opt.step()

    assert (qkv.B - b_before).abs().max() > 0
    # frozen params carry no grad, so neither the update nor weight decay reaches them
    assert torch.equal(qkv.base.weight, base_before)
    assert torch.equal(model.token_embedding_table.weight, emb_before)


def test_wrapping_a_wrapped_model_is_refused(model):
    apply_lora(model, r=4)
    with pytest.raises(AssertionError, match="no unwrapped Linear"):
        apply_lora(model, r=4)


def test_rank_zero_is_refused(model):
    with pytest.raises(AssertionError, match="r must be at least 1"):
        apply_lora(model, r=0)


def test_merging_reproduces_the_wrapped_forward(model, x):
    apply_lora(model, r=4)
    with torch.no_grad():  # fake training, or the merge is a no-op
        for layer in lora_layers(model):
            layer.B.normal_()
    wrapped = model(x)

    merge_lora(model)
    assert lora_layers(model) == []  # the wrappers are gone
    assert isinstance(model.blocks[0].attn.proj, ResidualProj)  # not a plain Linear
    assert (model(x) - wrapped).abs().max() < 1e-5


def test_the_checkpoint_holds_the_adapter_not_the_model(model, gpt_cfg, tmp_path):
    apply_lora(model, r=4)
    base = tmp_path / "base.pt"
    path = tmp_path / "adapter.pt"
    save_lora(path, model, gpt_cfg, base_ckpt=base, r=4, alpha=8.0, step=7, comp=1.5)

    saved = torch.load(path, weights_only=False)
    assert set(saved["model"]) == set(lora_state_dict(model))
    assert len(saved["model"]) == 8
    assert "token_embedding_table.weight" not in saved["model"]
    assert saved["lora"] == {
        "base_ckpt": str(base),
        "r": 4,
        "alpha": 8.0,
        "targets": list(ATTN_TARGETS),
    }
    assert saved["step"] == 7 and saved["comp"] == 1.5


def test_a_round_trip_reproduces_the_outputs(model, gpt_cfg, tmp_path, x):
    base = tmp_path / "base.pt"
    save_checkpoint(base, model, gpt_cfg, step=0, val_loss=1.0)

    apply_lora(model, r=4)
    with torch.no_grad():
        for layer in lora_layers(model):
            layer.B.normal_()
    model.eval()
    expected = model(x)

    path = tmp_path / "adapter.pt"
    save_lora(path, model, gpt_cfg, base_ckpt=base, r=4, alpha=8.0, step=7)
    reloaded, meta = load_lora(path)
    reloaded.eval()
    assert (reloaded(x) - expected).abs().max() == 0
    assert meta["step"] == 7 and "model" not in meta


def test_an_adapter_file_is_far_smaller_than_the_base(model, gpt_cfg, tmp_path):
    base = tmp_path / "base.pt"
    save_checkpoint(base, model, gpt_cfg, step=0, val_loss=1.0)
    apply_lora(model, r=4)
    path = tmp_path / "adapter.pt"
    save_lora(path, model, gpt_cfg, base_ckpt=base, r=4, alpha=8.0)
    assert path.stat().st_size < base.stat().st_size / 10


def test_an_adapter_hot_swaps_onto_a_wrapped_model(model, gpt_cfg, x):
    apply_lora(model, r=4)
    with torch.no_grad():
        for layer in lora_layers(model):
            layer.B.normal_()
    trained = lora_state_dict(model)
    expected = model(x)

    torch.manual_seed(0)
    fresh = apply_lora(GPT(**asdict(gpt_cfg)), r=4)
    assert (fresh(x) - expected).abs().max() > 0
    load_adapter(fresh, trained)
    assert (fresh(x) - expected).abs().max() == 0


def test_an_adapter_of_the_wrong_rank_is_refused(model, gpt_cfg, x):
    apply_lora(model, r=4)
    trained = lora_state_dict(model)
    wrong = apply_lora(GPT(**asdict(gpt_cfg)), r=8)
    with pytest.raises(AssertionError, match=r"have \(8, 64\), got \(4, 64\)"):
        load_adapter(wrong, trained)


def test_an_unwrapped_model_says_so(model, gpt_cfg):
    wrapped = apply_lora(GPT(**asdict(gpt_cfg)), r=4)
    with pytest.raises(AssertionError, match="call apply_lora first"):
        load_adapter(model, lora_state_dict(wrapped))


def test_an_adapter_trained_elsewhere_finds_the_base_by_name(
    model, gpt_cfg, tmp_path, x, monkeypatch
):
    real = tmp_path / "base.pt"
    save_checkpoint(real, model, gpt_cfg, step=0, val_loss=1.0)
    monkeypatch.setattr("lora.CKPT_DIR", tmp_path)

    apply_lora(model, r=4)
    model.eval()
    expected = model(x)
    path = tmp_path / "adapter.pt"
    # the path a pod would have recorded: right basename, wrong machine
    save_lora(
        path, model, gpt_cfg, base_ckpt="/workspace/llm/artifacts/checkpoints/base.pt",
        r=4, alpha=8.0,
    )  # fmt: skip
    reloaded, _ = load_lora(path)
    reloaded.eval()
    assert (reloaded(x) - expected).abs().max() == 0


def test_an_adapter_whose_base_is_nowhere_says_both_paths(model, gpt_cfg, tmp_path):
    apply_lora(model, r=4)
    path = tmp_path / "adapter.pt"
    save_lora(path, model, gpt_cfg, base_ckpt="/nope/missing.pt", r=4, alpha=8.0)
    with pytest.raises(AssertionError, match="not here, nor"):
        load_lora(path)


def test_a_plain_checkpoint_is_not_an_adapter(model, gpt_cfg, tmp_path):
    path = tmp_path / "plain.pt"
    save_checkpoint(path, model, gpt_cfg, step=0, val_loss=1.0)
    with pytest.raises(AssertionError, match="not an adapter checkpoint"):
        load_lora(path)
