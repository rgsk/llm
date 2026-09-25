import pytest
import torch

from cf_sft import IGNORE, collate
from grpo import advantages, pg_loss, trim

transformers = pytest.importorskip("transformers")


def test_advantage_is_reward_against_its_own_group():
    r = torch.tensor([1.0, 0, 0, 0, 1, 1, 0, 0])
    assert advantages(r, group=4).tolist() == pytest.approx(
        [1.5, -0.5, -0.5, -0.5, 0.866, 0.866, -0.866, -0.866], abs=1e-3
    )


def test_a_group_that_all_passes_or_all_fails_has_no_advantage():
    r = torch.tensor([1.0, 1, 1, 1, 0, 0, 0, 0])
    assert advantages(r, group=4).tolist() == [0.0] * 8


def test_one_rare_pass_earns_more_than_one_common_pass():
    rare = advantages(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), 8)[0]
    common = advantages(torch.tensor([1.0, 1, 1, 1, 1, 1, 1, 0]), 8)[0]
    assert (round(rare.item(), 2), round(common.item(), 2)) == (2.47, 0.35)


def test_trim_keeps_the_stop_token_and_drops_what_follows():
    assert trim([5, 6, 99, 0, 0], stop={99}) == [5, 6, 99]
    assert trim([5, 6, 7], stop={99}) == [5, 6, 7]


def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    return transformers.Qwen2ForCausalLM(cfg)


def completion_logp(model, row) -> float:
    x, y, att = collate([row], pad_id=0, device="cpu")
    with torch.no_grad():
        return -pg_loss(model, x, y, att, torch.tensor([1.0]), 1, 1.0).item()


def step_with(adv: float) -> tuple[float, float]:
    model = tiny_model()
    row = ([3, 4, 5, 6, 7], [IGNORE, IGNORE, 5, 6, 7])  # prompt 3 4, completion 5 6 7
    before = completion_logp(model, row)
    x, y, att = collate([row], pad_id=0, device="cpu")
    pg_loss(model, x, y, att, torch.tensor([adv]), 3, 1.0).backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad
    return before, completion_logp(model, row)


def test_positive_advantage_makes_the_completion_more_likely():
    before, after = step_with(+1.0)
    assert after > before


def test_negative_advantage_makes_the_completion_less_likely():
    before, after = step_with(-1.0)
    assert after < before


def test_pg_loss_counts_only_completion_tokens():
    model = tiny_model()
    x, y, att = collate([([3, 4, 5], [IGNORE, IGNORE, 5])], pad_id=0, device="cpu")
    with torch.no_grad():
        logits = model(input_ids=x).logits[0, 1]  # position 1 predicts token 5
        want = logits.log_softmax(-1)[5].item()
        got = -pg_loss(model, x, y, att, torch.tensor([1.0]), 1, 1.0).item()
    assert got == pytest.approx(want, abs=1e-5)
