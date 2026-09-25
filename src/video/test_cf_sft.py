import pytest
import torch

from cf_sft import IGNORE, collate, encode

transformers = pytest.importorskip("transformers")
NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained(NAME)
    except OSError:
        pytest.skip(f"{NAME} tokenizer not cached")


SOL = "#include <bits/stdc++.h>\nint main() { std::cout << 42; }"


def test_encode_masks_the_prompt_and_keeps_the_answer(tok):
    ids, labels = encode(tok, "Print 42.", SOL)
    answer = [t for t in labels if t != IGNORE]
    assert tok.decode(ids[: len(ids) - len(answer)]).endswith("<|im_start|>assistant\n")
    assert tok.decode(answer) == f"```cpp\n{SOL}\n```<|im_end|>"


def test_encode_mask_is_one_prompt_block_then_one_answer_block(tok):
    _, labels = encode(tok, "Print 42.", SOL)
    keep = [t != IGNORE for t in labels]
    n_prompt = keep.index(True)
    assert keep == [False] * n_prompt + [True] * (len(keep) - n_prompt)


def test_collate_right_pads_and_masks_the_padding():
    rows = [([5, 6, 7], [IGNORE, 6, 7]), ([8, 9], [IGNORE, 9])]
    x, y, att = collate(rows, pad_id=0, device="cpu")
    assert x.tolist() == [[5, 6, 7], [8, 9, 0]]
    assert y.tolist() == [[IGNORE, 6, 7], [IGNORE, 9, IGNORE]]
    assert att.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert att.dtype == torch.long
