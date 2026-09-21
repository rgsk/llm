import numpy as np
import pytest

from instruct import (
    Instruct,
    iter_records,
    parse,
    render,
    required_words,
    usable,
)
from task import Task, windows
from tokenizer import ENDOFTEXT

RECORD = """Features: Dialogue
Words: cat, jump, happy
Summary: A cat jumps on a wall.
Story: Tom saw a cat. "Jump!" he said. The cat was happy."""

# Words before Features, and no Summary -- the corpus varies both
OTHER = """Words: dog, run
Features: Twist
Story: The dog ran home."""


@pytest.fixture(autouse=True)
def _cache_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr("instruct.CACHE", tmp_path)


@pytest.fixture
def task(tok):
    return Instruct(tok, n_train=8, n_val=8, eval_pool=8)


def test_a_record_parses_into_its_fields():
    assert parse(RECORD) == {
        "Features": "Dialogue",
        "Words": "cat, jump, happy",
        "Summary": "A cat jumps on a wall.",
        "Story": 'Tom saw a cat. "Jump!" he said. The cat was happy.',
    }


def test_fields_keep_their_source_order():
    assert list(parse(RECORD)) == ["Features", "Words", "Summary", "Story"]
    assert list(parse(OTHER)) == ["Words", "Features", "Story"]


def test_the_story_is_the_completion_and_comes_last():
    prompt, completion = render(parse(RECORD))
    assert prompt == (
        "Features: Dialogue\n"
        "Words: cat, jump, happy\n"
        "Summary: A cat jumps on a wall.\n"
        "Story:\n"
    )
    assert completion == 'Tom saw a cat. "Jump!" he said. The cat was happy.'


def test_a_record_with_no_story_is_dropped():
    assert usable(parse(RECORD))
    assert not usable(parse("Words: cat\nSummary: nothing follows."))
    assert not usable(parse("Story: a story with nothing to condition on."))


def test_records_stream_across_a_chunk_boundary(tmp_path):
    src = tmp_path / "recs.txt"
    src.write_text(f"{RECORD}\n{ENDOFTEXT}\n{OTHER}\n{ENDOFTEXT}\n")
    whole = list(iter_records(src))
    # 7 bytes at a time: every record is split many times and carried forward
    assert list(iter_records(src, chunk_bytes=7)) == whole
    assert [r["Story"] for r in whole] == [
        'Tom saw a cat. "Jump!" he said. The cat was happy.',
        "The dog ran home.",
    ]


def test_required_words_reads_the_words_line():
    assert required_words(render(parse(RECORD))[0]) == ["cat", "jump", "happy"]
    assert required_words("Summary: no words line here\nStory:\n") == []


def test_reward_is_the_fraction_of_required_words_present(task):
    prompt, _ = render(parse(RECORD))
    assert task.reward(prompt, "the cat did jump and was happy") == 1.0
    assert task.reward(prompt, "the cat did jump") == 2 / 3
    assert task.reward(prompt, "nothing relevant") == 0.0


def test_a_required_word_counts_inside_an_inflection(task):
    prompt, _ = render(parse(RECORD))  # cat, jump, happy
    assert task.reward(prompt, "the cat jumped and was happy") == 1.0  # jump/jumped
    assert task.reward(prompt, "the cat jumped happily") == 2 / 3  # happy/happi-ly


def test_reward_ignores_everything_after_eot(task):
    prompt, _ = render(parse(RECORD))
    # "jumped" counts, "happy" is past the eot and does not
    assert task.reward(prompt, "the cat jumped" + ENDOFTEXT + " happy") == 2 / 3


def test_a_prompt_with_no_words_line_is_not_checkable(task):
    # a third of the corpus; inside a GRPO group this is a constant, not a signal
    assert task.reward("Summary: something\nStory:\n", "a perfectly good story") == 0.0


@pytest.mark.slow
def test_build_packs_prompt_then_story_then_eot(task, tok):
    p = task.build("val")
    assert len(p.starts) == 8
    s, e = p.starts[0], p.starts[1]
    ids, m = p.ids[s:e], p.mask[s:e]
    assert tok.decode(ids[~m].tolist()).endswith("Story:\n")  # the prompt
    assert ids[-1] == p.eot_id and m[-1]  # eot ends the example, and is trained
    assert not m[: (~m).sum()].any()  # the prompt is one unbroken False run


@pytest.mark.slow
def test_build_is_cached(task, tmp_path):
    a = task.build("val")
    assert list(tmp_path.glob("val_8_*.npz"))
    b = task.build("val")  # second call reads the npz
    assert np.array_equal(a.ids, b.ids) and np.array_equal(a.mask, b.mask)


@pytest.mark.slow
def test_windows_open_on_a_prompt(task, tok):
    x, _, keep = windows(task.build("val"), 4, 128, np.random.default_rng(0))
    for row in x:
        assert tok.decode(row[:4].tolist()).split(":")[0] in {
            "Random sentence",
            "Features",
            "Words",
            "Summary",
        }
    assert not keep[:, 0].any()  # no window starts inside a story


@pytest.mark.slow
def test_eval_prompts_are_all_checkable(task):
    prompts = task.prompts()
    assert len(prompts) == 8
    assert all(required_words(p) for p in prompts)


def test_instruct_is_a_task(task):
    assert isinstance(task, Task)
