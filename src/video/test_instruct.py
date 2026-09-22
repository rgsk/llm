import numpy as np
import pytest

from instruct import (
    Instruct,
    checks,
    copied,
    iter_records,
    parse,
    rarity,
    render,
    required_words,
    stem,
    target_words,
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
    # idf={}: every word weighs the same, and no train stories are read
    return Instruct(tok, n_train=8, n_val=8, eval_pool=8, idf={})


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


EOT = ENDOFTEXT

# all three checkable fields at once: 12% of val records look like this
FULL = """Random sentence: The sun was hot.
Features: Dialogue
Words: cat, jump, happy
Summary: A cat jumps on a wall.
Story:
"""

# the prompt behind two SFT samples that scored the same before rarity weighting
WAFFLE = """Words: meet, waffle, new
Summary: Lily wants a waffle but gets bitten by a dog while playing with a toy car and has to go to the hospital for stitches.
Story:
"""


def test_a_prompt_gets_only_the_checks_it_asks_for():
    prompt, _ = render(parse(RECORD))  # Dialogue + Words, no Random sentence
    assert set(checks(prompt, "x")) == {"stop", "words", "dialogue"}
    assert set(checks(FULL, "x")) == {"stop", "words", "sentence", "dialogue"}
    assert set(checks("Features: Twist\nStory:\n", "x")) == {"stop"}


def test_stem_folds_inflections_but_keeps_short_words_whole():
    assert stem("wants") == stem("wanted") == "want"
    assert stem("playing") == stem("play") == "play"
    assert stem("waffles") == stem("waffle") == "waffl"
    assert stem("cats") == stem("cat") == "cat"
    assert stem("car") != stem("careful")  # "car" is not a prefix match


def test_target_words_pool_the_words_line_and_the_summary():
    # 3 from Words: + 9 content words from the Summary; stopwords are dropped
    assert target_words(WAFFLE) == {
        "meet", "new", "waffl",
        "lily", "want", "bitte", "dog", "play", "toy", "car", "hospi", "stitc",
    }  # fmt: skip
    # cat, jump, happy from Words:, and jumps folds into jump; only wall is new
    assert target_words(FULL) == {"cat", "jump", "happy", "wall"}


def test_words_is_the_fraction_of_target_words_present():
    prompt, _ = render(parse(RECORD))  # cat, jump, happy, wall
    assert checks(prompt, "the cat did jump on the wall, happy")["words"] == 1.0
    assert checks(prompt, "the cat did jump and was happy")["words"] == 3 / 4
    assert checks(prompt, "the cats jumped")["words"] == 2 / 4
    assert checks(prompt, "nothing relevant")["words"] == 0.0


def test_rarity_is_high_for_words_few_stories_use():
    idf = rarity(["the dog ran", "the dog sat", "the cat sat"])
    assert idf["the"] == pytest.approx(np.log(4 / 4))  # in all 3: weighs 0
    assert idf["dog"] == pytest.approx(np.log(4 / 3))  # in 2
    assert idf["cat"] == pytest.approx(np.log(4 / 2))  # in 1: the rarest


def test_a_rare_word_costs_more_to_miss():
    idf = {"lily": 1.0, "stitc": 3.0}
    prompt = "Summary: Lily got stitches.\nStory:\n"  # lily, stitc
    assert checks(prompt, "Lily was fine.", idf)["words"] == 1 / 4
    assert checks(prompt, "The stitches hurt.", idf)["words"] == 3 / 4
    # a stem no story in the table has weighs as much as the rarest one
    prompt = "Summary: Lily met a zebra.\nStory:\n"  # lily, met, zebra
    assert checks(prompt, "A zebra.", idf)["words"] == 3 / 7


def test_the_plot_words_are_the_rare_ones():
    # both SFT samples miss 3 of the 12 target words
    friendly_dog = "Lily wanted a new waffle. A dog came to play with her toy car. Nice to meet you."
    bitten = "Lily wanted a new waffle. A dog bit her while playing. She was bitten and went to the hospital for stitches."
    idf = {
        "lily": 1.39,
        "want": 0.59,
        "play": 0.62,
        "dog": 2.49,
        "toy": 2.32,
        "car": 2.84,
        "hospi": 4.53,
        "bitte": 5.30,
        "stitc": 6.41,
        "waffl": 6.96,
        "meet": 2.0,
        "new": 1.0,
    }  # fmt: skip, measured on 20k
    assert checks(WAFFLE, friendly_dog, idf)["words"] == pytest.approx(
        1.0 - (4.53 + 5.30 + 6.41) / sum(idf.values())
    )
    assert checks(WAFFLE, bitten, idf)["words"] == pytest.approx(
        1.0 - (2.32 + 2.84 + 2.0) / sum(idf.values())
    )
    # counted evenly they tie at 9/12; weighted, missing the plot costs 2.3x more
    assert (
        checks(WAFFLE, friendly_dog)["words"]
        == checks(WAFFLE, bitten)["words"]
        == 9 / 12
    )
    assert (
        checks(WAFFLE, friendly_dog, idf)["words"]
        < checks(WAFFLE, bitten, idf)["words"]
    )


def test_the_random_sentence_must_appear_whole():
    story = "It was a fine day. The sun was hot. Tom went out."
    assert checks(FULL, story)["sentence"] == 1.0
    assert checks(FULL, story.lower())["sentence"] == 1.0  # case is not the point
    assert checks(FULL, "The sun was very hot.")["sentence"] == 0.0  # one word off


def test_dialogue_asks_for_a_quote():
    assert checks(FULL, 'Tom said, "Hi."')["dialogue"] == 1.0
    assert checks(FULL, "Tom said hi.")["dialogue"] == 0.0


LILY = (
    "Lily's favorite toy, a robot, stopped working and needed a new battery. "
    "Her mom took her to the store to buy one."
)


def test_pasting_the_summary_in_scores_zero():
    # gold stories share at most 4 words in a row with their summary (median)
    pasted = "Once upon a time. " + LILY
    assert copied(LILY, pasted)
    assert checks(f"Summary: {LILY}\nStory:\n", pasted)["words"] == 0.0
    # 11 words in a row is allowed, 12 is not
    run = "lily's favorite toy a robot stopped working and needed a new battery"
    assert not copied(LILY, " ".join(run.split()[:11]))
    assert copied(LILY, run)


def test_checks_ignore_everything_after_eot():
    prompt, _ = render(parse(RECORD))  # cat, jump, happy, wall
    c = checks(prompt, "the cat jumped" + EOT + ' "happy"')
    assert c["words"] == 2 / 4  # "happy" is past the eot
    assert c["dialogue"] == 0.0  # and so is the quote
    assert c["stop"] == 1.0


def test_reward_is_points_out_of_what_the_prompt_offers(task):
    # stop 1 + sentence 1 + dialogue 1 + words 2 x (cat, jump of 4) = 4 of 5
    story = 'The sun was hot. "Jump!" said the cat.'
    assert task.reward(FULL, story + EOT) == pytest.approx(4 / 5)


def test_a_story_that_never_stops_scores_zero(task):
    # the hack the gate is for: run to the budget and list every word
    story = 'The sun was hot. "The cat jumps on the wall, happy."'
    assert task.reward(FULL, story + EOT) == 1.0
    assert checks(FULL, story)["words"] == 1.0  # earned everything else
    assert task.reward(FULL, story) == 0.0


def test_a_prompt_with_nothing_to_check_is_scored_on_stopping_alone(task):
    prompt = "Features: Twist\nStory:\n"
    assert task.reward(prompt, "a perfectly good story" + EOT) == 1.0
    assert task.reward(prompt, "a perfectly good story") == 0.0


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
