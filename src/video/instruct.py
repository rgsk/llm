"""The second Task: TinyStoriesInstruct. Fields in, a story out.

    Features: Dialogue
    Words: cat, jump, happy
    Summary: Tom and Anna go on a holiday.
    Story:
    <- the completion starts here

Reverse tested the trainer; this one is the job. The base writes fluent text
already (0.942 bpc on these stories, teacher-forced) but cannot produce the
format and emits eot in 4-8% of continuations, so format and termination are
what SFT has to buy.

Fields keep their source order, which varies per record -- that is what teaches
order-independence. Three fields are checkable by string match, which is what
makes this a DPO/GRPO target: `Words` (61% of val records), `Random sentence`
(30%, verbatim in its gold story 100% of the time) and `Features: Dialogue`
(35%, a quote in 91% of gold stories vs 30% without). The Summary (100%) is
checked only by its content words, pooled with `Words` -- whether the plot is
followed needs a model to judge.

2.5M records at 2.66 GB, so build() streams and caches. The first call costs
minutes, the rest load a .npz.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from generate import generate
from paths import DATA_ROOT
from task import Packed
from tokenizer import ENDOFTEXT, tokenizer_for

HF_DATASET = "roneneldan/TinyStoriesInstruct"
FILES = {
    "train": "TinyStories-Instruct-train.txt",
    "val": "TinyStories-Instruct-valid.txt",
}
FIELDS = ("Random sentence:", "Features:", "Words:", "Summary:", "Story:")
SEP = ENDOFTEXT
CACHE = DATA_ROOT / "instruct"


def parse(record: str) -> dict[str, str]:
    """One record's text -> {field: value}, in source order."""
    fields: dict[str, list[str]] = {}
    cur = None
    for line in record.split("\n"):
        hit = next((f for f in FIELDS if line.startswith(f)), None)
        if hit:
            cur = hit[:-1]
            fields[cur] = [line[len(hit) :].strip()]
        elif cur:
            fields[cur].append(line.strip())
    return {k: "\n".join(v).strip() for k, v in fields.items()}


def usable(f: dict[str, str]) -> bool:
    """A story to learn, and at least one field to condition on."""
    return bool(f.get("Story")) and any(f[k] for k in f if k != "Story")


def render(f: dict[str, str]) -> tuple[str, str]:
    """-> (prompt, completion). Story forced last, everything else in source order."""
    prompt = "".join(f"{k}: {f[k]}\n" for k in f if k != "Story") + "Story:\n"
    return prompt, f["Story"]


def required_words(prompt: str) -> list[str]:
    """The `Words:` line, or [] -- a third of the records do not have one."""
    m = re.search(r"^Words: (.+)$", prompt, re.MULTILINE)
    return [w.strip().lower() for w in m.group(1).split(",") if w.strip()] if m else []


# function words carry no plot; what is left is names, objects and verbs
_STOP = (
    "a an the and or but to of in on at for with was were is are he she it "
    "they his her their them him had has have be been this that from as by up "
    "so then when one day who what very did not while gets got get after "
    "before into out about because all some there could would"
)
STOPWORDS = frozenset(_STOP.split())
COPY_RUN = 12  # words in a row shared with the summary: 0.6% of gold stories
# what each check is worth; a prompt is scored out of the checks it has
POINTS = {"stop": 1, "sentence": 1, "dialogue": 1, "words": 2}


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


def stem(w: str) -> str:
    """wants/wanted -> want, playing -> play, waffles -> waffl, cats -> cat.
    Short words are never cut to a prefix, so car does not match careful."""
    if len(w) > 4:
        return re.sub(r"(ing|ed|es|s)$", "", w)[:5]
    return w[:-1] if len(w) == 4 and w.endswith("s") else w


def target_words(prompt: str) -> set[str]:
    """The `Words:` line and the Summary's content words, as one set of stems."""
    summary = parse(prompt).get("Summary", "")
    content = [w for w in _words(summary) if w not in STOPWORDS and len(w) > 2]
    return {stem(w) for w in required_words(prompt) + content}


def copied(summary: str, story: str, n: int = COPY_RUN) -> bool:
    """Does the story share n words in a row with the summary?"""
    a, b = _words(summary), _words(story)
    grams = {tuple(a[i : i + n]) for i in range(len(a) - n + 1)}
    return any(tuple(b[i : i + n]) in grams for i in range(len(b) - n + 1))


def checks(
    prompt: str, completion: str, idf: dict[str, float] | None = None
) -> dict[str, float]:
    """Every check this prompt supports -> a score in [0, 1]. `stop` always
    applies; the rest only where the prompt asks for them, because a quote in
    a story that was not asked for dialogue is not wrong -- 30% of gold stories
    without the feature have one.

    `words` is the rarity-weighted fraction of target_words present: missing
    `stitches` costs more than missing `lily`, which is in most stories. With
    no idf every word weighs the same. It checks words, not plot -- scoring the
    plot needs a model, the PMI version after the DPO loop. Pasting the summary
    in would score 1.0, so a copied run zeroes it."""
    f = parse(prompt)
    story = completion.split(ENDOFTEXT)[0].lower()
    out = {"stop": float(ENDOFTEXT in completion)}
    if want := target_words(prompt):
        # a stem no story in the table has is as rare as the table can say
        top = max(idf.values(), default=1.0) if idf else 1.0
        weight = {w: idf.get(w, top) if idf else 1.0 for w in want}
        have = {stem(w) for w in _words(story)}
        hit = (
            0.0
            if copied(f.get("Summary", ""), story)
            else sum(weight[w] for w in want & have)
        )
        out["words"] = hit / sum(weight.values())
    if sentence := f.get("Random sentence"):
        out["sentence"] = float(sentence.lower() in story)
    if "Dialogue" in f.get("Features", ""):
        out["dialogue"] = float('"' in story)
    return out


def rarity(stories: list[str]) -> dict[str, float]:
    """stem -> idf, log((1 + N) / (1 + stories containing it)). Smoothed so a
    stem in every story weighs ~0, never less: lily 1.39, dog 2.49, stitches
    6.41 over 20k train stories."""
    df: dict[str, int] = {}
    for s in stories:
        for w in {stem(w) for w in _words(s)}:
            df[w] = df.get(w, 0) + 1
    n = len(stories)
    return {w: math.log((1 + n) / (1 + c)) for w, c in df.items()}


def iter_records(path: Path, chunk_bytes: int = 1 << 26) -> Iterator[dict[str, str]]:
    """Stream a source file -> parsed records. Reading 2.66 GB whole is ~20 GB
    of Python objects; here one 64 MB chunk is resident, and the trailing
    partial record is carried onto the next chunk instead of parsed as whole."""
    rest = ""
    with open(path, encoding="utf-8") as f:
        while chunk := f.read(chunk_bytes):
            parts = (rest + chunk).split(SEP)
            rest = parts.pop()
            for p in parts:
                if usable(rec := parse(p.strip())):
                    yield rec
    if usable(rec := parse(rest.strip())):
        yield rec


def source_path(split: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_DATASET, FILES[split], repo_type="dataset"))


class Instruct:
    """Records -> Packed, cached per (split, n). n_train caps the corpus: 200k
    records is ~50M tokens, already far more than a finetune sees."""

    name = "instruct"

    def __init__(
        self,
        tok=None,
        n_train: int = 200_000,
        n_val: int = 2_000,
        eval_pool: int = 500,
        idf: dict[str, float] | None = None,
        n_idf: int = 20_000,
    ):
        # the corpus the base was pretrained on picks the vocab, not GPT-2's
        self.tok = tokenizer_for() if tok is None else tok
        self.n_train, self.n_val, self.eval_pool = n_train, n_val, eval_pool
        self.eot = self.tok.specials[ENDOFTEXT]
        self._prompts: list[str] | None = None
        self._idf, self.n_idf = idf, n_idf

    @property
    def idf(self) -> dict[str, float]:
        """Rarity over the first n_idf train stories, cached: 2 s to build."""
        if self._idf is None:
            path = CACHE / f"idf_{self.n_idf}.json"
            if path.exists():
                self._idf = json.loads(path.read_text())
            else:
                recs = iter_records(source_path("train"))
                self._idf = rarity(
                    [r["Story"] for _, r in zip(range(self.n_idf), recs)]
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._idf))
        return self._idf

    def _n(self, split: str) -> int:
        return self.n_train if split == "train" else self.n_val

    def build(self, split: str) -> Packed:
        assert split in ("train", "val")
        path = CACHE / f"{split}_{self._n(split)}_{self.tok.vocab_size}.npz"
        if path.exists():
            z = np.load(path)
            return Packed(
                ids=z["ids"], mask=z["mask"], starts=z["starts"],
                vocab_size=self.tok.vocab_size, eot_id=self.eot,
            )  # fmt: skip

        recs = []
        for rec in iter_records(source_path(split)):
            recs.append(render(rec))
            if len(recs) >= self._n(split):
                break
        # batched, threaded: 400k separate encode calls is the slow way. prompt
        # and completion are encoded apart so the boundary is exact, not inferred
        prompts = self.tok.encode_batch([p for p, _ in recs])
        comps = self.tok.encode_batch([c for _, c in recs])

        ids: list[int] = []
        mask: list[bool] = []
        starts: list[int] = []
        for p, c in zip(prompts, comps):
            starts.append(len(ids))
            ids += p + c + [self.eot]
            mask += [False] * len(p) + [True] * (len(c) + 1)

        packed = Packed(
            ids=np.array(ids, np.uint16),
            mask=np.array(mask, bool),
            starts=np.array(starts, np.int64),
            vocab_size=self.tok.vocab_size,
            eot_id=self.eot,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=packed.ids, mask=packed.mask, starts=packed.starts)
        return packed

    def reward(self, prompt: str, completion: str) -> float:
        """Points earned out of the points this prompt offers: stop, sentence
        and dialogue 1 each, words 2. Scaled to [0, 1] so prompts compare.

        A story that never stops scores 0 whatever else it earned: otherwise
        running to the token budget and listing every word is the best policy.
        """
        c = checks(prompt, completion, self.idf)
        if not c["stop"]:
            return 0.0
        return sum(POINTS[k] * v for k, v in c.items()) / sum(POINTS[k] for k in c)

    def prompts(self) -> list[str]:
        """Held-out prompts that carry a `Words:` line, so the score is checkable."""
        if self._prompts is None:
            pool = []
            for rec in iter_records(source_path("val")):
                p, _ = render(rec)
                if required_words(p):
                    pool.append(p)
                if len(pool) >= self.eval_pool:
                    break
            self._prompts = pool
        return self._prompts

    def evaluate(
        self,
        model,
        n: int = 20,
        max_new_tokens: int = 256,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> dict[str, float]:
        """Stories for n held-out prompts, one generate call each -- prompts
        differ in length and generate() has no padding, so this is the expensive
        metric in the loop. ~0.8 s per sample on a 123.6M model.

        Greedy by default, which makes the number reproducible and is also the
        worst case for repetition: this base loops 100% of the time at
        temperature 0 and 4% at 1.0. Pass a temperature to see the gap.

        `words` now includes the Summary's words, so it does not compare with
        runs before 2026-09-22. 11.4% of gold stories run past 256 tokens and
        lose the stop point at the default budget.
        """
        device = next(model.parameters()).device
        rng = np.random.default_rng(seed)
        pool = self.prompts()
        picks = rng.choice(len(pool), size=min(n, len(pool)), replace=False)

        rewards, scored, lengths = [], [], []
        for i in picks:
            prompt = pool[int(i)]
            x = torch.tensor([self.tok.encode(prompt)], device=device)
            out = generate(
                model, x, max_new_tokens, temperature=temperature, top_p=top_p,
                use_cache=True, generator=torch.Generator(device=device).manual_seed(seed),
            )  # fmt: skip
            text = self.tok.decode(out[0, x.size(1) :].tolist())
            story = text.split(ENDOFTEXT)[0]
            rewards.append(self.reward(prompt, text))
            scored.append(checks(prompt, text, self.idf))
            lengths.append(len(story.split()))

        # each check is averaged over the prompts that have it, so at n=20
        # sentence and dialogue rest on ~6 prompts each
        def mean_of(k, fn=lambda v: v):
            vs = [fn(c[k]) for c in scored if k in c]
            return float(np.mean(vs)) if vs else float("nan")

        return {
            "words": mean_of("words"),
            "words_all": mean_of("words", lambda v: v == 1.0),
            "sentence": mean_of("sentence"),
            "dialogue": mean_of("dialogue"),
            "stop_rate": mean_of("stop"),
            "reward": float(np.mean(rewards)),
            "story_words": float(np.mean(lengths)),
        }


if __name__ == "__main__":
    # assertions are in test_instruct.py; this just prints the data
    task = Instruct(n_train=200, n_val=50, eval_pool=50)
    packed = task.build("val")
    tok = task.tok

    s, e = packed.starts[0], packed.starts[1]
    ids, m = packed.ids[s:e], packed.mask[s:e]
    # decode the two halves apart: a token index is not a character index
    print(f"{len(packed)} tokens, {len(packed.starts)} examples\n")
    print("--- prompt (out of the loss) ---")
    print(tok.decode(ids[~m].tolist()))
    print("--- completion (in the loss) ---")
    print(tok.decode(ids[m].tolist()).replace(ENDOFTEXT, "<eot>")[:400])

    lens = np.diff([*packed.starts, len(packed)])
    print(f"\nexample length: mean {lens.mean():.0f} tokens, max {lens.max()}")
    print(f"completion is {packed.mask.mean():.0%} of tokens")

    p = task.prompts()[0]
    print(f"\nrequired words: {required_words(p)}")
    stuffed = " ".join(required_words(p))
    print(f"all words, stopped:   {checks(p, stuffed + ENDOFTEXT)}")
    print(f"all words, no stop:   {task.reward(p, stuffed)}")
    print(f"none of them:         {task.reward(p, 'nothing here' + ENDOFTEXT)}")
