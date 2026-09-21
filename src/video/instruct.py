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
order-independence. `Words` appears in 66.6% of records and is the checkable
part: the story either contains those words or it does not, which is what makes
this a DPO/GRPO target later.

2.5M records at 2.66 GB, so build() streams and caches. The first call costs
minutes, the rest load a .npz.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from generate import generate
from paths import DATA_ROOT
from task import Packed
from tokenizer import ENDOFTEXT, load_tokenizer

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
    ):
        self.tok = load_tokenizer() if tok is None else tok
        self.n_train, self.n_val, self.eval_pool = n_train, n_val, eval_pool
        self.eot = self.tok.specials[ENDOFTEXT]
        self._prompts: list[str] | None = None

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
        """Fraction of the prompt's required words that made it into the story.

        A prompt with no `Words:` line is not checkable and scores 0.0 -- inside
        a GRPO group that is a constant, so those prompts simply do not train.
        """
        words = required_words(prompt)
        if not words:
            return 0.0
        story = completion.split(ENDOFTEXT)[0].lower()
        return sum(w in story for w in words) / len(words)

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
        """
        device = next(model.parameters()).device
        rng = np.random.default_rng(seed)
        pool = self.prompts()
        picks = rng.choice(len(pool), size=min(n, len(pool)), replace=False)

        rewards, stops, lengths = [], [], []
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
            stops.append(ENDOFTEXT in text)
            lengths.append(len(story.split()))

        return {
            "words": float(np.mean(rewards)),
            "words_all": float(np.mean([r == 1.0 for r in rewards])),
            "stop_rate": float(np.mean(stops)),
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
    print(
        f"reward, story with all of them:  {task.reward(p, ' '.join(required_words(p)))}"
    )
    print(f"reward, story with none of them: {task.reward(p, 'nothing here')}")
