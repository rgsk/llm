"""Codeforces problems from deepmind/code_contests, judged in the sandbox."""

from __future__ import annotations

import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pyarrow.parquet as pq
import torch
from datasets import load_dataset

from sandbox import Verdict, judge

REPO = "hf://datasets/deepmind/code_contests"
LANG_IDS = {"cpp": 2, "python": 3}  # code_contests language enum


@dataclass(frozen=True)
class Problem:
    name: str
    rating: int
    statement: str
    tests: list[tuple[str, str]]  # public, then private, then generated
    time_limit: float
    solutions: dict[str, list[str]]  # accepted human code, by lang


def to_problem(r: dict) -> Problem:
    tests = [
        (i, o)
        for k in ("public_tests", "private_tests", "generated_tests")
        for i, o in zip(r[k]["input"], r[k]["output"])
    ]
    sols = r["solutions"]
    by_lang = {
        name: [s for s, i in zip(sols["solution"], sols["language"]) if i == k]
        for name, k in LANG_IDS.items()
    }
    tl = r["time_limit"]
    return Problem(
        name=r["name"],
        rating=r["cf_rating"],
        statement=r["description"],
        tests=tests,
        time_limit=tl["seconds"] + tl["nanos"] / 1e9 if tl else 2.0,
        solutions=by_lang,
    )


def load(rating: int = 800, splits=("valid", "test")) -> list[Problem]:
    """valid + test are two small parquet files; train is 39 shards, see load_train."""
    out = []
    for split in splits:
        # the plain parquet builder: the repo's own builder checks disk for all 25 GB
        ds = load_dataset(
            "parquet",
            data_files={split: f"{REPO}/data/{split}-*.parquet"},
            split=split,
        )
        out += [
            to_problem(r) for r in ds if r["source"] == 2 and r["cf_rating"] == rating
        ]
    return out


def load_train(max_rating: int = 1000) -> list[Problem]:
    """Codeforces problems rated <= max_rating from the 39 train shards (hub cache)."""
    hub = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pat = (
        f"{hub}/hub/datasets--deepmind--code_contests/snapshots/*/data/train-*.parquet"
    )
    shards = sorted(glob.glob(pat))
    assert len(shards) == 39, f"expected 39 train shards under {pat}, got {len(shards)}"
    out = []
    for f in shards:
        t = pq.read_table(
            f, filters=[("source", "=", 2), ("cf_rating", "<=", max_rating)]
        )
        out += [to_problem(r) for r in t.to_pylist() if r["cf_rating"] > 0]
    return out


FENCES = {"python": {"python", "py", "python3", ""}, "cpp": {"cpp", "c++", ""}}


def extract_code(text: str, lang="python") -> str | None:
    """The last fenced block tagged lang, or untagged."""
    blocks = re.findall(r"```([\w+]*)[ \t]*\n(.*?)```", text, re.DOTALL)
    ok = [body for tag, body in blocks if tag.lower() in FENCES[lang]]
    return ok[-1] if ok else None


def check(p: Problem, code: str, lang="python", slack: float = 2.0) -> Verdict:
    """Our CPU is not the judge's, so the time limit gets slack."""
    return judge(code, p.tests, timeout=p.time_limit * slack, lang=lang)


SYSTEM = "You are an expert competitive programmer."
ASK = {
    "python": "Solve the following problem in Python 3. Read from standard input "
    "and write to standard output. Put the complete program in one ```python "
    "code block.\n\n",
    "cpp": "Solve the following problem in C++17. Read from standard input and "
    "write to standard output. Put the complete program in one ```cpp code "
    "block.\n\n",
}


def prompt(tok, statement: str, lang="python") -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ASK[lang] + statement},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(
    model,
    tok,
    problems: list[Problem],
    lang="python",
    max_new=1024,
    batch=8,
    temperature=0.0,
) -> list[str]:
    """One completion per problem. Left padding so every row ends at the prompt."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(problems), batch):
        texts = [prompt(tok, p.statement, lang) for p in problems[i : i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        sample = (
            {"do_sample": True, "temperature": temperature, "top_p": 0.95}
            if temperature
            else {"do_sample": False}
        )
        ids = model.generate(
            **enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **sample
        )
        out += tok.batch_decode(
            ids[:, enc.input_ids.shape[1] :], skip_special_tokens=True
        )
    return out


def judge_all(problems, outs, lang, workers=32) -> list[dict]:
    """Grade completion j against problems[j]; subprocess waits release the GIL."""

    def one(p, o):
        code = extract_code(o, lang)
        v = check(p, code, lang) if code else None
        return {"name": p.name, "status": v.status if v else "no_code", "out": o}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, problems, outs))


def pass_at_k(model, tok, problems, lang, k=8, temperature=0.8, batch=16):
    """k samples per problem; returns per-problem pass counts and every row."""
    outs = generate(
        model,
        tok,
        [p for p in problems for _ in range(k)],
        lang,
        batch=batch,
        temperature=temperature,
    )
    rows = judge_all([problems[j // k] for j in range(len(outs))], outs, lang)
    passes = [
        sum(r["status"] == "accepted" for r in rows[i * k : (i + 1) * k])
        for i in range(len(problems))
    ]
    return passes, rows


if __name__ == "__main__":
    ps = load(800)
    print(len(ps), "problems at 800")
    p = ps[0]
    n = {k: len(v) for k, v in p.solutions.items()}
    print(p.name, "|", len(p.tests), "tests |", n, "human solutions")
    print(p.statement[:300], "...")
    for lang in ("python", "cpp"):
        print(lang, check(p, p.solutions[lang][1], lang))
