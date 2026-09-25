"""Run untrusted Python or C++ on stdin and judge its stdout. The reward function for
the code rungs."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# no network: a fresh net namespace with only loopback, if the kernel allows it
NO_NET = (
    ["unshare", "-rn"]
    if shutil.which("unshare")
    and subprocess.run(
        ["unshare", "-rn", "true"], capture_output=True, check=False
    ).returncode
    == 0
    else []
)


@dataclass(frozen=True)
class Run:
    status: str  # ok | error | timeout | compile_error
    stdout: str
    stderr: str


def build(code: str, lang: str, d: str) -> list[str] | str:
    """Write the source into d; the command to run it, or a compiler error."""
    if lang == "python":
        (Path(d) / "main.py").write_text(code)
        return [sys.executable, "-I", "main.py"]
    assert lang == "cpp", lang
    (Path(d) / "main.cpp").write_text(code)
    c = subprocess.run(
        ["g++", "-O2", "-std=c++17", "-o", "main", "main.cpp"],
        cwd=d,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return ["./main"] if c.returncode == 0 else c.stderr


def execute(cmd: list[str], d: str, stdin: str, timeout: float, mem_mb: int) -> Run:
    # prlimit, not preexec_fn: a preexec_fn forces a full fork of this process,
    # 85 ms from a 4 GB trainer vs 0.9 ms
    lim = ["prlimit", f"--as={mem_mb * 2**20}", f"--fsize={2**20}", "--"]
    try:
        p = subprocess.run(
            [*lim, *NO_NET, *cmd],
            input=stdin,
            capture_output=True,
            text=True,
            errors="replace",  # a program may print any bytes
            timeout=timeout,
            cwd=d,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Run("timeout", "", "")
    return Run("ok" if p.returncode == 0 else "error", p.stdout, p.stderr)


def run(
    code: str, stdin: str, timeout: float = 2.0, mem_mb: int = 512, lang="python"
) -> Run:
    with tempfile.TemporaryDirectory() as d:
        cmd = build(code, lang, d)
        if isinstance(cmd, str):
            return Run("compile_error", "", cmd)
        return execute(cmd, d, stdin, timeout, mem_mb)


def same(got: str, want: str) -> bool:
    """Token-wise, like Codeforces' default checker: whitespace is free."""
    return got.split() == want.split()


@dataclass(frozen=True)
class Verdict:
    passed: bool
    status: str  # accepted | wrong | error | timeout | compile_error
    tests_run: int


def judge(
    code: str,
    tests: list[tuple[str, str]],
    timeout: float = 2.0,
    mem_mb: int = 512,
    lang="python",
) -> Verdict:
    """Compile once, then stop at the first failing test, as a judge does."""
    with tempfile.TemporaryDirectory() as d:
        cmd = build(code, lang, d)
        if isinstance(cmd, str):
            return Verdict(False, "compile_error", 0)
        for i, (inp, out) in enumerate(tests, 1):
            r = execute(cmd, d, inp, timeout, mem_mb)
            if r.status != "ok":
                return Verdict(False, r.status, i)
            if not same(r.stdout, out):
                return Verdict(False, "wrong", i)
    return Verdict(True, "accepted", len(tests))


if __name__ == "__main__":
    add = "a, b = map(int, input().split())\nprint(a + b)\n"
    print(run(add, "2 3\n"))
    print(judge(add, [("2 3\n", "5\n"), ("10 -4\n", "6")]))
    print(judge("while True: pass", [("", "")], timeout=0.5))
    print("net isolation:", "on" if NO_NET else "OFF")
