"""Every module's `if __name__ == "__main__"` block must still run.

The older modules here keep their assertions in that block rather than in a
test file. Rather than port ~40 of them -- they narrate, and reading one top to
bottom is the point -- this sweeps them: each runs in a subprocess, and a
non-zero exit is a failing test.

Subprocess, not runpy: these blocks build models, mutate global RNG and in one
case call os._exit, none of which belongs in the collecting process. The cost
is ~0.85s of `import torch` each, which is why the whole file is `slow`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
# basic.py's __main__ is the training entry point (run_training = True), not a
# self-test -- sweeping it would kick off a real run, and it hardcodes a config
# whose vocab only matches VIDEO_DATASET=fineweb_edu
ENTRY_POINTS = {"basic.py"}
MODULES = sorted(
    p.name
    for p in HERE.glob("*.py")
    if not p.name.startswith("test_")
    and p.name != "conftest.py"
    and p.name not in ENTRY_POINTS
)


@pytest.mark.slow
@pytest.mark.parametrize("module", MODULES)
def test_main_block_runs(module):
    r = subprocess.run(
        [sys.executable, module],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # a non-zero exit is the assertion below, not an exception
    )
    assert r.returncode == 0, f"{module} exited {r.returncode}\n{r.stderr[-2000:]}"
