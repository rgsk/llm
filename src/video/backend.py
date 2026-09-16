"""Which implementation the model layers use: ours, or torch's.

VIDEO_BACKEND=ours (the default) exports what this package implements;
VIDEO_BACKEND=torch exports torch's. The leaves were written to match torch's
API exactly -- same constructor, keys, init and forward -- so the two are
interchangeable at import time and a checkpoint crosses the boundary either way.

The flag reaches the leaves and the Module base together; they have to come from
the same side of it (see module.py). Only speed changes, not the model.

Leaf tests name their own class (OurLinear, not Linear), so they check our code
whatever the flag says.
"""

import os

_BACKEND = os.getenv("VIDEO_BACKEND", "ours").lower()
assert _BACKEND in ("torch", "ours"), (
    f"VIDEO_BACKEND must be 'torch' or 'ours', got {_BACKEND!r}"
)

USE_TORCH = _BACKEND == "torch"


if __name__ == "__main__":
    # 1. unset means `ours`: the package behaves as it did before the switch
    assert not USE_TORCH or os.getenv("VIDEO_BACKEND") == "torch"
    assert isinstance(USE_TORCH, bool)

    # 2. a typo fails at import, not silently
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", "import backend"],
        env=os.environ | {"VIDEO_BACKEND": "pytorch"},
        capture_output=True,
        text=True,
        check=False,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert r.returncode != 0 and "must be 'torch' or 'ours'" in r.stderr

    print(f"VIDEO_BACKEND={_BACKEND}  USE_TORCH={USE_TORCH}")
    print("ok")
