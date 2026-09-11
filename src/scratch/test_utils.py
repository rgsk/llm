from types import FunctionType


def selftest(fn):
    """Expose fn's nested `test` block as fn.test, bound with self=fn."""
    tests = [c for c in fn.__code__.co_consts if getattr(c, "co_name", None) == "test"]
    assert len(tests) == 1, (
        f"{fn.__name__}: expected 1 nested test def, got {len(tests)}"
    )
    fn.test = FunctionType(tests[0], fn.__globals__, f"{fn.__name__}.test", (fn,))
    return fn


def check_raises(exc, fn, msg=None):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(msg or f"expected {exc.__name__}, nothing raised")
