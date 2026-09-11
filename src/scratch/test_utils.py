def selftest(fn):
    """@selftest(fn) over a test block: attaches it as fn.test, with fn passed in."""

    def attach(t):
        assert not hasattr(fn, "test"), f"{fn.__qualname__}: test already attached"
        t.__defaults__ = (fn,)
        t.__qualname__ = f"{fn.__qualname__}.test"
        fn.test = t
        return t

    return attach


def run_tests(*objs):
    """Call .test() on every attached test under objs (functions, classes, modules)."""
    found = []

    def walk(o):
        if callable(o) and not isinstance(o, type):
            t = getattr(o, "test", None)
            if t is not None and t not in found:
                found.append(t)
            return
        for it in vars(o).values():
            if isinstance(it, property):
                walk(it.fget)
            elif isinstance(it, type) or callable(it):
                walk(it)

    for o in objs:
        walk(o)
    for t in found:
        t()
    return [t.__qualname__ for t in found]


def check_raises(exc, fn, msg=None):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(msg or f"expected {exc.__name__}, nothing raised")
