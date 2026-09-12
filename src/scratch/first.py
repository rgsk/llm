from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any, overload

from test_utils import check_raises, selftest

type Scalar = int | float
type Nested = Scalar | list[Nested]
type Shape = tuple[int, ...]
type Strides = tuple[int, ...]
type Index = tuple[int, ...]


def contiguous_strides(shape: Shape) -> Strides:
    res = []
    p = 1
    for s in reversed(shape):
        res.append(p)
        p *= s
    return tuple(reversed(res))


@selftest(contiguous_strides)
def _(fn):
    assert fn((2, 3)) == (3, 1)
    assert fn((2, 3, 4)) == (12, 4, 1)


def infer_shape(nested: Nested) -> Shape:
    shape = []
    cur: Nested = nested
    while isinstance(cur, list):
        shape.append(len(cur))
        if len(cur) == 0:
            break
        cur = cur[0]
    return tuple(shape)


@selftest(infer_shape)
def _(fn):
    assert fn(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ) == (2, 3)


def flatten(nested: Nested) -> list[Scalar]:
    if not isinstance(nested, list):
        return [nested]
    out = []
    for e in nested:
        out += flatten(e)
    return out


@selftest(flatten)
def _(fn):
    assert fn(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ) == [1, 2, 3, 4, 5, 6]


def prod(shape: Iterable[int]) -> int:
    out = 1
    for v in shape:
        out *= v
    return out


@selftest(prod)
def _(fn):
    assert fn((2, 3)) == 6
    assert fn((2, 3, 4)) == 24


def broadcast_shape(s1: Shape, s2: Shape) -> Shape:
    n = max(len(s1), len(s2))
    s1 = (1,) * (n - len(s1)) + tuple(s1)  # pad on the LEFT
    s2 = (1,) * (n - len(s2)) + tuple(s2)
    out = []
    for a, b in zip(s1, s2):
        if a == b or a == 1 or b == 1:
            out.append(max(a, b))
        else:
            raise ValueError(f"cannot broadcast {s1} with {s2}")
    return tuple(out)


@selftest(broadcast_shape)
def _(fn):
    assert fn((128, 30), (30,)) == (128, 30)  # right-aligned, missing dims are 1
    assert fn((30,), (128, 30)) == (128, 30)  # either side can be the shorter one
    assert fn((2, 1, 4), (3, 4)) == (2, 3, 4)
    assert fn((5,), ()) == (5,)
    assert fn((2, 3), (2, 3)) == (2, 3)

    e = check_raises(ValueError, lambda: fn((2, 3), (4, 3)))
    assert "cannot broadcast" in str(e)


class Tensor:
    def __init__(
        self,
        data: list,
        shape: Shape | None = None,
        strides: Strides | None = None,
    ) -> None:
        if shape is None:
            shape = infer_shape(data)
            data = flatten(data)
        self.data = data
        self.shape = shape
        self.strides = contiguous_strides(shape) if strides is None else strides

    @selftest(__init__)
    def _(fn):
        a = Tensor.__new__(Tensor)  # bare instance; fn is the unbound __init__
        fn(a, [1, 2, 3, 4, 5, 6], (2, 3))
        assert (a.data, a.shape, a.strides) == ([1, 2, 3, 4, 5, 6], (2, 3), (3, 1))

        b = Tensor.__new__(Tensor)
        fn(b, [[1, 2, 3], [4, 5, 6]])  # shape inferred, data flattened
        assert (b.data, b.shape, b.strides) == ([1, 2, 3, 4, 5, 6], (2, 3), (3, 1))

        c = Tensor.__new__(Tensor)
        fn(c, [1, 2, 3, 4, 5, 6], (2, 3), (1, 2))  # given strides kept as-is
        assert c.strides == (1, 2)

    def _offset(self, idx: Index) -> int:
        assert len(idx) == len(self.shape), (
            f"got {len(idx)} indices for shape {self.shape}"
        )
        return sum(i * s for i, s in zip(idx, self.strides))

    @selftest(_offset)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a, (1, 2)) == 5
        assert a.data[fn(a, (1, 2))] == 6

        b = Tensor(list(range(24)), (2, 3, 4))
        assert fn(b, (1, 2, 3)) == 23

        e = check_raises(
            AssertionError,
            lambda: fn(b, (1, 2)),
            "_offset accepted 2 indices for a 3-D shape",
        )
        assert "shape" in str(e)

    def transpose(self, d0: int, d1: int) -> Tensor:
        shape, strides = list(self.shape), list(self.strides)
        shape[d0], shape[d1] = shape[d1], shape[d0]
        strides[d0], strides[d1] = strides[d1], strides[d0]
        return Tensor(self.data, tuple(shape), tuple(strides))

    @selftest(transpose)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        t = fn(a, 0, 1)
        assert t.shape == (3, 2)
        assert t.strides == (1, 3)
        assert t.data is a.data  # a view, not a copy
        assert t.tolist() == [[1, 4], [2, 5], [3, 6]]

    def tolist(self) -> Nested:
        def build(idx):
            if len(idx) == len(self.shape):
                return self.data[self._offset(idx)]
            d = len(idx)
            return [build(idx + (i,)) for i in range(self.shape[d])]

        return build(())

    @selftest(tolist)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a) == [[1, 2, 3], [4, 5, 6]]
        assert fn(a.transpose(0, 1)) == [[1, 4], [2, 5], [3, 6]]

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @selftest(numel.fget)  # type: ignore[attr-defined]
    def _(fn):
        assert fn(Tensor([1, 2, 3, 4, 5, 6], (2, 3))) == 6

    def is_contiguous(self) -> bool:
        return self.strides == contiguous_strides(self.shape)

    @selftest(is_contiguous)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a)
        assert not fn(a.transpose(0, 1))

    def flat(self) -> list[Scalar]:
        return flatten(self.tolist())

    @selftest(flat)
    def _(fn):
        t = Tensor([1, 2, 3, 4, 5, 6], (2, 3)).transpose(0, 1)
        assert t.data == [1, 2, 3, 4, 5, 6]  # storage order
        assert fn(t) == [1, 4, 2, 5, 3, 6]  # logical order

    def contiguous(self) -> Tensor:
        if self.is_contiguous():
            return self
        return Tensor(self.flat(), self.shape)

    @selftest(contiguous)
    def _(fn):
        t = Tensor([1, 2, 3, 4, 5, 6], (2, 3)).transpose(0, 1)
        tc = fn(t)
        assert tc.data == [1, 4, 2, 5, 3, 6]
        assert tc.shape == t.shape == (3, 2)
        assert tc.strides == (2, 1)
        assert tc.is_contiguous()
        assert (
            tc.tolist() == t.tolist() == [[1, 4], [2, 5], [3, 6]]
        )  # same values, different storage
        assert tc.data is not t.data
        assert fn(tc) is tc  # already contiguous: no copy

    @overload
    def reshape(self, *shape: int) -> Tensor: ...
    @overload
    def reshape(self, shape: Shape | list[int], /) -> Tensor: ...

    def reshape(self, *shape: Any) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            known = prod([s for s in shape if s != -1])
            shape = tuple(self.numel // known if s == -1 else s for s in shape)
        assert prod(shape) == self.numel, f"cannot reshape {self.shape} -> {shape}"
        src = self.contiguous()
        return Tensor(src.data, shape)

    @selftest(reshape)
    def _(fn):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        r = fn(a, 3, 2)
        assert r.tolist() == [[1, 2], [3, 4], [5, 6]]
        assert (r.data, r.shape, r.strides) == ([1, 2, 3, 4, 5, 6], (3, 2), (2, 1))
        assert r.data is a.data  # already contiguous: no copy

        assert fn(a, (3, 2)).shape == (3, 2)  # shape given as a tuple
        assert fn(a, -1, 2).shape == (3, 2)  # inferred dim
        assert fn(a, 3, -1).shape == (3, 2)
        assert fn(a, -1).shape == (6,)

        e = check_raises(AssertionError, lambda: fn(a, 1, 2))
        assert "cannot reshape" in str(e)

        t = a.transpose(0, 1)
        tr = fn(t, 6)
        assert tr.data is not t.data  # non-contiguous: copied in logical order
        assert tr.data == [1, 4, 2, 5, 3, 6]
        assert fn(t, 2, 3).tolist() == [[1, 4, 2], [5, 3, 6]]

    def indices(self) -> Iterator[Index]:
        """every logical index tuple, row-major order"""
        if len(self.shape) == 0:
            yield ()
            return
        idx = [0] * len(self.shape)
        for _ in range(self.numel):
            yield tuple(idx)
            for d in reversed(range(len(self.shape))):  # odometer: roll from the right
                idx[d] += 1
                if idx[d] < self.shape[d]:
                    break
                idx[d] = 0

    @selftest(indices)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert list(fn(a)) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        # logical order follows the view, not storage
        assert list(fn(a.transpose(0, 1))) == [
            (0, 0), (0, 1),
            (1, 0), (1, 1),
            (2, 0), (2, 1),
        ]  # fmt: skip
        assert list(fn(Tensor([1, 2, 3]))) == [(0,), (1,), (2,)]
        assert list(fn(Tensor(5))) == [()]  # 0-d: one empty index

    def expand(self, shape: Shape) -> Tensor:
        n = len(shape)
        pad = n - len(self.shape)
        assert pad >= 0
        old_shape = (1,) * pad + tuple(self.shape)
        old_strides = (0,) * pad + tuple(self.strides)

        strides = []
        for old, new, st in zip(old_shape, shape, old_strides):
            if old == new:
                strides.append(st)
            elif old == 1:
                strides.append(0)  # <- stay put
            else:
                raise ValueError(f"cannot expand {self.shape} -> {shape}")
        return Tensor(self.data, tuple(shape), tuple(strides))

    @selftest(expand)
    def _(fn):
        c = Tensor([[1], [2]])  # (2, 1)
        e = fn(c, (2, 3))
        assert (e.shape, e.strides) == ((2, 3), (1, 0))  # stride 0 = stay put
        assert e.tolist() == [[1, 1, 1], [2, 2, 2]]
        assert e.data is c.data  # a view, no copy

        r = Tensor([10, 20, 30])  # (3,) padded to (1, 3)
        assert fn(r, (2, 3)).strides == (0, 1)
        assert fn(r, (2, 3)).tolist() == [[10, 20, 30], [10, 20, 30]]

        err = check_raises(ValueError, lambda: fn(Tensor([[1, 2], [3, 4]]), (2, 3)))
        assert "cannot expand" in str(err)

    def _binop(
        self, other: Tensor | Scalar, f: Callable[[Scalar, Scalar], Scalar]
    ) -> Tensor:
        if not isinstance(other, Tensor):
            # scalar -> shape ()
            other = Tensor(other)  # type: ignore[arg-type]

        shape = broadcast_shape(self.shape, other.shape)
        a, b = self.expand(shape), other.expand(shape)
        data = [f(a.data[a._offset(i)], b.data[b._offset(i)]) for i in a.indices()]
        return Tensor(data, shape)

    @selftest(_binop)
    def _(fn):
        def add(x, y):
            return x + y

        a = Tensor([[1, 2, 3], [4, 5, 6]])

        out = fn(a, Tensor([10, 20, 30]), add)  # (2,3) with (3,): right-aligned
        assert (out.shape, out.tolist()) == ((2, 3), [[11, 22, 33], [14, 25, 36]])
        assert fn(a, 10, add).tolist() == [[11, 12, 13], [14, 15, 16]]  # scalar operand

        out = fn(a, a, lambda x, y: x * y)
        assert out.is_contiguous()  # result is always fresh and contiguous
        assert out.data is not a.data

        e = check_raises(ValueError, lambda: fn(a, Tensor([1, 2]), add))
        assert "cannot broadcast" in str(e)

    def __add__(self, other: Tensor | Scalar) -> Tensor:
        return self._binop(other, lambda x, y: x + y)

    @selftest(__add__)
    def _(fn):
        a = Tensor([[1, 2], [3, 4]])
        assert fn(a, a).tolist() == [[2, 4], [6, 8]]
        assert fn(a, 10).tolist() == [[11, 12], [13, 14]]
        assert (2 + a).tolist() == fn(a, 2).tolist()  # __radd__ is this same function

    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        return self._binop(other, lambda x, y: x * y)

    @selftest(__mul__)
    def _(fn):
        a = Tensor([[1, 2], [3, 4]])
        assert fn(a, a).tolist() == [[1, 4], [9, 16]]
        assert fn(a, Tensor([10, 100])).tolist() == [[10, 200], [30, 400]]  # row bcast
        assert (3 * a).tolist() == fn(a, 3).tolist()  # __rmul__

    def __sub__(self, other: Tensor | Scalar) -> Tensor:
        return self._binop(other, lambda x, y: x - y)

    @selftest(__sub__)
    def _(fn):
        a = Tensor([[5, 6], [7, 8]])
        assert fn(a, Tensor([[1, 2], [3, 4]])).tolist() == [[4, 4], [4, 4]]
        assert fn(a, 1).tolist() == [[4, 5], [6, 7]]

    __radd__ = __add__  # 2 + t  ->  t + 2
    __rmul__ = __mul__

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, strides={self.strides}, data={self.tolist()})"
        )

    @selftest(__repr__)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a) == (
            "Tensor(shape=(2, 3), strides=(3, 1), data=[[1, 2, 3], [4, 5, 6]])"
        )
        assert str(a) == repr(a) == fn(a)  # str falls back to __repr__


if __name__ == "__main__":
    from tests import tests

    tests()
