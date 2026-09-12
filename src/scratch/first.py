from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any, overload

from test_utils import check_raises, selftest

type Scalar = int | float
type Nested = Scalar | list[Nested]
type Shape = tuple[int, ...]
type Strides = tuple[int, ...]
type Index = tuple[int, ...]
type Grad = list[Scalar]  # flat, logical order
type BinFn = Callable[[Scalar, Scalar], Scalar]


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


def iter_indices(shape: Shape) -> Iterator[Index]:
    if len(shape) == 0:
        yield ()
        return
    idx = [0] * len(shape)
    for _ in range(prod(shape)):
        yield tuple(idx)
        for d in reversed(range(len(shape))):
            idx[d] += 1
            if idx[d] < shape[d]:
                break
            idx[d] = 0


@selftest(iter_indices)
def _(fn):
    assert list(fn((2, 3))) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert list(fn((3,))) == [(0,), (1,), (2,)]
    assert list(fn(())) == [()]  # 0-d: one empty index
    assert list(fn((2, 2, 2)))[:3] == [(0, 0, 0), (0, 0, 1), (0, 1, 0)]  # rolls right
    assert len(list(fn((2, 3, 4)))) == 24


def topo(root: Tensor) -> list[Tensor]:
    order: list[Tensor] = []
    seen: set[int] = set()

    def visit(t: Tensor) -> None:
        if id(t) in seen:
            return
        seen.add(id(t))
        for p in t._parents:
            visit(p)
        order.append(t)  # appended only after all parents are in

    visit(root)
    return order  # inputs first, root last


@selftest(topo)
def _(fn):
    x, w = Tensor([[1.0, 2.0]]), Tensor([[3.0], [4.0]])
    y = x @ w
    z = y * y

    order = fn(z)
    assert order[-1] is z  # root last
    assert order.index(x) < order.index(y)  # parents before children
    assert len(order) == 4  # x, w, matmul, mul: y feeds mul twice, listed once
    assert fn(x) == [x]  # a leaf is its own whole graph


def unbroadcast(g: Grad, out_shape: Shape, shape: Shape) -> Grad:
    """fold a gradient of shape out_shape back down to shape"""
    if tuple(out_shape) == tuple(shape):
        return g
    pad = len(out_shape) - len(shape)
    padded = (1,) * pad + tuple(shape)  # align at the right, as in broadcast_shape
    res = [0.0] * prod(shape)
    strides = contiguous_strides(shape)
    for gi, idx in enumerate(iter_indices(out_shape)):
        # wherever the input had a 1, every output position maps back to index 0
        tgt = tuple(0 if padded[d] == 1 else idx[d] for d in range(len(out_shape)))
        res[sum(i * s for i, s in zip(tgt[pad:], strides))] += g[gi]
    return res


@selftest(unbroadcast)
def _(fn):
    g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # a (2,3) gradient
    assert fn(g, (2, 3), (2, 3)) is g  # same shape: handed straight back
    assert fn(g, (2, 3), (3,)) == [5.0, 7.0, 9.0]  # a row operand: columns summed
    assert fn(g, (2, 3), (2, 1)) == [6.0, 15.0]  # a column operand: rows summed
    assert fn(g, (2, 3), ()) == [21.0]  # a scalar: everything summed
    assert fn([1.0, 2.0], (2,), (1,)) == [3.0]
    assert fn([1.0] * 24, (2, 3, 4), (3, 1)) == [8.0, 8.0, 8.0]  # 2*4 per row


class Tensor:
    def __init__(
        self,
        data: list,
        shape: Shape | None = None,
        strides: Strides | None = None,
        _parents: tuple[Tensor, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        if shape is None:
            shape = infer_shape(data)
            data = flatten(data)
        self.data = data
        self.shape = shape
        self.strides = contiguous_strides(shape) if strides is None else strides
        self.grad: Grad | None = None
        self._parents = tuple(_parents)
        self._op = _op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

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
        assert (c.label, c._op, c._parents, c.grad) == ("", "", (), None)

    def _accum(self, g: Grad) -> None:
        """add g (flat, logical order) into self.grad"""
        if self.grad is None:
            self.grad = [0.0] * self.numel
        for i in range(self.numel):
            self.grad[i] += g[i]

    @selftest(_accum)
    def _(fn):
        a = Tensor([1.0, 2.0, 3.0])
        assert a.grad is None  # nothing until something flows back
        fn(a, [1.0, 1.0, 1.0])
        assert a.grad == [1.0, 1.0, 1.0]
        fn(a, [0.5, 0.5, 0.5])
        assert a.grad == [1.5, 1.5, 1.5]  # accumulates, never replaces

    def backward(self) -> None:
        self.grad = [1.0] * self.numel
        for t in reversed(topo(self)):
            t._backward()

    @selftest(backward)
    def _(fn):
        x = Tensor([2.0, 3.0])
        y = x * x
        fn(y)
        assert y.grad == [1.0, 1.0]  # the root is seeded with ones
        assert x.grad == [4.0, 6.0]  # 2x: both edges of the same node accumulate

        z = Tensor([5.0])
        fn(z)  # a leaf: nothing to walk, just the seed
        assert z.grad == [1.0]

        a = Tensor([2.0])
        c = a * a * a  # two levels deep: only a reverse walk propagates through
        fn(c)
        assert a.grad == [12.0]  # 3a^2

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
        return Tensor(
            self.data, tuple(shape), tuple(strides), _parents=(self,), _op="transpose"
        )

    @selftest(transpose)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        t = fn(a, 0, 1)
        assert t.shape == (3, 2)
        assert t.strides == (1, 3)
        assert t.data is a.data  # a view, not a copy
        assert t.tolist() == [[1, 4], [2, 5], [3, 6]]
        assert (t._op, t._parents) == ("transpose", (a,))  # stays in the graph

    def tolist(self) -> Nested:
        def build(idx: Index) -> Nested:
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
        return Tensor(self.flat(), self.shape, _parents=(self,), _op="contiguous")

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
        assert (tc._op, tc._parents) == ("contiguous", (t,))
        assert fn(tc) is tc  # already contiguous: same node, nothing added

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
        return Tensor(src.data, shape, _parents=(src,), _op="reshape")

    @selftest(reshape)
    def _(fn):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        r = fn(a, 3, 2)
        assert r.tolist() == [[1, 2], [3, 4], [5, 6]]
        assert (r.data, r.shape, r.strides) == ([1, 2, 3, 4, 5, 6], (3, 2), (2, 1))
        assert r.data is a.data  # already contiguous: no copy
        assert (r._op, r._parents) == ("reshape", (a,))

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
        assert tr._parents[0]._op == "contiguous"  # the copy is its own node
        assert fn(t, 2, 3).tolist() == [[1, 4, 2], [5, 3, 6]]

    def indices(self) -> Iterator[Index]:
        return iter_indices(self.shape)

    @selftest(indices)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert list(fn(a)) == list(iter_indices((2, 3)))
        # a view walks its own logical shape, not the storage it borrows
        assert list(fn(a.transpose(0, 1))) == list(iter_indices((3, 2)))
        assert list(fn(Tensor(5))) == [()]  # 0-d

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
        return Tensor(
            self.data, tuple(shape), tuple(strides), _parents=(self,), _op="expand"
        )

    @selftest(expand)
    def _(fn):
        c = Tensor([[1], [2]])  # (2, 1)
        e = fn(c, (2, 3))
        assert (e.shape, e.strides) == ((2, 3), (1, 0))  # stride 0 = stay put
        assert e.tolist() == [[1, 1, 1], [2, 2, 2]]
        assert e.data is c.data  # a view, no copy
        assert (e._op, e._parents) == ("expand", (c,))  # stays in the graph

        r = Tensor([10, 20, 30])  # (3,) padded to (1, 3)
        assert fn(r, (2, 3)).strides == (0, 1)
        assert fn(r, (2, 3)).tolist() == [[10, 20, 30], [10, 20, 30]]

        err = check_raises(ValueError, lambda: fn(Tensor([[1, 2], [3, 4]]), (2, 3)))
        assert "cannot expand" in str(err)

    def _binop(
        self, other: Tensor | Scalar, f: BinFn, da: BinFn, db: BinFn, op: str
    ) -> Tensor:
        if not isinstance(other, Tensor):
            # scalar -> shape ()
            other = Tensor(other)  # type: ignore[arg-type]
        shape = broadcast_shape(self.shape, other.shape)
        a, b = self.expand(shape), other.expand(shape)
        av = [a.data[a._offset(i)] for i in a.indices()]  # broadcast values
        bv = [b.data[b._offset(i)] for i in b.indices()]
        out = Tensor(
            [f(x, y) for x, y in zip(av, bv)], shape, _parents=(self, other), _op=op
        )

        def _backward() -> None:
            assert out.grad is not None  # backward seeds it before calling us
            ga = [g * da(x, y) for g, x, y in zip(out.grad, av, bv)]
            gb = [g * db(x, y) for g, x, y in zip(out.grad, av, bv)]
            self._accum(unbroadcast(ga, shape, self.shape))
            other._accum(unbroadcast(gb, shape, other.shape))

        out._backward = _backward
        return out

    @selftest(_binop)
    def _(fn):
        def add(x, y):
            return x + y

        def one(x, y):
            return 1.0

        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = Tensor([10.0, 20.0, 30.0])
        out = fn(a, r, add, one, one, "add")  # (2,3) with (3,): right-aligned
        assert (out.shape, out.tolist()) == ((2, 3), [[11, 22, 33], [14, 25, 36]])
        assert (out._op, out._parents) == ("add", (a, r))  # operands, unexpanded

        out.grad = [1.0] * 6
        out._backward()  # da/db are both 1, so the seed passes straight through
        assert a.grad == [1.0] * 6
        assert r.grad == [2.0, 2.0, 2.0]  # broadcast row: folded back over 2 rows

        flipped = fn(r, a, add, one, one, "add")  # now `self` is the (3,) operand
        flipped.grad = [1.0] * 6
        r.grad = None
        flipped._backward()
        assert r.grad == [2.0, 2.0, 2.0]  # folded on the self side as well

        scalar_out = fn(a, 10, add, one, one, "add")  # scalar operand
        assert scalar_out.tolist() == [[11, 12, 13], [14, 15, 16]]
        assert scalar_out._parents[1].shape == ()  # the scalar became a 0-d parent

        p, q = Tensor([2.0, 3.0]), Tensor([5.0, 7.0])
        out = fn(p, q, lambda x, y: x * y, lambda x, y: y, lambda x, y: x, "mul")
        out.grad = [10.0, 100.0]  # a non-unit seed: the chain rule must use it
        out._backward()
        assert p.grad == [50.0, 700.0]  # g * q
        assert q.grad == [20.0, 300.0]  # g * p

        m = Tensor([[1.0, 2.0], [3.0, 4.0]])
        out = fn(m, m, lambda x, y: x * y, lambda x, y: y, lambda x, y: x, "mul")
        assert out.is_contiguous()  # result is always fresh and contiguous
        assert out.data is not m.data
        assert (out._op, out._parents) == ("mul", (m, m))  # same node twice
        out.grad = [1.0] * 4
        out._backward()
        assert m.grad == [2.0, 4.0, 6.0, 8.0]  # 2x, accumulated from both sides

        e = check_raises(
            ValueError, lambda: fn(a, Tensor([1.0, 2.0]), add, one, one, "add")
        )
        assert "cannot broadcast" in str(e)

    def __add__(self, other: Tensor | Scalar) -> Tensor:
        return self._binop(
            other, lambda x, y: x + y, lambda x, y: 1.0, lambda x, y: 1.0, "add"
        )

    @selftest(__add__)
    def _(fn):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[10, 20], [30, 40]])
        assert fn(a, a).tolist() == [[2, 4], [6, 8]]
        assert fn(a, 10).tolist() == [[11, 12], [13, 14]]
        assert (2 + a).tolist() == fn(a, 2).tolist()  # __radd__ is this same function
        assert (fn(a, b)._op, fn(a, b)._parents) == ("add", (a, b))

        p, q = Tensor([5.0]), Tensor([3.0])
        fn(p, q).backward()
        assert (p.grad, q.grad) == ([1.0], [1.0])  # add splits the grad evenly

    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        return self._binop(
            other, lambda x, y: x * y, lambda x, y: y, lambda x, y: x, "mul"
        )

    @selftest(__mul__)
    def _(fn):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[10, 20], [30, 40]])
        assert fn(a, a).tolist() == [[1, 4], [9, 16]]
        assert fn(a, Tensor([10, 100])).tolist() == [[10, 200], [30, 400]]  # row bcast
        assert (3 * a).tolist() == fn(a, 3).tolist()  # __rmul__
        assert (fn(a, b)._op, fn(a, b)._parents) == ("mul", (a, b))

        p, q = Tensor([5.0]), Tensor([3.0])
        fn(p, q).backward()
        assert (p.grad, q.grad) == ([3.0], [5.0])  # each side gets the other's value

    def __sub__(self, other: Tensor | Scalar) -> Tensor:
        return self._binop(
            other, lambda x, y: x - y, lambda x, y: 1.0, lambda x, y: -1.0, "sub"
        )

    @selftest(__sub__)
    def _(fn):
        a = Tensor([[5, 6], [7, 8]])
        b = Tensor([[1, 2], [3, 4]])
        assert fn(a, b).tolist() == [[4, 4], [4, 4]]
        assert fn(a, 1).tolist() == [[4, 5], [6, 7]]
        assert (fn(a, b)._op, fn(a, b)._parents) == ("sub", (a, b))

        p, q = Tensor([5.0]), Tensor([3.0])
        fn(p, q).backward()
        assert (p.grad, q.grad) == ([1.0], [-1.0])  # the right operand flips sign

    __radd__ = __add__  # 2 + t  ->  t + 2
    __rmul__ = __mul__

    def __matmul__(self, other: Tensor) -> Tensor:
        assert len(self.shape) >= 2 and len(other.shape) >= 2
        n, k = self.shape[-2:]
        k2, m = other.shape[-2:]
        assert k == k2, f"cannot matmul {self.shape} @ {other.shape}"

        batch = broadcast_shape(self.shape[:-2], other.shape[:-2])
        a = self.expand(batch + (n, k))
        b = other.expand(batch + (k, m))

        out = []
        for bi in iter_indices(batch):
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for p in range(k):
                        s += (
                            a.data[a._offset(bi + (i, p))]
                            * b.data[b._offset(bi + (p, j))]
                        )
                    out.append(s)
        return Tensor(out, batch + (n, m), _parents=(self, other), _op="matmul")

    @selftest(__matmul__)
    def _(fn):
        a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
        b = Tensor([[7, 8], [9, 10], [11, 12]])  # (3,2)
        out = fn(a, b)
        assert (out.shape, out.tolist()) == ((2, 2), [[58, 64], [139, 154]])
        assert out.is_contiguous()  # always a fresh row-major result
        assert (out._op, out._parents) == ("matmul", (a, b))

        i = Tensor([[1, 0], [0, 1]])
        assert fn(i, i).tolist() == [[1, 0], [0, 1]]

        col = Tensor([[1], [2], [3]])  # (3,1): non-square result pins (n, m) order
        assert (fn(a, col).shape, fn(a, col).tolist()) == ((2, 1), [[14], [32]])

        t = b.transpose(0, 1)  # (2,3) view: read through strides, not storage
        assert fn(t, b).tolist() == [[251, 278], [278, 308]]

        e = check_raises(AssertionError, lambda: fn(a, a))  # inner dims disagree
        assert "cannot matmul" in str(e)

        check_raises(AssertionError, lambda: fn(a, Tensor([1, 2, 3])))  # needs 2 dims

        # batched: last two dims multiply, leading dims broadcast
        ba = Tensor(list(range(12)), (2, 2, 3))
        bb = Tensor(list(range(12)), (2, 3, 2))
        assert fn(ba, bb).shape == (2, 2, 2)
        assert fn(ba, bb).tolist() == [[[10, 13], [28, 40]], [[172, 193], [244, 274]]]

        sel = Tensor([[1, 0], [0, 1], [0, 0]])  # (3,2) reused across the batch
        assert fn(ba, sel).shape == (2, 2, 2)
        assert fn(ba, sel).tolist() == [[[0, 1], [3, 4]], [[6, 7], [9, 10]]]

        wide = Tensor(list(range(40)), (2, 4, 5))
        assert fn(Tensor(list(range(12)), (1, 3, 4)), wide).shape == (2, 3, 5)

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


def label_locals(scope: dict[str, Any]) -> None:
    """name every still-unnamed Tensor in `scope` after the variable holding it"""
    for name, v in scope.items():
        if isinstance(v, Tensor) and not v.label:
            v.label = name


@selftest(label_locals)
def _(fn):
    a, b = Tensor([1.0]), Tensor([2.0])
    c = a + b
    fn({"a": a, "b": b, "c": c})
    assert (a.label, b.label, c.label) == ("a", "b", "c")  # intermediates too

    fn({"other": a})
    assert a.label == "a"  # first name wins; an existing label is never overwritten


def trace(root: Tensor) -> list[str]:
    """one line per node of root's graph, inputs first"""
    lines = []
    for i, t in enumerate(topo(root)):
        name = t.label or t._op or "leaf"
        if t.label and t._op:
            name = f"{t.label} = {t._op}"  # a label alone would hide the op
        src = ", ".join(p.label or p._op or "leaf" for p in t._parents)
        lines.append(f"{i} {name} {t.shape}" + (f" <- {src}" if src else ""))
    return lines


@selftest(trace)
def _(fn):
    x, w = Tensor([[1.0, 2.0]], label="x"), Tensor([[3.0], [4.0]], label="w")
    y = x @ w
    assert fn(y) == [
        "0 x (1, 2)",
        "1 w (2, 1)",
        "2 matmul (1, 1) <- x, w",
    ]
    assert fn(x) == ["0 x (1, 2)"]  # a leaf traces to itself

    y.label = "y"
    assert fn(y)[-1] == "2 y = matmul (1, 1) <- x, w"  # label AND op, never one alone


if __name__ == "__main__":
    from tests import tests

    tests()
