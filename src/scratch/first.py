from __future__ import annotations

import math
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
type UnFn = Callable[[Scalar], Scalar]
type Key = int | slice | tuple[int | slice, ...] | list[int] | Tensor


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
        offset: int = 0,
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
        self.offset = offset  # where this view starts in data
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
        assert c.offset == 0

        d = Tensor.__new__(Tensor)
        fn(d, [0, 1, 2, 3], (2,), (1,), 2)  # a view starting 2 elements in
        assert (d.offset, d.tolist()) == (2, [2, 3])

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
        return self.offset + sum(i * s for i, s in zip(idx, self.strides))

    @selftest(_offset)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a, (1, 2)) == 5
        assert a.data[fn(a, (1, 2))] == 6

        b = Tensor(list(range(24)), (2, 3, 4))
        assert fn(b, (1, 2, 3)) == 23

        v = Tensor([0, 1, 2, 3, 4, 5], (2,), (2,), 1)  # reads data[1], data[3]
        assert (fn(v, (0,)), fn(v, (1,))) == (1, 3)

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
        out = Tensor(
            self.data,
            tuple(shape),
            tuple(strides),
            self.offset,
            _parents=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            assert out.grad is not None
            g = Tensor(out.grad, out.shape).transpose(d0, d1)  # back to self.shape
            self._accum(g.flat())  # .flat() re-reads it in self's logical order

        out._backward = _backward
        return out

    @selftest(transpose)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        t = fn(a, 0, 1)
        assert t.shape == (3, 2)
        assert t.strides == (1, 3)
        assert t.data is a.data  # a view, not a copy
        assert t.tolist() == [[1, 4], [2, 5], [3, 6]]
        assert (t._op, t._parents) == ("transpose", (a,))  # stays in the graph
        assert fn(a[:, 1:], 0, 1).tolist() == [[2, 5], [3, 6]]  # keeps the offset

        # the view reads the base's 1..6 as [1, 4, 2, 5, 3, 6]. seed each slot with
        # the value it reads, and a correct backward gives every element its own back
        g = fn(Tensor([1, 2, 3, 4, 5, 6], (2, 3)), 0, 1)  # (3,2) view of a (2,3)
        g.grad = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        g._backward()
        assert g._parents[0].grad == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2 gets 2, not 4

    @property
    def T(self) -> Tensor:
        assert len(self.shape) == 2, f"T is 2-D only, got {self.shape}"
        return self.transpose(0, 1)

    @selftest(T.fget)  # type: ignore[attr-defined]
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a).tolist() == a.transpose(0, 1).tolist()  # just the 2-D shorthand
        assert fn(a).shape == (3, 2)

        e = check_raises(AssertionError, lambda: fn(Tensor(list(range(8)), (2, 2, 2))))
        assert "2-D only" in str(e)  # batched code must say which dims it means

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
        return (
            self.offset == 0
            and len(self.data) == self.numel
            and self.strides == contiguous_strides(self.shape)
        )

    @selftest(is_contiguous)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a)
        assert not fn(a.transpose(0, 1))
        assert not fn(a[1])  # starts 3 elements into data
        assert not fn(a[0])  # reads only 3 of data's 6

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
        out = Tensor(self.flat(), self.shape, _parents=(self,), _op="contiguous")

        def _backward() -> None:
            assert out.grad is not None
            self._accum(out.grad)  # same shape, same logical order: straight through

        out._backward = _backward
        return out

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

        tc.grad = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tc._backward()
        assert t.grad == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # copy only moved the storage

        s = Tensor([1, 2, 3, 4])[1:3]
        sc = fn(s)
        assert (sc.data, sc.offset) == ([2, 3], 0)  # a slice copies out just its part

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
        out = Tensor(src.data, shape, _parents=(src,), _op="reshape")

        def _backward() -> None:
            assert out.grad is not None
            src._accum(out.grad)  # only the shape changed, so the flat grad carries

        out._backward = _backward
        return out

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

        base = Tensor([[1, 2, 3], [4, 5, 6]])
        r6 = fn(base, 6)
        r6.grad = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        r6._backward()
        assert base.grad == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # flat order is preserved

        p = Tensor([1, 2, 3, 4, 5, 6])[2:]  # a slice: offset 2
        rp = fn(p, 2, 2)  # copied out before relabeling
        assert (rp.tolist(), rp.data) == ([[3, 4], [5, 6]], [3, 4, 5, 6])

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
        out = Tensor(
            self.data,
            tuple(shape),
            tuple(strides),
            self.offset,
            _parents=(self,),
            _op="expand",
        )

        def _backward() -> None:
            assert out.grad is not None
            self._accum(unbroadcast(out.grad, out.shape, self.shape))

        out._backward = _backward
        return out

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

        src = Tensor([10.0, 20.0, 30.0])
        ex = fn(src, (2, 3))
        ex.grad = [1.0] * 6
        ex._backward()
        assert src.grad == [2.0, 2.0, 2.0]  # a stride-0 dim sums its copies back down

        assert fn(Tensor([1, 2, 3, 4])[2:], (2, 2)).tolist() == [[3, 4], [3, 4]]

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

    def __truediv__(self, other: Tensor | Scalar) -> Tensor:
        return self._binop(
            other,
            lambda x, y: x / y,
            lambda x, y: 1.0 / y,
            lambda x, y: -x / (y * y),
            "div",
        )

    @selftest(__truediv__)
    def _(fn):
        a = Tensor([[2.0, 4.0], [6.0, 8.0]])
        assert fn(a, 2.0).tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert fn(a, Tensor([2.0, 4.0])).tolist() == [[1.0, 1.0], [3.0, 2.0]]  # bcast

        p, q = Tensor([6.0]), Tensor([3.0])
        fn(p, q).backward()
        assert (p.grad, q.grad) == ([1 / 3], [-6 / 9])  # 1/q and -p/q^2

    __radd__ = __add__  # 2 + t  ->  t + 2
    __rmul__ = __mul__

    def _unop(self, f: UnFn, df: BinFn, op: str) -> Tensor:
        xs = self.flat()
        ys = [f(x) for x in xs]
        out = Tensor(ys, self.shape, _parents=(self,), _op=op)

        def _backward() -> None:
            assert out.grad is not None
            self._accum([g * df(x, y) for g, x, y in zip(out.grad, xs, ys)])

        out._backward = _backward
        return out

    @selftest(_unop)
    def _(fn):
        a = Tensor([1.0, -2.0, 3.0])
        out = fn(a, lambda x: x * x, lambda x, y: 2 * x, "square")
        assert out.tolist() == [1.0, 4.0, 9.0]
        assert (out._op, out._parents) == ("square", (a,))

        out.grad = [1.0, 10.0, 100.0]
        out._backward()
        assert a.grad == [2.0, -40.0, 600.0]  # g * 2x

        t = Tensor([[1.0, 2.0], [3.0, 4.0]]).T
        assert fn(t, lambda x: x, lambda x, y: 1.0, "id").tolist() == t.tolist()

    def __neg__(self) -> Tensor:
        return self._unop(lambda x: -x, lambda x, y: -1.0, "neg")

    @selftest(__neg__)
    def _(fn):
        a = Tensor([1.0, -2.0])
        assert fn(a).tolist() == (-a).tolist() == [-1.0, 2.0]
        fn(a).backward()
        assert a.grad == [-1.0, -1.0]

    def exp(self) -> Tensor:
        return self._unop(math.exp, lambda x, y: y, "exp")

    @selftest(exp)
    def _(fn):
        a = Tensor([0.0, 1.0, -math.inf])
        assert fn(a).tolist() == [1.0, math.e, 0.0]  # -inf -> 0: a masked slot
        fn(a).backward()
        assert a.grad == [1.0, math.e, 0.0]  # the output is its own derivative

    def log(self) -> Tensor:
        return self._unop(math.log, lambda x, y: 1.0 / x, "log")

    @selftest(log)
    def _(fn):
        a = Tensor([1.0, math.e])
        assert fn(a).tolist() == [0.0, 1.0]
        fn(a).backward()
        assert a.grad == [1.0, 1 / math.e]

        check_raises(ValueError, lambda: fn(Tensor([0.0])))  # torch gives -inf

    def sqrt(self) -> Tensor:
        return self._unop(math.sqrt, lambda x, y: 0.5 / y, "sqrt")

    @selftest(sqrt)
    def _(fn):
        a = Tensor([4.0, 9.0])
        assert fn(a).tolist() == [2.0, 3.0]
        fn(a).backward()
        assert a.grad == [0.25, 1 / 6]

    def relu(self) -> Tensor:
        return self._unop(
            lambda x: x if x > 0 else 0.0, lambda x, y: 1.0 if x > 0 else 0.0, "relu"
        )

    @selftest(relu)
    def _(fn):
        a = Tensor([-2.0, 0.0, 3.0])
        assert fn(a).tolist() == [0.0, 0.0, 3.0]
        fn(a).backward()
        assert a.grad == [0.0, 0.0, 1.0]  # 0 at exactly 0, as torch picks

    def masked_fill(self, mask: Tensor, value: Scalar) -> Tensor:
        m = mask.expand(self.shape).flat()  # the mask broadcasts to self, not back
        xs = self.flat()
        ys = [value if mi else x for x, mi in zip(xs, m)]
        out = Tensor(ys, self.shape, _parents=(self,), _op="masked_fill")

        def _backward() -> None:
            assert out.grad is not None
            self._accum([0.0 if mi else g for g, mi in zip(out.grad, m)])

        out._backward = _backward
        return out

    @selftest(masked_fill)
    def _(fn):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        m = Tensor([[False, True], [False, False]])
        out = fn(a, m, -math.inf)
        assert out.tolist() == [[1.0, -math.inf], [3.0, 4.0]]
        assert (out._op, out._parents) == ("masked_fill", (a,))  # mask is no parent

        row = Tensor([True, False])  # broadcasts over both rows
        assert fn(a, row, 0.0).tolist() == [[0.0, 2.0], [0.0, 4.0]]
        t = Tensor(list(range(1, 9)), (2, 2, 2))
        tm = fn(t, m, 0)
        assert tm.shape == (2, 2, 2)
        assert tm.tolist() == [
            [[1, 0], [3, 4]],
            [[5, 0], [7, 8]],
        ]

        out.grad = [1.0, 2.0, 3.0, 4.0]
        out._backward()
        assert a.grad == [1.0, 0.0, 3.0, 4.0]  # nothing flows into a filled slot

        e = check_raises(ValueError, lambda: fn(a, Tensor([True, False, True]), 0.0))
        assert "cannot expand" in str(e)

    def _slice(self, key: tuple[int | slice, ...]) -> Tensor:
        nd = len(self.shape)
        assert len(key) <= nd, f"too many indices for shape {self.shape}"
        key = key + (slice(None),) * (nd - len(key))
        offset = self.offset
        shape: list[int] = []
        strides: list[int] = []
        starts: list[int] = []  # per input dim: the first index read
        steps: list[int] = []  # per input dim: 0 where an int dropped it
        for k, size, st in zip(key, self.shape, self.strides):
            if isinstance(k, int):
                if not -size <= k < size:
                    raise IndexError(f"index {k} out of range {size}")
                k %= size
                offset += k * st
                starts.append(k)
                steps.append(0)
            elif isinstance(k, slice):
                start, stop, step = k.indices(size)
                assert step > 0, "negative slice steps are not supported"
                offset += start * st
                shape.append(len(range(start, stop, step)))
                strides.append(st * step)
                starts.append(start)
                steps.append(step)
            else:
                raise TypeError(f"cannot index with {type(k).__name__}")
        out = Tensor(
            self.data,
            tuple(shape),
            tuple(strides),
            offset,
            _parents=(self,),
            _op="slice",
        )

        def _backward() -> None:
            assert out.grad is not None
            g = [0.0] * self.numel
            cs = contiguous_strides(self.shape)
            for k, oi in enumerate(iter_indices(out.shape)):
                pos, j = 0, 0
                for start, step, s in zip(starts, steps, cs):
                    i = start
                    if step:  # a kept dim: walk it
                        i += oi[j] * step
                        j += 1
                    pos += i * s
                g[pos] += out.grad[k]  # back to where the view read it
            self._accum(g)

        out._backward = _backward
        return out

    @selftest(_slice)
    def _(fn):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        r = fn(a, (1,))  # a[1]
        assert (r.tolist(), r.shape, r.strides, r.offset) == ([4, 5, 6], (3,), (1,), 3)
        assert r.data is a.data  # a view: only metadata changed
        assert (r._op, r._parents) == ("slice", (a,))

        c = fn(a, (slice(None), -1))  # a[:, -1]
        assert (c.tolist(), c.strides, c.offset) == ([3, 6], (3,), 2)
        e = fn(a, (slice(None), slice(None, None, 2)))  # a[:, ::2]
        assert (e.tolist(), e.strides) == ([[1, 3], [4, 6]], (3, 2))
        assert (fn(a, (0, 1)).shape, fn(a, (0, 1)).tolist()) == ((), 2)  # a[0, 1]
        assert fn(r, (slice(1, None),)).tolist() == [5, 6]  # offsets add up
        assert fn(a.T, (0,)).tolist() == [1, 4]  # a view of a view
        assert fn(a, (slice(5, 9),)).shape == (0, 3)  # clamped, empty

        b = Tensor([[1, 2, 3], [4, 5, 6]])
        s = fn(b, (slice(None), slice(1, None)))  # b[:, 1:]
        s.grad = [10, 20, 30, 40]
        s._backward()
        assert b.grad == [0, 10, 20, 0, 30, 40]  # back into the positions it read

        b = Tensor([[1, 2, 3], [4, 5, 6]])
        s = fn(b, (1,))  # an int drops the dim
        s.grad = [7, 8, 9]
        s._backward()
        assert b.grad == [0, 0, 0, 7, 8, 9]

        check_raises(IndexError, lambda: fn(a, (2,)))
        check_raises(AssertionError, lambda: fn(a, (0, 0, 0)))  # too many indices
        check_raises(AssertionError, lambda: fn(a, (slice(None, None, -1),)))
        check_raises(TypeError, lambda: fn(a, ("x",)))

    def __getitem__(self, key: Key) -> Tensor:
        if isinstance(key, (int, slice)):
            return self._slice((key,))  # basic indexing: a view
        if isinstance(key, tuple):
            return self._slice(key)
        ids = Tensor(key) if isinstance(key, list) else key  # ids: a copy
        if not isinstance(ids, Tensor):
            raise TypeError(f"cannot index with {type(ids).__name__}")
        assert len(self.shape) >= 1, "cannot index a 0-d tensor"
        rows = self.shape[0]
        row = prod(self.shape[1:])  # elements per row
        idx: list[int] = []
        for i in ids.flat():
            assert isinstance(i, int), f"ids must be ints, got {i!r}"
            if not -rows <= i < rows:
                raise IndexError(f"id {i} out of range {rows}")
            idx.append(i % rows)  # -1 -> rows - 1
        flat = self.flat()
        data: list[Scalar] = []
        for i in idx:
            data += flat[i * row : (i + 1) * row]
        out = Tensor(data, ids.shape + self.shape[1:], _parents=(self,), _op="index")

        def _backward() -> None:
            assert out.grad is not None
            g = [0.0] * self.numel
            for k, i in enumerate(idx):
                for j in range(row):
                    g[i * row + j] += out.grad[k * row + j]  # repeated ids pile up
            self._accum(g)

        out._backward = _backward
        return out

    @selftest(__getitem__)
    def _(fn):
        w = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 rows
        assert fn(w, Tensor([2, 0])).tolist() == [[5, 6], [1, 2]]
        assert fn(w, Tensor([[0, 1], [2, 2]])).tolist() == [
            [[1, 2], [3, 4]],
            [[5, 6], [5, 6]],
        ]  # (2, 2) ids -> (2, 2, 2)
        assert fn(w, Tensor(1)).tolist() == [3, 4]  # a 0-d id picks one row
        assert fn(Tensor([10, 20, 30]), Tensor([2, 1])).tolist() == [30, 20]  # 1-D
        assert fn(w.T, Tensor([1])).tolist() == [[2, 4, 6]]  # a view
        out = fn(w, Tensor([2, 0, 2]))
        assert (out._op, out._parents) == ("index", (w,))  # ids are no parent

        out.grad = [1] * 6
        out._backward()
        assert w.grad == [1, 1, 0, 0, 2, 2]  # row 2 was picked twice

        w = Tensor([[1, 2], [3, 4], [5, 6]])
        out = fn(w, Tensor([2, 0, 1, 2]))
        out.grad = [10, 20, 1, 2, 3, 4, 5, 6]  # one grad row per id
        out._backward()
        assert w.grad == [
            1, 2,    # row 0 <- id 0's grad
            3, 4,    # row 1 <- id 1's grad
            15, 26,  # row 2 <- picked twice: [10, 20] + [5, 6]
        ]  # fmt: skip

        w = Tensor([[1, 2], [3, 4], [5, 6]])
        out = fn(w, Tensor([1, 2, 1]))
        out.grad = [1, 2, 3, 4, 5, 6]
        out._backward()
        assert w.grad == [
            0, 0,  # row 0 <- never picked
            6, 8,  # row 1 <- picked twice: [1, 2] + [5, 6]
            3, 4,  # row 2 <- picked once
        ]  # fmt: skip

        e = check_raises(IndexError, lambda: fn(w, Tensor([3])))
        assert "out of range" in str(e)
        assert fn(w, Tensor([-1])).tolist() == [[5, 6]]  # negative ids wrap
        assert fn(w, [2, 0]).tolist() == [[5, 6], [1, 2]]  # a list is ids too

        assert fn(w, 1).data is w.data  # an int or slice is a view, not a copy
        assert fn(w, (slice(None), 0)).tolist() == [1, 3, 5]  # w[:, 0]
        assert [r.tolist() for r in w] == [[1, 2], [3, 4], [5, 6]]  # w[0], w[1], ...
        check_raises(TypeError, lambda: fn(w, "a"))

    def __matmul__(self, other: Tensor) -> Tensor:
        assert len(self.shape) >= 2 and len(other.shape) >= 2
        n, k = self.shape[-2:]
        k2, m = other.shape[-2:]
        assert k == k2, f"cannot matmul {self.shape} @ {other.shape}"

        batch = broadcast_shape(self.shape[:-2], other.shape[:-2])
        a = self.expand(batch + (n, k))
        b = other.expand(batch + (k, m))

        data = []
        for bi in iter_indices(batch):
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for p in range(k):
                        s += (
                            a.data[a._offset(bi + (i, p))]
                            * b.data[b._offset(bi + (p, j))]
                        )
                    data.append(s)
        out = Tensor(data, batch + (n, m), _parents=(self, other), _op="matmul")

        def _backward() -> None:
            assert out.grad is not None  # backward seeds it before calling us
            nd = len(out.shape)
            gc = Tensor(out.grad, out.shape)
            ga = gc @ b.transpose(nd - 2, nd - 1)  # dC @ B.T -> batch + (n, k)
            gb = a.transpose(nd - 2, nd - 1) @ gc  # A.T @ dC -> batch + (k, m)
            # a/b are the expanded operands, so fold the batch dims back down
            self._accum(unbroadcast(ga.flat(), ga.shape, self.shape))
            other._accum(unbroadcast(gb.flat(), gb.shape, other.shape))

        out._backward = _backward
        return out

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

        # backward: dA = dC @ B.T and dB = A.T @ dC, i.e. two more matmuls
        p, q = Tensor([[1.0, 2.0]]), Tensor([[3.0], [4.0]])
        out = fn(p, q)
        out.grad = [10.0]  # a non-unit seed has to ride through both of them
        out._backward()
        assert p.grad == [30.0, 40.0]  # g @ q.T
        assert q.grad == [10.0, 20.0]  # p.T @ g

        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        fn(x, y).backward()  # seed of ones, so each grad is a plain sum
        assert x.grad == [15.0, 19.0, 23.0] * 2  # row sums of y, once per row of x
        assert y.grad == [5.0, 5.0, 7.0, 7.0, 9.0, 9.0]  # column sums of x

        sq = Tensor([[1.0, 2.0], [3.0, 4.0]])
        fn(sq, sq).backward()  # one node on both sides: the two grads accumulate
        assert sq.grad == [7.0, 11.0, 9.0, 13.0]

        bx = Tensor([float(v) for v in range(12)], (2, 2, 3))
        bs = Tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # (3,2) shared by the batch
        fn(bx, bs).backward()
        assert bx.grad == [1.0, 1.0, 0.0] * 4  # row sums of bs, for every batch row
        assert bs.grad == [18.0, 18.0, 22.0, 22.0, 26.0, 26.0]  # folded over the batch

        tv = Tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)  # a strided operand
        fn(tv, sq).backward()
        assert tv.grad == [3.0, 7.0, 3.0, 7.0]  # lands on the view, in logical order

    def sum(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        nd = len(self.shape)
        if dim is None:
            kept: Shape = (1,) * nd
            shape: Shape = kept if keepdim else ()
        else:
            assert -nd <= dim < nd, f"dim {dim} out of range for {self.shape}"
            dim %= nd
            kept = self.shape[:dim] + (1,) + self.shape[dim + 1 :]
            shape = kept if keepdim else self.shape[:dim] + self.shape[dim + 1 :]
        data = unbroadcast(self.flat(), self.shape, kept)  # fold the dim down to 1
        out = Tensor(data, shape, _parents=(self,), _op="sum")

        def _backward() -> None:
            assert out.grad is not None
            g = Tensor(out.grad, kept).expand(self.shape)  # stretch back out
            self._accum(g.flat())

        out._backward = _backward
        return out

    @selftest(sum)
    def _(fn):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert (fn(a).shape, fn(a).tolist()) == ((), 21.0)  # everything
        assert fn(a, 0).tolist() == [5.0, 7.0, 9.0]
        assert fn(a, 1).tolist() == fn(a, -1).tolist() == [6.0, 15.0]
        assert (fn(a, 1, True).shape, fn(a, 1, True).tolist()) == ((2, 1), [[6], [15]])
        assert fn(a, None, True).shape == (1, 1)
        assert fn(a.T, 0).tolist() == [6.0, 15.0]  # reads through strides
        assert (fn(a, 0)._op, fn(a, 0)._parents) == ("sum", (a,))

        out = fn(a, 1)
        out.grad = [10.0, 100.0]
        out._backward()
        assert a.grad == [10.0] * 3 + [100.0] * 3  # each row's grad, per element

        e = check_raises(AssertionError, lambda: fn(a, 2))
        assert "out of range" in str(e)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        n = self.numel if dim is None else self.shape[dim]
        return self.sum(dim, keepdim) * (1.0 / n)

    @selftest(mean)
    def _(fn):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert fn(a).tolist() == 3.5
        assert fn(a, 0).tolist() == [2.5, 3.5, 4.5]
        assert fn(a, -1, True).tolist() == [[2.0], [5.0]]

        fn(a).backward()  # a scalar root: seeded with a single 1.0
        assert a.grad == [1 / 6] * 6

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


def cat(tensors: list[Tensor], dim: int = 0) -> Tensor:
    assert tensors, "cat needs at least one tensor"
    first = tensors[0]
    nd = len(first.shape)
    assert -nd <= dim < nd, f"dim {dim} out of range for {first.shape}"
    dim %= nd
    rest = first.shape[:dim] + first.shape[dim + 1 :]
    for t in tensors[1:]:
        same = t.shape[:dim] + t.shape[dim + 1 :] == rest
        assert len(t.shape) == nd and same, (
            f"cannot cat {first.shape} with {t.shape} on dim {dim}"
        )

    outer = prod(first.shape[:dim])
    inner = prod(first.shape[dim + 1 :])
    sizes = [t.shape[dim] * inner for t in tensors]  # block size per tensor
    flats = [t.flat() for t in tensors]
    data: list[Scalar] = []
    for o in range(outer):
        for flat, sz in zip(flats, sizes):
            data += flat[o * sz : (o + 1) * sz]
    size = sum(t.shape[dim] for t in tensors)
    shape = first.shape[:dim] + (size,) + first.shape[dim + 1 :]
    out = Tensor(data, shape, _parents=tuple(tensors), _op="cat")

    def _backward() -> None:
        assert out.grad is not None
        grads: list[Grad] = [[] for _ in tensors]
        pos = 0
        for _ in range(outer):
            for g, sz in zip(grads, sizes):
                g += out.grad[pos : pos + sz]  # cut the grad into the same blocks
                pos += sz
        for t, g in zip(tensors, grads):
            t._accum(g)

    out._backward = _backward
    return out


@selftest(cat)
def _(fn):
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0]])
    c = Tensor([[7.0], [8.0]])
    assert fn([a, b], 0).tolist() == [[1, 2], [3, 4], [5, 6]]  # rows stacked
    assert (
        fn([a, c], 1).tolist() == fn([a, c], -1).tolist() == [[1, 2, 7], [3, 4, 8]]
    )  # columns side by side
    assert fn([a, c], -1).shape == (2, 3)
    assert fn([a.T, a], 0).tolist() == [[1, 3], [2, 4], [1, 2], [3, 4]]  # a view
    assert (fn([a, c], 1)._op, fn([a, c], 1)._parents) == ("cat", (a, c))

    out = fn([a, c], 1)
    out.grad = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out._backward()
    assert a.grad == [1.0, 2.0, 4.0, 5.0]  # each input gets back its own slots
    assert c.grad == [3.0, 6.0]

    d = Tensor([1.0, 2.0])
    fn([d, d]).backward()
    assert d.grad == [2.0, 2.0]  # the same tensor twice: both blocks accumulate

    e = check_raises(AssertionError, lambda: fn([a, c], 0))
    assert "cannot cat" in str(e)


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
