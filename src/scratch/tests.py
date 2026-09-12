import random

import first
import torch
from first import (
    Shape,
    Tensor,
    broadcast_shape,
    contiguous_strides,
    flatten,
    infer_shape,
    iter_indices,
    label_locals,
    prod,
    topo,
    trace,
)
from test_utils import check_raises, run_tests


def t1():
    Tensor.__init__.test()
    Tensor._offset.test()

    a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    assert a.data == [1, 2, 3, 4, 5, 6]
    assert a.shape == (2, 3)

    assert a._offset((1, 2)) == 5
    assert a.data[a._offset((1, 2))] == 6


def t2():
    contiguous_strides.test()

    assert contiguous_strides((2, 3)) == (3, 1)
    a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    assert a.strides == (3, 1)

    assert contiguous_strides((2, 3, 4)) == (12, 4, 1)
    b = Tensor(list(range(24)), (2, 3, 4))
    assert b.strides == (12, 4, 1)
    assert b._offset((1, 2, 3)) == 23

    e = check_raises(
        AssertionError,
        lambda: b._offset((1, 2)),
        "_offset accepted 2 indices for a 3-D shape",
    )
    assert "shape" in str(e)


def t3():
    Tensor.transpose.test()

    a = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
    assert a.strides == (3, 1)

    assert [
        [a.data[a._offset((i, j))] for j in range(a.shape[1])]
        for i in range(a.shape[0])
    ] == [[1, 2, 3], [4, 5, 6]]

    t = a.transpose(0, 1)
    assert t.shape == (3, 2)
    assert t.strides == (1, 3)
    assert t.data is a.data
    assert [
        [t.data[t._offset((i, j))] for j in range(t.shape[1])]
        for i in range(t.shape[0])
    ] == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]


def t4():
    Tensor.tolist.test()
    Tensor.__repr__.test()

    a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    assert a.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert a.transpose(0, 1).tolist() == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]
    assert "data=[[1, 2, 3], [4, 5, 6]]" in str(a)
    assert "data=[[1, 2, 3], [4, 5, 6]]" in repr(a)


def t5():
    infer_shape.test()
    flatten.test()

    a_list = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    assert infer_shape(a_list) == (2, 3)

    assert flatten(a_list) == [1, 2, 3, 4, 5, 6]

    a = Tensor(a_list)
    assert a.shape == (2, 3)
    assert a.data == [1, 2, 3, 4, 5, 6]

    b = Tensor(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ]
    )

    assert b.shape == (2, 2, 2)
    assert b.data == [1, 2, 3, 4, 5, 6, 7, 8]


def t6():
    prod.test()
    Tensor.numel.fget.test()
    Tensor.is_contiguous.test()
    Tensor.flat.test()
    Tensor.contiguous.test()

    assert prod((2, 3)) == 6
    assert prod((2, 3, 4)) == 24
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    assert a.numel == 6

    t = a.transpose(0, 1)
    assert t.data is a.data
    assert a.is_contiguous()
    assert not t.is_contiguous()
    assert t.data == [1, 2, 3, 4, 5, 6]  # storage order
    assert t.tolist() == [[1, 4], [2, 5], [3, 6]]  # logical order
    assert t.shape == (3, 2)
    assert t.strides == (1, 3)

    assert t.flat() == [1, 4, 2, 5, 3, 6]

    tc = t.contiguous()
    assert tc.data == [1, 4, 2, 5, 3, 6]
    assert tc.shape == t.shape == (3, 2)
    assert tc.strides == (2, 1)
    assert tc.is_contiguous()
    assert (
        tc.tolist() == t.tolist() == [[1, 4], [2, 5], [3, 6]]
    )  # same values, different storage
    assert tc.data is not t.data

    # contiguous returns the same Tensor reference if already contiguous
    tcc = tc.contiguous()
    assert tcc is tc


def t7():
    Tensor.reshape.test()

    def dshst(t: Tensor):
        return (t.data, t.shape, t.strides)

    a = Tensor([[1, 2, 3], [4, 5, 6]])
    assert dshst(a) == ([1, 2, 3, 4, 5, 6], (2, 3), (3, 1))
    ar = a.reshape(3, 2)
    assert ar.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert dshst(ar) == ([1, 2, 3, 4, 5, 6], (3, 2), (2, 1))
    assert ar.data is a.data  # no copy, already contiguous

    # passing shape as tuple
    assert dshst(a.reshape((3, 2))) == dshst(ar)

    # infer dim
    assert dshst(a.reshape(-1, 2)) == dshst(ar)
    assert dshst(a.reshape(3, -1)) == dshst(ar)
    assert dshst(a.reshape(-1)) == dshst(a.reshape(6))

    e = check_raises(
        AssertionError,
        lambda: a.reshape(1, 2),
    )
    assert "cannot reshape" in str(e)

    t = a.transpose(0, 1)
    assert t.data == [1, 2, 3, 4, 5, 6]
    assert t.tolist() == [[1, 4], [2, 5], [3, 6]]
    tr = t.reshape(6)
    assert tr.data is not t.data  # copy created as non-contiguous
    assert tr.data == [1, 4, 2, 5, 3, 6]

    assert t.reshape(2, 3).tolist() == [[1, 4, 2], [5, 3, 6]]

    tr2 = t.reshape(t.shape)
    assert tr2.data is not t.data
    assert tr2.tolist() == t.tolist()


def t8():
    broadcast_shape.test()

    for s1, s2, res in [
        ((128, 30), (30,), (128, 30)),
        ((2, 3, 4), (3, 1), (2, 3, 4)),
        ((5,), (), (5,)),
        ((2, 1, 4), (3, 4), (2, 3, 4)),
        ((1,), (7, 8), (7, 8)),
        ((2, 3), (2, 3), (2, 3)),
    ]:
        assert (
            broadcast_shape(s1, s2) == res == tuple(torch.broadcast_shapes(s1, s2))
        ), (
            s1,
            s2,
        )

    # mismatched dim, neither of them 1
    e = check_raises(ValueError, lambda: broadcast_shape((2, 3), (4, 3)))
    assert "cannot broadcast" in str(e)


def t9():
    iter_indices.test()
    Tensor.indices.test()
    Tensor.expand.test()

    a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    assert list(a.indices()) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    t = a.transpose(0, 1)
    assert list(t.indices()) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert list(Tensor([1, 2, 3]).indices()) == [(0,), (1,), (2,)]
    assert list(Tensor(5).indices()) == [()]  # 0-d tensor: a single empty index

    c = Tensor([[1], [2]])  # (2,1)
    e = c.expand((2, 3))
    assert (e.shape, e.strides) == ((2, 3), (1, 0))
    assert e.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert e.data is c.data  # still no copy

    r = Tensor([10, 20, 30])  # (3,) -> padded to (1,3)
    assert r.expand((2, 3)).strides == (0, 1)
    assert r.expand((2, 3)).tolist() == [[10, 20, 30], [10, 20, 30]]

    b = Tensor([[[1, 2], [3, 4]]])  # (1,2,2)
    assert (
        b.expand((3, 2, 2)).tolist()
        == torch.tensor(b.tolist()).expand(3, 2, 2).tolist()
    )

    e = check_raises(ValueError, lambda: Tensor([[1, 2], [3, 4]]).expand((2, 3)))
    assert "cannot expand" in str(e)


def t10():
    Tensor._binop.test()
    Tensor.__add__.test()
    Tensor.__mul__.test()
    Tensor.__sub__.test()

    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([10, 20, 30])  # (3,) -> broadcasts across rows
    c = Tensor([[1], [2]])  # (2,1) -> broadcasts across columns
    ta, tb, tc = (torch.tensor(t.tolist()) for t in (a, b, c))

    assert (a + b).tolist() == (ta + tb).tolist()
    assert (a * b).tolist() == (ta * tb).tolist()
    assert (a - b).tolist() == (ta - tb).tolist()
    assert (a + c).tolist() == (ta + tc).tolist()
    assert (a + a).tolist() == (ta + ta).tolist()

    # scalars, and the reflected forms that map to __radd__ / __rmul__
    assert (a + 10).tolist() == (ta + 10).tolist()
    assert (10 + a).tolist() == (10 + ta).tolist()
    assert (a * 2).tolist() == (a + a).tolist() == (ta * 2).tolist()
    assert (2 * a).tolist() == (2 * ta).tolist()
    assert (a - 1).tolist() == (ta - 1).tolist()

    # the result is a fresh contiguous tensor, never a view of either operand
    out = a + b
    assert out.shape == (2, 3)
    assert out.is_contiguous()
    assert out.data is not a.data and out.data is not b.data

    # broadcasting still applies to a transposed (non-contiguous) operand
    t = a.transpose(0, 1)  # (3,2)
    assert (t + Tensor([100, 200])).tolist() == (
        ta.T + torch.tensor([100, 200])
    ).tolist()

    e = check_raises(ValueError, lambda: a + Tensor([1, 2]))
    assert "cannot broadcast" in str(e)


def rnd(shape: Shape):
    return Tensor([random.uniform(-3, 3) for _ in range(prod(shape))], shape)


def tt(t: Tensor):
    return torch.tensor(t.tolist(), dtype=torch.float64)


def matmul2d(self: Tensor, other: Tensor) -> Tensor:
    assert len(self.shape) == 2 and len(other.shape) == 2
    n, k = self.shape
    k2, m = other.shape
    assert k == k2, f"cannot matmul {self.shape} @ {other.shape}"

    out = [0.0] * (n * m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += self.data[self._offset((i, p))] * other.data[other._offset((p, j))]
            out[i * m + j] = s
    return Tensor(out, (n, m))


def t11():
    random.seed(0)

    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
    b = Tensor([[7, 8], [9, 10], [11, 12]])  # (3,2)
    ta, tb = torch.tensor(a.tolist()), torch.tensor(b.tolist())

    assert matmul2d(a, b).shape == (2, 2)
    assert matmul2d(a, b).tolist() == (ta @ tb).tolist()

    # transposed operands: the loop must go through _offset, not raw storage
    assert matmul2d(a.transpose(0, 1), a).tolist() == (ta.T @ ta).tolist()
    assert matmul2d(b.transpose(0, 1), b).tolist() == (tb.T @ tb).tolist()

    out = matmul2d(a, b)
    assert out.is_contiguous()
    assert out.data is not a.data and out.data is not b.data

    e = check_raises(AssertionError, lambda: matmul2d(a, a))  # (2,3) @ (2,3)
    assert "cannot matmul" in str(e)

    check_raises(
        AssertionError, lambda: matmul2d(a, Tensor([1, 2, 3]))
    )  # 2-D operands only

    for s1, s2 in [
        ((2, 3), (3, 4)),
        ((1, 5), (5, 1)),
        ((7, 7), (7, 7)),
        ((4, 1), (1, 6)),
    ]:
        m1, m2 = rnd(s1), rnd(s2)
        assert torch.allclose(tt(matmul2d(m1, m2)), tt(m1) @ tt(m2)), (s1, s2)

    m1, m2 = rnd((3, 4)), rnd((5, 4))
    assert torch.allclose(tt(matmul2d(m1, m2.transpose(0, 1))), tt(m1) @ tt(m2).T)

    w = rnd((30, 20))  # (out, in), torch's layout
    bias = rnd((30,))
    x = rnd((128, 20))

    y = (
        matmul2d(x, w.transpose(0, 1)) + bias
    )  # broadcasting adds bias across all 128 rows
    assert y.shape == (128, 30)
    assert torch.allclose(tt(y), torch.nn.functional.linear(tt(x), tt(w), tt(bias)))


def t12():
    Tensor.__matmul__.test()
    random.seed(0)
    for s1, s2 in [
        ((2, 3), (3, 4)),
        ((2, 3, 5, 8), (2, 3, 8, 4)),  # matching batch
        ((2, 3, 5, 8), (8, 4)),  # one matrix reused
        ((5, 8), (3, 8, 4)),  # batch only on the right
        ((1, 3, 5, 8), (2, 1, 8, 4)),  # batch dims broadcast against each other
        ((4, 2, 3), (4, 3, 6)),
    ]:
        a, b = rnd(s1), rnd(s2)
        mine, ref = a @ b, tt(a) @ tt(b)
        assert mine.shape == tuple(ref.shape), (s1, s2)
        assert torch.allclose(tt(mine), ref), (s1, s2)

    # the shape attention actually uses: (B, nh, T, hs) @ (B, nh, hs, T)
    q, k = rnd((2, 4, 6, 8)), rnd((2, 4, 6, 8))
    scores = q @ k.transpose(2, 3)
    assert scores.shape == (2, 4, 6, 6)
    assert torch.allclose(tt(scores), tt(q) @ tt(k).transpose(2, 3))


def t13():
    topo.test()
    label_locals.test()
    trace.test()

    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    w = Tensor([[10.0, 20.0], [30.0, 40.0]])
    b = Tensor([100.0, 200.0])

    y = x @ w + b
    z = y * y
    label_locals(locals())  # names the intermediates too, not just the inputs

    assert topo(z)[-1] is z  # root comes last
    assert len(topo(z)) == 6  # y feeds mul twice, but appears once
    assert [t.label or t._op for t in topo(z)] == [
        "x",
        "w",
        "matmul",
        "b",
        "y",
        "z",
    ]

    assert trace(z) == [
        "0 x (2, 2)",
        "1 w (2, 2)",
        "2 matmul (2, 2) <- x, w",
        "3 b (2,)",
        "4 y = add (2, 2) <- matmul, b",
        "5 z = mul (2, 2) <- y, y",
    ]

    # views are graph nodes too, so the chain back to w is unbroken
    wt = w.transpose(0, 1)
    out = x @ wt
    assert any(t is w for t in topo(out))
    assert trace(out) == [
        "0 x (2, 2)",
        "1 w (2, 2)",
        "2 transpose (2, 2) <- w",
        "3 matmul (2, 2) <- x, transpose",
    ]

    # reshaping a non-contiguous view goes through a contiguous copy
    assert [t._op or "leaf" for t in topo(wt.reshape(4))] == [
        "leaf",
        "transpose",
        "contiguous",
        "reshape",
    ]


def tests():
    run_tests(first)
    t1()
    t2()
    t3()
    t4()
    t5()
    t6()
    t7()
    t8()
    t9()
    t10()
    t11()
    t12()
    t13()
    print("✅ ok")


if __name__ == "__main__":
    tests()
