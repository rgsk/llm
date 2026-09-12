import first
import torch
from first import (
    Tensor,
    broadcast_shape,
    contiguous_strides,
    flatten,
    infer_shape,
    prod,
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
    print("✅ ok")


if __name__ == "__main__":
    tests()
