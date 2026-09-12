import first
from first import Tensor, contiguous_strides, flatten, infer_shape, prod
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


def tests():
    run_tests(first)
    t1()
    t2()
    t3()
    t4()
    t5()
    t6()
    t7()
    print("✅ ok")


if __name__ == "__main__":
    tests()
