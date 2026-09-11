from first import Tensor, contiguous_strides, flatten, infer_shape, prod


def check_raises(exc, fn, msg=None):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(msg or f"expected {exc.__name__}, nothing raised")


def t1():
    a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    assert a.data == [1, 2, 3, 4, 5, 6]
    assert a.shape == (2, 3)
    assert a._offset((1, 2)) == 5
    assert a.data[a._offset((1, 2))] == 6


def t2():
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


def tests():
    t1()
    t2()
    t3()
    t4()
    t5()
    t6()
    print("✅ ok")


if __name__ == "__main__":
    tests()
