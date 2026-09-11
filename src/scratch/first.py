from test_utils import check_raises, selftest


def contiguous_strides(shape):
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


def infer_shape(nested):
    shape = []
    while isinstance(nested, list):
        shape.append(len(nested))
        if len(nested) == 0:
            break
        nested = nested[0]
    return tuple(shape)


@selftest(infer_shape)
def _(fn):
    assert fn(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ) == (2, 3)


def flatten(nested):
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


def prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


@selftest(prod)
def _(fn):
    assert fn((2, 3)) == 6
    assert fn((2, 3, 4)) == 24


class Tensor:
    def __init__(self, data, shape=None, strides=None):
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

    def _offset(self, idx):
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

    def transpose(self, d0, d1):
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

    def tolist(self):
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
    def numel(self):
        return prod(self.shape)

    @selftest(numel.fget)
    def _(fn):
        assert fn(Tensor([1, 2, 3, 4, 5, 6], (2, 3))) == 6

    def is_contiguous(self):
        return self.strides == contiguous_strides(self.shape)

    @selftest(is_contiguous)
    def _(fn):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        assert fn(a)
        assert not fn(a.transpose(0, 1))

    def flat(self):
        return flatten(self.tolist())

    @selftest(flat)
    def _(fn):
        t = Tensor([1, 2, 3, 4, 5, 6], (2, 3)).transpose(0, 1)
        assert t.data == [1, 2, 3, 4, 5, 6]  # storage order
        assert fn(t) == [1, 4, 2, 5, 3, 6]  # logical order

    def contiguous(self):
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

    def __repr__(self):
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
