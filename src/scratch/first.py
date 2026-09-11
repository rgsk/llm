from test_utils import selftest


def contiguous_strides(shape):
    res = []
    p = 1
    for s in reversed(shape):
        res.append(p)
        p *= s
    return tuple(reversed(res))


def infer_shape(nested):
    shape = []
    while isinstance(nested, list):
        shape.append(len(nested))
        if len(nested) == 0:
            break
        nested = nested[0]
    return tuple(shape)


def flatten(nested):
    if not isinstance(nested, list):
        return [nested]
    out = []
    for e in nested:
        out += flatten(e)
    return out


@selftest
def prod(xs):
    def test(self):
        assert self((2, 3)) == 6
        assert self((2, 3, 4)) == 24

    out = 1
    for x in xs:
        out *= x
    return out


class Tensor:
    def __init__(self, data, shape=None, strides=None):
        if shape is None:
            shape = infer_shape(data)
            data = flatten(data)
        self.data = data
        self.shape = shape
        self.strides = contiguous_strides(shape) if strides is None else strides

    def _offset(self, idx):
        assert len(idx) == len(self.shape), (
            f"got {len(idx)} indices for shape {self.shape}"
        )
        return sum(i * s for i, s in zip(idx, self.strides))

    def transpose(self, d0, d1):
        shape, strides = list(self.shape), list(self.strides)
        shape[d0], shape[d1] = shape[d1], shape[d0]
        strides[d0], strides[d1] = strides[d1], strides[d0]
        return Tensor(self.data, tuple(shape), tuple(strides))

    def tolist(self):
        def build(idx):
            if len(idx) == len(self.shape):
                return self.data[self._offset(idx)]
            d = len(idx)
            return [build(idx + (i,)) for i in range(self.shape[d])]

        return build(())

    @property
    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return self.strides == contiguous_strides(self.shape)

    def flat(self):
        return flatten(self.tolist())

    def contiguous(self):
        if self.is_contiguous():
            return self
        return Tensor(self.flat(), self.shape)

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, strides={self.strides}, data={self.tolist()})"
        )


if __name__ == "__main__":
    from tests import tests

    tests()
