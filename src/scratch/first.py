def contiguous_strides(shape):
    res = []
    p = 1
    for s in reversed(shape):
        res.append(p)
        p *= s
    return tuple(reversed(res))


class Tensor:
    def __init__(self, data, shape, strides=None):
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

    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, strides={self.strides}, data={self.tolist()})"
        )


if __name__ == "__main__":
    from tests import tests

    tests()
