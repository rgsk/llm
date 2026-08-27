from collections.abc import Iterable, Iterator

from module import Module


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = ()):
        self._n = 0
        for m in modules:
            self.append(m)

    def append(self, m: Module) -> "ModuleList":
        setattr(self, str(self._n), m)  # attribute name "0", "1", ... -> "blocks.0.w"
        self._n += 1
        return self

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> Module:
        return getattr(self, str(range(self._n)[i]))  # range handles negative indices

    def __iter__(self) -> Iterator[Module]:
        return (self[i] for i in range(self._n))


if __name__ == "__main__":
    import torch
    from parameter import Parameter
    from torch import nn

    class MyLin(Module):
        def __init__(self, i, o):
            self.w = Parameter(torch.randn(o, i))
            self.b = Parameter(torch.zeros(o))

        def forward(self, x):
            return x @ self.w.T + self.b

    class RefLin(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.w = nn.Parameter(torch.randn(o, i))
            self.b = nn.Parameter(torch.zeros(o))

        def forward(self, x):
            return x @ self.w.T + self.b

    class Net(Module):
        def __init__(self):
            self.layers = ModuleList([MyLin(4, 8), MyLin(8, 8)])
            self.head = MyLin(8, 2)

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return self.head(x)

    class RefNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([RefLin(4, 8), RefLin(8, 8)])
            self.head = RefLin(8, 2)

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return self.head(x)

    net, ref = Net(), RefNet()
    assert [n for n, _ in net.named_parameters()] == [
        n for n, _ in ref.named_parameters()
    ]
    print([n for n, _ in net.named_parameters()])

    net.load_state_dict(ref.state_dict())
    x = torch.randn(5, 4)
    assert (net(x) - ref(x)).abs().max() == 0

    # list protocol
    assert len(net.layers) == 2
    assert net.layers[-1] is net.layers[1]
    assert [type(m).__name__ for m in net.layers] == ["MyLin", "MyLin"]

    # train/eval reaches through the container
    net.eval()
    assert not net.layers[0].training
    net.train()
    assert net.layers[1].training

    # append after construction keeps numbering going
    net.layers.append(MyLin(2, 2))
    assert len(net.layers) == 3
    assert [n for n, _ in net.layers.named_parameters()][-2:] == ["2.w", "2.b"]

    print("ok")
