from module_list import ModuleList


class Sequential(ModuleList):
    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, x):
        for m in self:
            x = m(x)
        return x


if __name__ == "__main__":
    import torch
    from module import Module
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

    seq = Sequential(MyLin(4, 8), MyLin(8, 2))
    ref = nn.Sequential(RefLin(4, 8), RefLin(8, 2))

    assert seq.state_dict().keys() == ref.state_dict().keys()
    print(list(seq.state_dict().keys()))

    seq.load_state_dict(ref.state_dict())
    x = torch.randn(5, 4)
    assert (seq(x) - ref(x)).abs().max() == 0
    assert len(seq) == 2 and seq[-1] is seq[1]

    seq.eval()
    assert not seq[0].training
    seq.train()
    assert seq[0].training

    print("ok")
