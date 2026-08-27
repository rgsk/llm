from linear import Linear
from module import Module
from relu import ReLU
from residual_proj import ResidualProj
from sequential import Sequential
from torch import Tensor


class FeedForward(Module):
    def __init__(self, n_embed: int):
        self.net = Sequential(
            Linear(n_embed, 4 * n_embed),
            ReLU(),
            ResidualProj(4 * n_embed, n_embed),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # [B, T, E]


if __name__ == "__main__":
    import torch
    from torch import nn

    B, T, E = 2, 8, 32
    x = torch.randn(B, T, E)
    ff = FeedForward(E)

    # 1. shape is preserved; the 4x expansion is internal
    assert ff(x).shape == (B, T, E)
    assert ff.net[0].weight.shape == (4 * E, E)
    assert ff.net[2].weight.shape == (E, 4 * E)

    # 2. keys match main.py's FeedForward, dropout or not
    #    (ReLU at index 1 and Dropout at index 3 have no parameters)
    assert list(ff.state_dict()) == [
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
    ]
    assert sum(p.numel() for p in ff.parameters()) == 8 * E**2 + 5 * E

    # 3. matches a torch mirror, forward and grads -- exactly
    ref = nn.Sequential(nn.Linear(E, 4 * E), nn.ReLU(), nn.Linear(4 * E, E))
    assert ff.net.state_dict().keys() == ref.state_dict().keys()
    ff.net.load_state_dict(ref.state_dict())

    xm = x.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    om, orf = ff(xm), ref(xr)
    assert (om - orf).abs().max() == 0
    om.square().sum().backward()
    orf.square().sum().backward()
    gd = max(
        (a.grad - b.grad).abs().max().item()
        for a, b in zip(ff.parameters(), ref.parameters())
    )
    assert gd == 0 and (xm.grad - xr.grad).abs().max() == 0

    # 4. position-wise: perturbing token 3 changes only token 3
    y = ff(x)
    x2 = x.clone()
    x2[:, 3] += 10.0
    y2 = ff(x2)
    assert (y2[:, :3] - y[:, :3]).abs().max() == 0
    assert (y2[:, 4:] - y[:, 4:]).abs().max() == 0
    assert (y2[:, 3] - y[:, 3]).abs().max() > 1e-3

    # 5. one token alone == that token inside the batch
    #    (not bitwise: torch picks a different matmul blocking for [1,1,E])
    assert (ff(x[:1, 3:4]) - y[:1, 3:4]).abs().max() < 1e-6

    print("ok")
