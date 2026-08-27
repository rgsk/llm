import torch
from module import Module
from parameter import Parameter
from torch import Tensor


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim).normal_())

    def forward(self, idx: Tensor) -> Tensor:
        return self.weight[idx]  # [...] of ids -> [..., embedding_dim]


if __name__ == "__main__":
    from torch import nn

    V, E = 10, 4

    # 1. same init, draw for draw
    torch.manual_seed(0)
    mine = Embedding(V, E)
    torch.manual_seed(0)
    ref = nn.Embedding(V, E)
    assert torch.equal(mine.weight, ref.weight)
    assert mine.state_dict().keys() == ref.state_dict().keys()

    # 2. forward: 1D positions [T] and 2D token ids [B, T]
    mine.load_state_dict(ref.state_dict())
    for idx in [torch.arange(3), torch.randint(0, V, (2, 3))]:
        assert (mine(idx) - ref(idx)).abs().max() == 0
        assert mine(idx).shape == (*idx.shape, E)

    # 3. a lookup really is a row of the table
    assert torch.equal(mine(torch.tensor(7)), mine.weight[7])

    # 4. backward scatter-ADDS on repeated ids
    idx = torch.tensor([3, 3, 3, 5])
    mine(idx).sum().backward()
    ref(idx).sum().backward()
    assert (mine.weight.grad - ref.weight.grad).abs().max() == 0
    assert mine.weight.grad[3].tolist() == [3.0] * E  # seen 3x -> 3, not 1
    assert mine.weight.grad[5].tolist() == [1.0] * E
    assert mine.weight.grad[0].tolist() == [0.0] * E  # never looked up -> no grad

    # 5. output is a plain Tensor, not a Parameter
    assert type(mine(torch.tensor([1]))) is torch.Tensor

    print("ok")
