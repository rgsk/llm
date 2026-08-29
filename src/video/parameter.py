import torch
from torch import nn

Parameter = nn.Parameter

# class Parameter(Tensor):
#     """A Tensor marked as trainable. Modules collect these; the optimizer updates them."""

#     __torch_function__ = torch._C._disabled_torch_function_impl

#     @staticmethod
#     def __new__(cls, data: Tensor) -> "Parameter":
#         return data.detach().as_subclass(cls).requires_grad_(True)


if __name__ == "__main__":
    p = Parameter(torch.randn(3))

    assert p.requires_grad
    assert p.is_leaf  # or .grad never gets populated
    assert type(p * 2) is torch.Tensor  # subclass must not leak into activations
    print("ok")
