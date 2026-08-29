import torch
from torch import Tensor


class Buffer(Tensor):
    """A Tensor that belongs to the module's state but is never trained.

    Parameter says "the optimizer owns this". Buffer says "I am state, carry me
    along -- move me with .to(), and save me with the weights -- but no gradient
    and no update". A causal mask, or the running mean in a BatchNorm.

    persistent=False keeps it out of state_dict: use that for anything the
    module can rebuild from its own config, so it never appears in a checkpoint
    and never has to match one.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, data: Tensor, persistent: bool = True) -> "Buffer":
        b = data.detach().as_subclass(cls)
        b.persistent = persistent
        return b


if __name__ == "__main__":
    from parameter import Parameter
    from torch import nn

    t = torch.ones(3, 3).tril()
    b = Buffer(t)

    # 1. it is a Tensor, and it is not a Parameter
    assert isinstance(b, Tensor)
    assert not isinstance(b, Parameter)
    assert torch.equal(b, t)

    # 2. no gradient, ever -- that is the whole difference from Parameter
    assert not b.requires_grad
    assert Parameter(t).requires_grad
    assert b.grad is None

    # 3. it does not infect the results of arithmetic (same fix as Parameter)
    assert type(b + 1) is Tensor
    assert type(b @ b) is Tensor

    # 4. torch agrees on both counts: nn.Buffer is a Tensor, not a Parameter
    ref = nn.Buffer(t)
    assert isinstance(ref, Tensor) and not isinstance(ref, nn.Parameter)
    assert not ref.requires_grad

    # 5. persistent is a plain attribute, and it survives .to()
    assert b.persistent is True
    assert Buffer(t, persistent=False).persistent is False
    b.data = b.data.to(torch.float64)  # what Module.to does
    assert b.persistent is True and b.dtype == torch.float64

    # 6. it detaches, like Parameter -- a buffer built from a live graph does
    #    not drag that graph along
    live = (torch.randn(3, requires_grad=True) * 2).sum().expand(3)
    assert live.grad_fn is not None
    assert Buffer(live).grad_fn is None

    print("ok")
