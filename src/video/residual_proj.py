from linear import Linear


class ResidualProj(Linear):
    """Linear whose output is added to the residual stream; gets 1/sqrt(2*n_layer) init.

    Purely a marker -- forward is Linear's. GPT._init_weights looks for the type.
    Subclasses whatever backend.py exported, so it follows the switch.
    """


if __name__ == "__main__":
    import torch
    from torch import nn

    from backend import USE_TORCH
    from linear import OurLinear

    # 1. same keys and forward as a plain Linear -- the subclass adds nothing
    rp = ResidualProj(6, 3)
    ref = nn.Linear(6, 3)
    assert rp.state_dict().keys() == ref.state_dict().keys()
    ref.load_state_dict(rp.state_dict())
    x = torch.randn(4, 6)
    assert (rp(x) - ref(x)).abs().max() == 0

    # 2. it tracks the backend (can't be asserted from linear.py's __main__)
    assert issubclass(ResidualProj, Linear)
    assert isinstance(rp, Linear)
    assert (Linear is nn.Linear) == USE_TORCH
    assert (Linear is OurLinear) != USE_TORCH

    # 3. isinstance still separates it from a plain Linear -- _init_weights
    #    depends on that
    assert isinstance(rp, Linear) and not isinstance(Linear(6, 3), ResidualProj)

    print(f"ResidualProj <- {Linear.__module__}.{Linear.__name__}")
    print("ok")
