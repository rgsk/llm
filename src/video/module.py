from collections.abc import Iterator

import torch
from parameter import Parameter
from torch import Tensor


class Module:
    training: bool = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def children(self) -> Iterator["Module"]:
        for v in self.__dict__.values():
            if isinstance(v, Module):
                yield v

    def _walk(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Every Parameter under this module, with torch-style dotted names."""
        for name, v in self.__dict__.items():
            path = f"{prefix}{name}"
            if isinstance(v, Parameter):
                yield path, v
            elif isinstance(v, Module):
                yield from v._walk(f"{path}.")

    def register_buffer(self, name: str, tensor: Tensor, persistent: bool = True):
        """State that travels with the module but is never trained. persistent=False
        keeps it out of state_dict -- right for anything __init__ can rebuild.

        The tensor is stored as-is, and the NAME is what marks it a buffer. A
        Tensor subclass would have been tidier, but torch.compile refuses to
        trace an arbitrary one: dynamo special-cases nn.Parameter and nn.Buffer
        and routes anything else through machinery that cannot handle it.
        """
        self.__dict__.setdefault("_buffer_names", {})[name] = persistent
        setattr(self, name, tensor.detach())

    def _walk_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor, bool]]:
        registered = self.__dict__.get("_buffer_names", {})
        for name, v in self.__dict__.items():
            path = f"{prefix}{name}"
            if name in registered:
                yield path, v, registered[name]
            elif isinstance(v, Module):
                yield from v._walk_buffers(f"{path}.")

    def named_buffers(self, prefix="", remove_duplicate=True, persistent_only=False):
        seen: set[int] = set()
        for path, b, persistent in self._walk_buffers(prefix):
            if persistent_only and not persistent:
                continue
            if remove_duplicate and id(b) in seen:
                continue
            seen.add(id(b))
            yield path, b

    def buffers(self) -> Iterator[Tensor]:
        for _, b in self.named_buffers():
            yield b

    def named_parameters(self, prefix="", remove_duplicate=True):
        seen: set[int] = set()
        for path, p in self._walk(prefix):
            if remove_duplicate and id(p) in seen:
                continue
            seen.add(id(p))
            yield path, p

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def to(self, *args, **kwargs) -> "Module":
        for p in self.parameters():
            p.data = p.data.to(*args, **kwargs)  # rebind in place: ties survive

        for b in self.buffers():
            # a dtype change must not touch a bool mask, or masked_fill breaks --
            # but a device move applies to everything. Same rule torch uses.
            moved = b.data.to(*args, **kwargs)
            b.data = moved if b.data.is_floating_point() else b.data.to(moved.device)

        return self

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> dict[str, Tensor]:
        # every name, including both halves of a tied pair -- matches torch
        sd = {n: p.data for n, p in self.named_parameters(remove_duplicate=False)}
        sd.update(
            {
                n: b.data
                for n, b in self.named_buffers(
                    remove_duplicate=False, persistent_only=True
                )
            }
        )
        return sd

    def load_state_dict(self, sd: dict[str, Tensor]) -> None:
        own = dict(self.named_parameters(remove_duplicate=False))
        own.update(
            {
                n: b
                for n, b in self.named_buffers(
                    remove_duplicate=False, persistent_only=True
                )
            }
        )
        assert not (missing := own.keys() - sd.keys()), f"missing: {sorted(missing)}"
        assert not (extra := sd.keys() - own.keys()), f"unexpected: {sorted(extra)}"
        seen: dict[int, str] = {}  # a tied Parameter arrives under several names
        with torch.no_grad():
            for name, p in own.items():
                assert p.shape == sd[name].shape, (
                    f"{name}: have {tuple(p.shape)}, got {tuple(sd[name].shape)}"
                )
                # this model ties them, so one copy_ overwrites the other -- a
                # checkpoint that disagrees would be half-discarded in silence
                if (first := seen.setdefault(id(p), name)) != name:
                    assert torch.equal(sd[first], sd[name]), (
                        f"{first} and {name} are tied here but differ in the checkpoint"
                    )
                p.copy_(sd[name])

    def apply(self, fn) -> "Module":
        for child in self.children():
            child.apply(fn)
        fn(self)  # children first, then self -- same order as torch
        return self


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from torch import nn

    class MyLin(Module):
        def __init__(self, i, o):
            self.w = Parameter(torch.randn(o, i))
            self.b = Parameter(torch.zeros(o))

        def forward(self, x):
            return x @ self.w.T + self.b

    class Net(Module):
        def __init__(self):
            self.a = MyLin(4, 8)
            self.b = MyLin(8, 2)

        def forward(self, x):
            return self.b(self.a(x))

    net = Net()
    assert [n for n, _ in net.named_parameters()] == ["a.w", "a.b", "b.w", "b.b"]
    assert net(torch.randn(5, 4)).shape == (5, 2)

    # training flag starts True and is not in __dict__ until set
    assert net.training and net.a.training
    assert "training" not in net.__dict__
    net.eval()
    assert "training" in net.__dict__
    assert not net.training and not net.a.training and not net.b.training
    net.train()
    assert net.training and net.a.training

    # to() reaches every parameter and keeps them Parameters
    net.to(torch.float64)
    assert all(p.dtype == torch.float64 for p in net.parameters())
    assert all(type(p) is Parameter for p in net.parameters())
    assert all(p.requires_grad and p.is_leaf for p in net.parameters())

    # zero_grad
    net(torch.randn(5, 4, dtype=torch.float64)).sum().backward()
    assert all(p.grad is not None for p in net.parameters())
    net.zero_grad()
    assert all(p.grad is None for p in net.parameters())

    # tied weight: one object, two names -- and to() keeps them tied
    class Tied(Module):
        def __init__(self):
            self.a = MyLin(4, 4)
            self.b = MyLin(4, 4)
            self.b.w = self.a.w

    t = Tied()
    assert len(list(t.parameters())) == 3  # deduped
    t.to(torch.float64)
    assert t.a.w is t.b.w and t.a.w.data_ptr() == t.b.w.data_ptr()

    # torch agrees: nn.Module.parameters() dedupes too
    ref = nn.Module()
    ref.x = nn.Parameter(torch.randn(3))
    ref.y = ref.x
    assert len(list(ref.parameters())) == 1

    class RefLin(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.w = nn.Parameter(torch.randn(o, i))
            self.b = nn.Parameter(torch.zeros(o))

        def forward(self, x):
            return x @ self.w.T + self.b

    class RefNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = RefLin(4, 8)
            self.b = RefLin(8, 2)

        def forward(self, x):
            return self.b(self.a(x))

    net, ref = Net(), RefNet()
    assert net.state_dict().keys() == ref.state_dict().keys()
    assert all(type(v) is torch.Tensor for v in net.state_dict().values())

    # torch -> ours: identical forward and grads
    net.load_state_dict(ref.state_dict())
    x = torch.randn(5, 4)
    out_m, out_r = net(x), ref(x)
    assert (out_m - out_r).abs().max() == 0
    out_m.sum().backward()
    out_r.sum().backward()
    assert all(
        (a.grad - b.grad).abs().max() == 0
        for a, b in zip(net.parameters(), ref.parameters())
    )

    # ours -> torch works too
    ref.load_state_dict(Net().state_dict())

    # round-trip through disk
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sd.pt"
        torch.save(net.state_dict(), path)
        fresh = Net()
        fresh.load_state_dict(torch.load(path))
        assert (fresh(x) - out_m).abs().max() == 0

    # tied: deduped in named_parameters, both names in state_dict
    class Tied(Module):
        def __init__(self):
            self.a = MyLin(4, 4)
            self.b = MyLin(4, 4)
            self.b.w = self.a.w

    class RefTied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = RefLin(4, 4)
            self.b = RefLin(4, 4)
            self.b.w = self.a.w

    t, rt = Tied(), RefTied()
    assert [n for n, _ in t.named_parameters()] == [n for n, _ in rt.named_parameters()]
    assert t.state_dict().keys() == rt.state_dict().keys()
    t.load_state_dict(rt.state_dict())
    assert t.a.w is t.b.w  # still one object after loading

    # a checkpoint whose two tied names disagree cannot be half-loaded in silence
    conflict = rt.state_dict()
    conflict["b.w"] = torch.zeros_like(conflict["b.w"])
    try:
        t.load_state_dict(conflict)
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "tied here but differ" in str(e)

    # bad loads are caught
    try:
        net.load_state_dict({k: v for k, v in ref.state_dict().items() if k != "a.w"})
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "missing" in str(e)

    # loading must not alias the source dict
    src, dst = Net(), Net()
    sd = src.state_dict()
    dst.load_state_dict(sd)
    with torch.no_grad():
        for p in dst.parameters():
            p.add_(1.0)
    assert (src.a.w - dst.a.w).abs().max() > 0

    # loading must not change the model's dtype/device
    m64 = Net().to(torch.float64)
    m64.load_state_dict(Net().state_dict())  # a float32 checkpoint
    assert all(p.dtype == torch.float64 for p in m64.parameters())

    # an untied model must stay untied after loading a tied checkpoint
    class Two(Module):
        def __init__(self, tie: bool):
            self.a = MyLin(4, 4)
            self.c = MyLin(4, 4)
            if tie:
                self.c.w = self.a.w

    untied = Two(tie=False)
    untied.load_state_dict(Two(tie=True).state_dict())
    assert untied.a.w.data_ptr() != untied.c.w.data_ptr()

    # ---- buffers ----
    class WithBuf(Module):
        def __init__(self):
            self.lin = MyLin(4, 4)
            self.register_buffer("tril", torch.ones(4, 4, dtype=torch.bool).tril())
            self.register_buffer("running", torch.zeros(4))
            self.register_buffer("scratch", torch.zeros(4), persistent=False)

        def forward(self, x):
            return self.lin(x).masked_fill(~self.tril, 0.0) + self.running

    class RefBuf(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = RefLin(4, 4)
            self.register_buffer("tril", torch.ones(4, 4, dtype=torch.bool).tril())
            self.register_buffer("running", torch.zeros(4))
            self.register_buffer("scratch", torch.zeros(4), persistent=False)

    wb, rb = WithBuf(), RefBuf()

    # buffers are state, not parameters: the optimizer never sees them
    assert [n for n, _ in wb.named_parameters()] == ["lin.w", "lin.b"]
    assert [n for n, _ in wb.named_buffers()] == ["tril", "running", "scratch"]
    assert all(not b.requires_grad for b in wb.buffers())

    # ... but they ARE in state_dict, except the non-persistent one -- same as torch
    assert wb.state_dict().keys() == rb.state_dict().keys()
    assert "scratch" not in wb.state_dict()
    assert "tril" in wb.state_dict() and "running" in wb.state_dict()

    # a checkpoint round trip carries them
    wb2 = WithBuf()
    with torch.no_grad():
        wb.running.fill_(7.0)
    wb2.load_state_dict(wb.state_dict())
    assert (wb2.running == 7.0).all()

    # .to(dtype) must NOT cast a bool mask, or masked_fill stops working;
    # .to(device) moves everything. torch does exactly this
    wb.to(torch.float64)
    rb.to(torch.float64)
    assert wb.tril.dtype == rb.tril.dtype == torch.bool
    assert wb.running.dtype == rb.running.dtype == torch.float64
    assert wb.lin.w.dtype == torch.float64
    assert wb(torch.randn(4, 4, dtype=torch.float64)).shape == (4, 4)

    # a non-persistent buffer still moves and still works -- it is just not saved
    assert wb.scratch.dtype == torch.float64

    # buffers take no gradient even when they take part in the forward
    wb.to(torch.float32)
    out = wb(torch.randn(4, 4))
    out.sum().backward()
    assert all(b.grad is None for b in wb.buffers())

    print("ok")
