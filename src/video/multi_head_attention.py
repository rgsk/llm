import torch
from dropout import Dropout
from linear import Linear
from module import Module
from module_list import ModuleList
from residual_proj import ResidualProj
from softmax import softmax
from torch import Tensor


class Head(Module):
    def __init__(self, n_embed: int, head_size: int, dropout: float = 0.0):
        self.head_size = head_size
        self.query = Linear(n_embed, head_size, bias=False)
        self.key = Linear(n_embed, head_size, bias=False)
        self.value = Linear(n_embed, head_size, bias=False)
        self.attn_dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.shape
        q = self.query(x)  # [B, T, hs]
        k = self.key(x)  # [B, T, hs]
        v = self.value(x)  # [B, T, hs]
        scores = q @ k.transpose(-2, -1) * self.head_size**-0.5  # [B, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        w = softmax(scores, dim=-1)  # [B, T, T]
        w = self.attn_dropout(w)  # drop whole connections, not activations
        return w @ v  # [B, T, hs]


class MultiHeadAttention(Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.0):
        assert n_embed % n_head == 0, "n_embed must divide by n_head"
        head_size = n_embed // n_head
        self.heads = ModuleList(
            [Head(n_embed, head_size, dropout) for _ in range(n_head)]
        )
        self.proj = ResidualProj(n_embed, n_embed)  # was Linear
        self.resid_dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, E]
        return self.resid_dropout(self.proj(out))  # [B, T, E]


if __name__ == "__main__":
    import torch.nn.functional as F
    from torch import nn

    B, T, E, NH = 2, 8, 32, 4
    HS = E // NH
    x = torch.randn(B, T, E)

    # 1. one head == F.scaled_dot_product_attention(is_causal=True) on the same q,k,v
    h = Head(E, HS)
    q, k, v = h.query(x), h.key(x), h.value(x)
    assert (
        h(x) - F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ).abs().max() < 1e-6

    # 2. shapes and param names
    mha = MultiHeadAttention(E, NH)
    assert mha(x).shape == (B, T, E)
    names = [n for n, _ in mha.named_parameters()]
    assert names[:3] == [
        "heads.0.query.weight",
        "heads.0.key.weight",
        "heads.0.value.weight",
    ]
    assert names[-2:] == ["proj.weight", "proj.bias"]
    assert sum(p.numel() for p in mha.parameters()) == 3 * E * E + E * E + E

    # 3. causality: perturbing token t cannot change outputs before t
    out = mha(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        out2 = mha(x2)
        assert (out2[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"
        assert (
            out2[:, t] - out[:, t]
        ).abs().max() > 1e-3  # and it DOES change t itself
    print("causality holds for all t")

    # 4. attention weights: rows sum to 1, strictly-upper triangle exactly 0
    scores = q @ k.transpose(-2, -1) * HS**-0.5
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    w = softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    assert (w.sum(-1) - 1).abs().max() < 1e-6
    assert (w * ~causal).abs().max() == 0
    assert w[0, 0, 0] == 1.0  # token 0 can only attend to itself
    print("row 3 of the attention matrix:", [f"{p:.2f}" for p in w[0, 3].tolist()])

    # 5. against a torch mirror: same forward, same grads
    class RefHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(E, HS, bias=False)
            self.key = nn.Linear(E, HS, bias=False)
            self.value = nn.Linear(E, HS, bias=False)

        def forward(self, x):
            return F.scaled_dot_product_attention(
                self.query(x), self.key(x), self.value(x), is_causal=True
            )

    class RefMHA(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleList([RefHead() for _ in range(NH)])
            self.proj = nn.Linear(E, E)

        def forward(self, x):
            return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

    ref = RefMHA()
    assert mha.state_dict().keys() == ref.state_dict().keys()
    mha.load_state_dict(ref.state_dict())

    xm = x.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    om, orf = mha(xm), ref(xr)
    assert (om - orf).abs().max() < 1e-6
    om.square().sum().backward()
    orf.square().sum().backward()
    gd = max(
        (a.grad - b.grad).abs().max().item()
        for a, b in zip(mha.parameters(), ref.parameters())
    )
    assert gd < 1e-5 and (xm.grad - xr.grad).abs().max() < 1e-5

    print("ok")
