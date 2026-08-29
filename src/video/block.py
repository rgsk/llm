from feed_forward import FeedForward
from layer_norm import LayerNorm
from module import Module
from multi_head_attention import MultiHeadAttention
from torch import Tensor


class Block(Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.0):
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, n_head, dropout)
        self.ffwd = FeedForward(n_embed, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))  # communicate
        x = x + self.ffwd(self.ln2(x))  # compute
        return x  # [B, T, E]


if __name__ == "__main__":
    import torch

    B, T, E, NH = 2, 8, 32, 4
    x = torch.randn(B, T, E)
    blk = Block(E, NH)

    # 1. shape preserved -- blocks are stackable
    assert blk(x).shape == (B, T, E)

    # 2. key layout matches main.py's Block
    keys = list(blk.state_dict())
    assert keys[:4] == ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"]
    assert any(k.startswith("attn.") for k in keys)
    assert keys[-4:] == [
        "ffwd.net.0.weight",
        "ffwd.net.0.bias",
        "ffwd.net.2.weight",
        "ffwd.net.2.bias",
    ]

    # 3. the residual path is an identity: zero both output projections
    #    and the block passes its input through untouched
    with torch.no_grad():
        blk.attn.proj.weight.zero_()
        blk.attn.proj.bias.zero_()
        blk.ffwd.net[2].weight.zero_()
        blk.ffwd.net[2].bias.zero_()
    assert (blk(x) - x).abs().max() == 0
    print("zeroed sublayers -> block is exactly identity")

    # 4. causality survives the block
    blk = Block(E, NH)
    out = blk(x)
    for t in range(1, T):
        x2 = x.clone()
        x2[:, t] += 10.0
        assert (blk(x2)[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"

    print("ok")
