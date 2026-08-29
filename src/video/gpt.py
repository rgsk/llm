import torch
from block import Block
from embedding import Embedding
from layer_norm import LayerNorm
from linear import Linear
from module import Module
from module_list import ModuleList
from residual_proj import ResidualProj
from torch import Tensor


class GPT(Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.0,
    ):
        self.block_size = block_size
        self.n_layer = n_layer
        self.token_embedding_table = Embedding(vocab_size, n_embed)
        self.position_embedding_table = Embedding(block_size, n_embed)
        self.blocks = ModuleList(
            [Block(n_embed, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(n_embed)
        self.lm_head = Linear(n_embed, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight  # tie, after init

    def _init_weights(self, module: Module) -> None:
        with torch.no_grad():
            if isinstance(module, Linear):
                std = 0.02
                # each block adds to the residual stream twice (attn + ffwd), so
                # over n_layer blocks the variance grows ~linearly with the number
                # of adds. shrink each contributing projection to keep it flat.
                if isinstance(module, ResidualProj):
                    std *= (2 * self.n_layer) ** -0.5
                module.weight.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, Embedding):
                module.weight.normal_(mean=0.0, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        _, T = idx.shape
        assert T <= self.block_size, f"sequence of {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits


if __name__ == "__main__":
    import math

    from cross_entropy import cross_entropy

    V, BS, E, NH, NL = 4096, 32, 64, 4, 3
    B, T = 4, 16
    torch.manual_seed(1337)

    m = GPT(V, BS, E, NH, NL)
    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))

    # 1. shape
    logits = m(idx)
    assert logits.shape == (B, T, V)

    # 2. weight tying: one object, two names
    assert m.lm_head.weight is m.token_embedding_table.weight
    sd = m.state_dict()
    assert "lm_head.weight" in sd and "token_embedding_table.weight" in sd
    n_tied = sum(p.numel() for p in m.parameters())
    n_untied = sum(p.numel() for _, p in m.named_parameters(remove_duplicate=False))
    assert n_untied - n_tied == V * E
    print(f"params: {n_tied / 1e6:.2f}M tied, {n_untied / 1e6:.2f}M if untied")

    # 3. init: every weight is N(0, 0.02), biases exactly 0
    assert abs(m.token_embedding_table.weight.std().item() - 0.02) < 0.002
    assert abs(m.blocks[0].ffwd.net[0].weight.std().item() - 0.02) < 0.002
    assert m.blocks[0].ffwd.net[0].bias.abs().max() == 0
    assert torch.equal(m.ln_f.weight, torch.ones(E))  # LayerNorm untouched

    # 4. THE check: loss at init is ln(vocab_size)
    loss = cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
    print(f"init loss {loss.item():.4f}   ln({V}) = {math.log(V):.4f}")
    assert abs(loss.item() - math.log(V)) < 0.1

    # 5. causality end to end
    out = m(idx)
    for t in range(1, T):
        idx2 = idx.clone()
        idx2[:, t] = (idx2[:, t] + 1) % V
        assert (m(idx2)[:, :t] - out[:, :t]).abs().max() < 1e-5, f"leak at t={t}"

    # 6. every parameter gets gradient (no dead weights)
    m.zero_grad()
    cross_entropy(m(idx).reshape(B * T, V), targets.reshape(B * T)).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} got no grad"
        if name == "position_embedding_table.weight":
            assert p.grad[:T].abs().max() > 0  # used rows
            assert p.grad[T:].abs().max() == 0  # rows past T never looked up
        else:
            assert p.grad.abs().max() > 0, f"{name} grad is all zero"
    print(f"all {len(list(m.named_parameters()))} params have gradient")

    # 7. block_size is enforced
    try:
        m(torch.randint(0, V, (1, BS + 1)))
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "block_size" in str(e)

    # 8. positional embeddings are what let the model see word order.
    #
    #    reverse the context (0..6 -> 6..0, keeping the last token fixed) with
    #    positions zeroed. every k_j and v_j then depends on token_j alone, not
    #    on where it sits, so a position's output is unchanged iff BOTH its own
    #    token and its prefix multiset are unchanged:
    #
    #      - positions 0-5 change: each pools its own prefix, and reversing
    #        changes what is in it. original pos2 pooled tokens {0,1,2};
    #        shuffled pos2 pools tokens {6,5,4}.
    #      - position 6 changes too, but for a different reason: its prefix
    #        {0..6} is the same multiset, only reordered -- what moved is its own
    #        token, which changes its query and its residual. (reverse just 0..5
    #        instead and pos6 comes out exactly unchanged.)
    #      - position 7 does NOT change (n_layer=1): its token is fixed and its
    #        mask row is all ones, so it pools all 8 tokens -- the same multiset,
    #        just reordered. a weighted sum does not care about term order.
    #      - position 7 DOES change once n_layer=2: block 2 pools the block-1
    #        outputs, and those changed at 0-6. order-blind pooling over a changed
    #        set gives a changed answer. so the causal mask is itself a positional
    #        signal, and depth is what makes it usable.
    #
    #    residual ~1e-7 below is float non-associativity: permuting the context
    #    permutes the summands in w @ v. it drops to ~1e-16 in float64.

    """
        explained simply
    
        0-6 change because, because the past that token at pos sees changes, original 2 saw, 0, 1, 2. shuffled 2 saw  6, 5, 4
    
        7th index doesn't change in n_layers=1 case, because it sees sum of those same 7 values just reordered, and it's own token value
    
        in case of n_layers=2, positions contains values computed from first layer, and each token got to attend tokens in past. for orginal, pos 2 was sum over 0, 1, 2, in shuffled pos 2 is sum over 6, 5, 4. now on these computed sum, second block acts, that's why shuffled and original produce different results.
    """

    one = GPT(V, BS, E, NH, n_layer=1)
    ctx = torch.randint(0, V, (1, 8))
    shuffled = ctx.clone()
    shuffled[:, :7] = ctx[
        :, torch.arange(7).flip(0)
    ]  # reverse context, keep last token
    assert not torch.equal(ctx, shuffled)

    with torch.no_grad():
        one.position_embedding_table.weight.zero_()
    assert (one(ctx)[:, -1] - one(shuffled)[:, -1]).abs().max() < 1e-6

    with torch.no_grad():
        one.position_embedding_table.weight.normal_(std=0.02)
    assert (one(ctx)[:, -1] - one(shuffled)[:, -1]).abs().max() > 1e-3

    # with n_layer >= 2 this no longer holds: the causal mask is itself a
    # positional signal, so depth recovers order without any embedding
    two = GPT(V, BS, E, NH, n_layer=2)
    with torch.no_grad():
        two.position_embedding_table.weight.zero_()
    assert (two(ctx)[:, -1] - two(shuffled)[:, -1]).abs().max() > 1e-3

    # 9. ResidualProj is scaled by 1/sqrt(2*n_layer); plain Linears are not
    exp = 0.02 * (2 * NL) ** -0.5
    assert abs(m.blocks[0].attn.proj.weight.std().item() - exp) < 0.1 * exp
    assert abs(m.blocks[0].ffwd.net[2].weight.std().item() - exp) < 0.1 * exp
    assert abs(m.blocks[0].ffwd.net[0].weight.std().item() - 0.02) < 0.002
    assert abs(m.blocks[0].attn.heads[0].query.weight.std().item() - 0.02) < 0.002
    assert abs(m.lm_head.weight.std().item() - 0.02) < 0.002

    # and the point of it: residual stream growth stops depending on depth
    def stream_growth(n_layer):
        torch.manual_seed(1337)
        mm = GPT(V, BS, E, NH, n_layer)
        ids = torch.randint(0, V, (4, 16))
        h = mm.token_embedding_table(ids) + mm.position_embedding_table(
            torch.arange(16)
        )
        start = h.std().item()
        for b in mm.blocks:
            h = b(h)
        return h.std().item() / start

    growths = [stream_growth(n) for n in (1, 3, 8, 24)]
    print("stream growth at depth 1/3/8/24:", [f"{g:.2f}x" for g in growths])
    # - stream growth at depth 1/3/8/24: ['1.38x', '1.34x', '1.40x', '1.39x']
    # without ResidualProj:
    # - stream growth at depth 1/3/8/24: ['1.67x', '2.39x', '4.09x', '6.93x']
    assert max(growths) - min(growths) < 0.2  # flat in depth

    print("ok")
