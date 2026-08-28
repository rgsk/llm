import torch
from gpt import GPT
from multinomial import multinomial
from softmax import softmax
from torch import Tensor


def filter_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Keep the k highest logits, kill the rest before the softmax."""
    k = min(top_k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[:, [-1]]  # [B, 1]
    return logits.masked_fill(logits < kth, float("-inf"))


def filter_top_p(probs: Tensor, top_p: float) -> Tensor:
    """Nucleus: keep the smallest set of tokens whose probabilities sum to top_p."""
    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
    cumprobs = sorted_probs.cumsum(dim=-1)
    # subtract the token's own mass so the one that crosses the threshold is kept
    sorted_remove = (cumprobs - sorted_probs) > top_p
    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_idx, sorted_remove)
    probs = probs.masked_fill(remove, 0.0)
    return probs / probs.sum(dim=-1, keepdim=True)


def filter_min_p(probs: Tensor, min_p: float) -> Tensor:
    """Keep tokens at least min_p as likely as the top token. Scales with confidence:
    a peaked distribution keeps few, a flat one keeps many."""
    keep = probs >= min_p * probs.max(dim=-1, keepdim=True).values
    probs = probs.masked_fill(~keep, 0.0)
    return probs / probs.sum(dim=-1, keepdim=True)


@torch.no_grad()
def generate(
    model: GPT,
    idx: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,  # 1.0 no-op | 0.0 greedy | 0.8 recommended
    top_k: int | None = None,  # None no-op | 1 greedy
    top_p: float | None = None,  # 1.0 no-op | 0.0 greedy | 0.95 recommended
    min_p: float | None = None,  # 0.0 no-op | 1.0 greedy | 0.05-0.1 recommended
    generator: torch.Generator | None = None,
) -> Tensor:
    assert temperature >= 0.0
    assert top_k is None or top_k > 0
    assert top_p is None or 0.0 <= top_p <= 1.0
    assert min_p is None or 0.0 <= min_p <= 1.0

    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        cropped = idx[:, -model.block_size :]  # positions past block_size do not exist
        logits = model(cropped)[:, -1, :]  # [B, V] -- only the last position matters

        if temperature == 0.0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = filter_top_k(logits, top_k)
            probs = softmax(logits, dim=-1)
            if top_p is not None:
                probs = filter_top_p(probs, top_p)
            if min_p is not None:
                probs = filter_min_p(probs, min_p)
            nxt = multinomial(probs, generator)
        idx = torch.cat([idx, nxt], dim=1)

    if was_training:
        model.train()
    return idx


if __name__ == "__main__":
    torch.manual_seed(0)

    # 2. the filters
    lg = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    assert torch.isinf(filter_top_k(lg, 2)[0, 2:]).all()
    assert not torch.isinf(filter_top_k(lg, 2)[0, :2]).any()

    p = torch.tensor([[0.6, 0.3, 0.07, 0.03]])
    assert torch.allclose(
        filter_top_p(p, 0.9), torch.tensor([[2 / 3, 1 / 3, 0.0, 0.0]])
    )
    assert torch.allclose(filter_top_p(p, 0.0), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(filter_top_p(p, 1.0), p)
    assert filter_min_p(p, 0.1)[0, 3] == 0.0  # 0.03 < 0.1 * 0.6
    assert filter_min_p(p, 0.1)[0, 2] > 0.0  # 0.07 > 0.1 * 0.6
    assert torch.allclose(filter_min_p(p, 1.0), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    # 3. shape, and every greedy spelling agrees
    m = GPT(vocab_size=64, block_size=16, n_embed=32, n_head=4, n_layer=2)
    prompt = torch.randint(0, 64, (2, 5))
    greedy = generate(m, prompt, 10, temperature=0.0)
    assert greedy.shape == (2, 15)
    assert torch.equal(
        greedy, generate(m, prompt, 10, temperature=0.0)
    )  # deterministic
    assert torch.equal(greedy, generate(m, prompt, 10, top_k=1))
    assert torch.equal(greedy, generate(m, prompt, 10, top_p=0.0))
    assert torch.equal(greedy, generate(m, prompt, 10, min_p=1.0))

    # 4. seeded sampling is reproducible, different seeds differ
    a = generate(
        m, prompt, 10, temperature=0.8, generator=torch.Generator().manual_seed(0)
    )
    b = generate(
        m, prompt, 10, temperature=0.8, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(a, b)
    assert not torch.equal(
        a,
        generate(
            m, prompt, 10, temperature=0.8, generator=torch.Generator().manual_seed(1)
        ),
    )

    # 5. a prompt longer than block_size is cropped, not crashed
    long_prompt = torch.randint(0, 64, (1, 40))
    assert generate(m, long_prompt, 5, temperature=0.0).shape == (1, 45)

    # 6. mode is restored
    m.train()
    generate(m, prompt, 2, temperature=0.0)
    assert m.training

    print("ok")
