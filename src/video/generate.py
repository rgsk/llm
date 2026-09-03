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
    use_cache: bool = False,  # needs attention="fused" or "sdpa"
) -> Tensor:
    assert temperature >= 0.0
    assert top_k is None or top_k > 0
    assert top_p is None or 0.0 <= top_p <= 1.0
    assert min_p is None or 0.0 <= min_p <= 1.0
    if use_cache:
        # the uncached loop crops to the last block_size tokens and recomputes,
        # so it generates forever. the cached one cannot: cropping would throw
        # away keys it can never rebuild, and positions come from a table with
        # exactly block_size rows. so the whole run has to fit.
        fed = idx.size(1) + max_new_tokens - 1  # the last token is never fed back
        assert fed <= model.block_size, (
            f"use_cache needs prompt+max_new_tokens-1 ({idx.size(1)}+"
            f"{max_new_tokens}-1={fed}) <= block_size ({model.block_size}). "
            f"Use use_cache=False to crop+recompute."
        )

    was_training = model.training
    model.eval()
    kv_caches = None
    for _ in range(max_new_tokens):
        if use_cache:
            # first pass prefills the whole prompt, every one after it feeds a
            # single token -- the cache holds everything before it
            step = idx if kv_caches is None else idx[:, -1:]
            logits, kv_caches = model(step, kv_caches, use_cache=True)
        else:
            # positions past block_size do not exist
            logits = model(idx[:, -model.block_size :])
        logits = logits[:, -1, :]  # [B, V] -- only the last position matters

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

    # 7. THE test: the cache changes the cost, not the output. greedy is
    #    deterministic, so the two paths must produce identical token ids
    cm = GPT(
        vocab_size=64, block_size=16, n_embed=32, n_head=4, n_layer=2, attention="fused"
    )
    p5 = torch.randint(0, 64, (2, 5))
    assert torch.equal(
        generate(cm, p5, 10, temperature=0.0),
        generate(cm, p5, 10, temperature=0.0, use_cache=True),
    )

    # and sampling agrees too, given the same generator: the same probabilities
    # go into multinomial, so the same draws come out
    assert torch.equal(
        generate(
            cm, p5, 10, temperature=0.8, generator=torch.Generator().manual_seed(0)
        ),
        generate(
            cm,
            p5,
            10,
            temperature=0.8,
            use_cache=True,
            generator=torch.Generator().manual_seed(0),
        ),
    )
    print("cached generate matches uncached, greedy and sampled")

    # 8. the ceiling is real: cropping is not available to the cached path
    generate(cm, p5, 12, temperature=0.0, use_cache=True)  # 5 + 12 - 1 == 16, fits
    try:
        generate(cm, p5, 13, temperature=0.0, use_cache=True)
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "block_size" in str(e)
    # the uncached path still runs well past block_size, by forgetting
    assert generate(cm, p5, 40, temperature=0.0).shape == (2, 45)

    # 9. the two ways a cache can be stored, end to end. Same model, same
    #    weights, same tokens -- the only difference is whether each layer's
    #    KVCache preallocates [B, n_kv, block_size, hs] and writes at a cursor,
    #    or torch.cats a whole new cache into existence every step.
    #
    #    kv_cache.py test 8 times that copy on its own and reports 26x. Read
    #    the table below before believing it
    import time

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    big = {
        "vocab_size": 4096,
        "block_size": 512,
        "n_embed": 512,
        "n_head": 8,
        "n_layer": 8,
    }
    small = {
        "vocab_size": 256,
        "block_size": 128,
        "n_embed": 128,
        "n_head": 4,
        "n_layer": 2,
    }
    gm = GPT(
        **(big if dev == "cuda" else small),
        attention="gqa",
        n_kv_head=2,
        norm="rms",
        ffn="gated",
    ).to(dev)
    GN = gm.block_size  # prompt is 1 token, so this is the most the cache holds

    def cache_mode(preallocate: bool) -> None:
        """block_size is what sizes the buffer; None falls back to growing."""
        for blk in gm.blocks:
            blk.attn.block_size = gm.block_size if preallocate else None

    def timed_gen(batch: int) -> tuple[float, Tensor]:
        start = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        generate(gm, start, 4, temperature=0.0, use_cache=True)  # warm up
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = generate(gm, start, GN, temperature=0.0, use_cache=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, out

    print(f"\n  {GN} tokens, {gm.n_layer}L gqa n_kv_head=2, {dev}")
    print(f"  {'batch':>5}  {'cache':>9}  {'grow':>7}  {'buffer':>7}  {'gain':>5}")
    for batch in (1, 32) if dev == "cuda" else (1,):
        cache_mode(False)
        grow_s, grow_out = timed_gen(batch)
        cache_mode(True)
        buf_s, buf_out = timed_gen(batch)
        assert torch.equal(grow_out, buf_out)  # identical tokens, greedy
        held = 2 * gm.n_layer * batch * 2 * GN * (gm.blocks[0].attn.head_size) * 4
        print(
            f"  {batch:>5}  {held / 2**20:>6.0f} MB  {grow_s:>6.2f}s  "
            f"{buf_s:>6.2f}s  {grow_s / buf_s:>4.2f}x"
        )
    #    no assert on the timings, because at batch 1 there is nothing to assert
    """
    This is the honest number, and it is 1.01x at batch 1 -- rising to about
    1.15x at batch 64, where the cache is 256 MB. The 26x in kv_cache.py is
    real and it is also irrelevant here: that test times the copy alone, and a
    decode step is 8 layers of matmuls, an lm_head over the whole vocab, and
    sampling, next to which copying 4 MB is nothing. Isolate any component of a
    step and it will look like the bottleneck.

    So preallocation is not a speed optimisation at this scale. Do it anyway,
    for the two things it does buy: the allocator stops churning a fresh
    multi-megabyte tensor per layer per token, and the cache gains a cursor --
    which is what a ring buffer, a sliding window, and paged attention are all
    built on. It earns its place as the shape the cache has to have, not as a
    number on a chart. The number arrives later, with batch and context.
    """

    # 10. and what the cache itself was for -- measured on the real model, not
    #     a toy. the uncached loop re-reads the whole prefix every step, so its
    #     cost grows with what it has already written; the cached one does
    #     constant work per token. the gap widens with the length of the sample

    from checkpoint import latest_ckpt, load_checkpoint

    try:
        # not just the newest: a run that died at step 0 sits at the init loss,
        # ln(4096) = 8.3, and would win on timestamp alone. anything that
        # actually trained is far below 6
        ckpt = latest_ckpt("big", max_val_loss=6.0)
    except FileNotFoundError as e:
        print(f"\nskipping the benchmark: {e}")
        print("ok")
        raise SystemExit

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    trained, meta = load_checkpoint(ckpt, dev)
    n_params = sum(p.numel() for p in trained.parameters())
    print(f"\n{ckpt.name}   {n_params / 1e6:.1f}M params   val {meta['val_loss']:.3f}")

    # the most the cached path can do without cropping: it feeds prompt + N - 1
    # tokens through the model (the last one sampled is never fed back), and
    # that has to fit block_size -- so N tops out at block_size - prompt + 1
    start = torch.zeros(1, 1, dtype=torch.long, device=dev)
    N = trained.block_size - start.size(1) + 1

    def timed(use_cache: bool) -> tuple[float, Tensor]:
        gen = torch.Generator(device=dev).manual_seed(0)
        generate(trained, start, 4, use_cache=use_cache)  # warm up
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = generate(
            trained, start, N, temperature=0.8, generator=gen, use_cache=use_cache
        )
        if dev == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, out

    slow, out_slow = timed(False)
    fast, out_fast = timed(True)
    assert torch.equal(out_slow, out_fast)  # same tokens, either way
    print(
        f"{dev}  {N} tokens -- recompute {slow:.2f}s ({slow / N * 1e3:.1f} ms/tok)   "
        f"cached {fast:.2f}s ({fast / N * 1e3:.1f} ms/tok)   {slow / fast:.2f}x"
    )

    print("ok")
