import torch
from generate import filter_min_p, filter_top_k, filter_top_p
from gpt import GPT
from kv_cache import KVCache
from multinomial import multinomial
from softmax import softmax
from torch import Tensor


def rollback(caches: list, pos: int) -> list:
    """Truncate every layer's cache to `pos` positions.

    Two shapes turn up because the older attention files hand back the plain
    (k, v) tuple and only the newer ones hand back a KVCache. For a tuple this
    re-slices; for a buffer it moves the cursor. Same answer, and the second one
    is why the cursor exists.
    """
    out = []
    for c in caches:
        if isinstance(c, KVCache):
            c.rollback(pos)
            out.append(c)
        else:
            out.append((c[0][:, :, :pos], c[1][:, :, :pos]))
    return out


def sampling_dist(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    min_p: float | None,
) -> Tensor:
    """[B, V] logits -> the distribution generate() would actually sample from.

    Both models get shaped by this, identically, and that matters: the accept
    rule compares p(x) against q(x), so if the filters differ the two numbers
    are not about the same distribution and the guarantee evaporates. What
    speculative decoding reproduces is the target *as configured*, not the
    target's raw softmax.

    temperature 0 becomes a one-hot, which makes greedy fall out of the same
    code path: p(x)/q(x) is 1 when the two argmaxes agree and 0 when they do
    not, so every match is accepted and every mismatch is replaced by the
    target's own token.
    """
    if temperature == 0.0:
        onehot = torch.zeros_like(logits)
        return onehot.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    logits = logits / temperature
    if top_k is not None:
        logits = filter_top_k(logits, top_k)
    probs = softmax(logits, dim=-1)
    if top_p is not None:
        probs = filter_top_p(probs, top_p)
    if min_p is not None:
        probs = filter_min_p(probs, min_p)
    return probs


@torch.no_grad()
def speculative_generate(
    target: GPT,
    draft: GPT,
    idx: Tensor,
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    generator: torch.Generator | None = None,
    stats: dict | None = None,
) -> Tensor:
    """Sample from `target`, using `draft` to guess ahead. Same distribution, fewer
    target forwards.

    The whole trick rests on one asymmetry: scoring k tokens costs a target
    forward almost exactly what scoring one costs, because the expensive part is
    reading the weights, not the arithmetic. So let something cheap guess k
    tokens, then spend one target forward checking all of them at once.

    Each round:

      1. the draft samples x_0 .. x_{k-1} autoregressively, recording the
         distribution q_j it drew each one from
      2. the target runs ONCE over [everything new] + [x_0 .. x_{k-1}], which
         yields p_0 .. p_k -- its distribution at each drafted position, plus
         one past the end
      3. each x_j is accepted with probability min(1, p_j(x_j) / q_j(x_j)). The
         first rejection ends the round, and its token is redrawn from the
         residual max(0, p_j - q_j), renormalised
      4. if all k survive, p_k hands over a free extra token -- the target
         forward was going to produce it anyway

    Step 3 is the part that makes this lossless rather than approximate. The
    draft over-proposes some tokens and under-proposes others; accepting with
    ratio p/q removes exactly the excess, and the residual distribution puts
    back exactly the deficit. The tokens that come out are distributed as if
    the target had been sampled directly -- proved in Leviathan et al. 2023,
    and test 2 below checks it empirically rather than taking it on faith.

    Batch is 1. Rows would reject at different positions, so a batched version
    needs either ragged caches or syncing every row to the unluckiest one, and
    neither belongs in the file that explains the idea.
    """
    assert idx.size(0) == 1, "speculative decoding here is batch 1; see the docstring"
    assert k >= 1
    V_t = target.lm_head.weight.size(0)
    V_d = draft.lm_head.weight.size(0)
    assert V_t == V_d, f"models must share a vocabulary: {V_t} vs {V_d}"
    filters = (temperature, top_k, top_p, min_p)

    modes = target.training, draft.training
    target.eval()
    draft.eval()

    stop = idx.size(1) + max_new_tokens
    limit = min(target.block_size, draft.block_size)
    assert stop - 1 <= limit, (
        f"prompt + max_new_tokens - 1 ({stop - 1}) must fit both models' "
        f"block_size ({limit})"
    )

    seq = idx
    t_caches = d_caches = None
    t_pos = d_pos = 0  # how much of seq each model's cache already holds
    tested = accepted = rounds = 0

    while seq.size(1) < stop:
        n = seq.size(1)

        # 1. the draft runs ahead. It is fed whatever it has not seen yet, then
        #    its own guesses, one at a time -- this is ordinary autoregressive
        #    decoding, just done by the small model
        # a round writes n + k entries into the cache, so near the end of the
        # context there may be room for fewer than k guesses. Drafting past
        # block_size is not a smaller speedup, it is an assert inside the model
        k_round = min(k, limit - n)
        qs, xs = [], []
        step = seq[:, d_pos:]
        for i in range(k_round):
            logits, d_caches = draft(step, d_caches, use_cache=True)
            d_pos += step.size(1)
            q = sampling_dist(logits[:, -1], *filters)
            x = multinomial(q, generator)  # [1, 1]
            qs.append(q)
            xs.append(x)
            step = x
        # the last guess is deliberately never fed back: nothing would read it,
        # and it is a whole draft forward saved per round
        drafted = torch.cat(xs, dim=1) if xs else seq[:, :0]  # [1, k_round]

        # 2. one target forward covers all k. Feeding the tokens the target has
        #    not seen along with the guesses keeps this a single call
        t_in = torch.cat([seq[:, t_pos:], drafted], dim=1)
        logits, t_caches = target(t_in, t_caches, use_cache=True)
        t_pos += t_in.size(1)
        # the last k+1 positions are the ones being judged: p_j is the target's
        # distribution given everything up to (but not including) x_j
        ps = [
            sampling_dist(logits[:, j - (k_round + 1)], *filters)
            for j in range(k_round + 1)
        ]

        # 3. accept while the target agrees, stop at the first rejection
        n_acc = 0
        for j in range(k_round):
            x = drafted[0, j]
            ratio = ps[j][0, x] / qs[j][0, x].clamp(min=1e-20)
            u = torch.rand((), device=seq.device, generator=generator)
            if u >= ratio:
                break
            n_acc += 1

        if n_acc == k_round:
            # 4. every guess survived, so p_k is a token nobody had to pay for
            nxt = multinomial(ps[k_round], generator)
        else:
            # the rejected position is redrawn from what the target wanted and
            # the draft did not offer. clamp then renormalise: the mass that
            # survives is exactly the target's excess over the draft
            resid = (ps[n_acc] - qs[n_acc]).clamp(min=0)
            total = resid.sum(-1, keepdim=True)
            resid = torch.where(total > 0, resid / total.clamp(min=1e-20), ps[n_acc])
            nxt = multinomial(resid, generator)

        seq = torch.cat([seq, drafted[:, :n_acc], nxt], dim=1)
        # only the guesses the target actually judged count towards the rate:
        # after the first rejection the rest are dropped untested, so dividing
        # by k_round would quietly report a lower acceptance at larger k
        tested += n_acc if n_acc == k_round else n_acc + 1
        accepted += n_acc
        rounds += 1

        # 5. un-write the guesses that did not survive. The target holds n + k
        #    and the draft n + k - 1; both are valid only up to n + n_acc
        t_caches = rollback(t_caches, n + n_acc)
        t_pos = n + n_acc
        d_pos = n + min(n_acc, max(k_round - 1, 0))
        d_caches = rollback(d_caches, d_pos)

    if stats is not None:
        stats.update(
            tested=tested,
            accepted=accepted,
            rounds=rounds,
            accept_rate=accepted / max(tested, 1),
            tokens_per_target_pass=(seq.size(1) - idx.size(1)) / rounds,
        )
    if modes[0]:
        target.train()
    if modes[1]:
        draft.train()
    return seq[:, :stop]  # a round can overshoot by up to k, so trim


if __name__ == "__main__":
    import time
    from dataclasses import asdict

    from checkpoint import CKPT_DIR, latest_ckpt, load_checkpoint
    from evaluate import tokens_per_pass
    from generate import generate
    from gpt_config import GPTConfig

    torch.manual_seed(0)
    V, T = 64, 64
    small = GPTConfig(
        vocab_size=V, block_size=T, n_embed=32, n_head=4, n_layer=2, attention="sdpa"
    )
    tiny = GPTConfig(
        vocab_size=V, block_size=T, n_embed=16, n_head=2, n_layer=1, attention="sdpa"
    )
    tgt, dft = GPT(**asdict(small)), GPT(**asdict(tiny))
    prompt = torch.randint(0, V, (1, 5))
    # sharpen the target's output. Two untrained models are both near-uniform,
    # which would make every test below pass for the wrong reason -- acceptance
    # would be high because the distributions are indistinguishable, not because
    # the draft is good. Scaling the final norm gives the target opinions, so
    # the draft gets rejected constantly and the residual branch does real work
    with torch.no_grad():
        tgt.ln_f.weight.mul_(8)

    # 1. THE claim, in its exactly-checkable form. Greedy is deterministic, so
    #    "same distribution" becomes "same token ids" -- and they must match to
    #    the last token even though the draft here is random and gets rejected
    #    constantly. Every rejection is a cache rollback, so this also proves
    #    the rollback: get it wrong and the sequences diverge immediately
    for k in (1, 2, 4, 8):
        spec = speculative_generate(tgt, dft, prompt, 20, k=k, temperature=0.0)
        assert torch.equal(spec, generate(tgt, prompt, 20, temperature=0.0)), k
    print("greedy: identical token ids to generate(), for every k")

    # 2. and the sampled case, which cannot be checked by equality -- only in
    #    distribution. Draw the first token many times down each path and
    #    compare against the target's exact distribution. The draft is
    #    deliberately bad, so nearly every draw goes through the residual
    #    branch, which is precisely the code that has to be right
    with torch.no_grad():
        p_true = softmax(tgt(prompt)[:, -1], dim=-1)[0]

    def histogram(fn, trials: int = 4000) -> Tensor:
        counts = torch.zeros(V)
        for i in range(trials):
            g = torch.Generator().manual_seed(1000 + i)
            counts[fn(g)[0, -1].item()] += 1
        return counts / trials

    spec_hist = histogram(
        lambda g: speculative_generate(tgt, dft, prompt, 1, k=4, generator=g)
    )
    gen_hist = histogram(lambda g: generate(tgt, prompt, 1, generator=g))
    tv_spec = 0.5 * (spec_hist - p_true).abs().sum().item()
    tv_gen = 0.5 * (gen_hist - p_true).abs().sum().item()
    #    at 4000 draws over V=64 the expected TV from sampling noise alone is
    #    about 0.05, so an absolute threshold would only be measuring the noise.
    #    The comparison that means something is against generate(), plus a
    #    control that proves the test can see a difference at all: sampling from
    #    the DRAFT lands far away, so any draft distribution leaking into the
    #    output would show up here
    draft_hist = histogram(lambda g: generate(dft, prompt, 1, generator=g))
    tv_draft = 0.5 * (draft_hist - p_true).abs().sum().item()
    print(
        f"TV from the target's true distribution -- spec {tv_spec:.4f}   "
        f"generate {tv_gen:.4f}   (draft, as a control: {tv_draft:.4f})"
    )
    assert tv_spec < 2 * tv_gen + 0.02, "speculative output drifted from the target"
    assert tv_draft > 4 * tv_spec, "control failed: the test cannot see a difference"

    # 3. the ceiling, and a check that acceptance is doing anything at all: a
    #    draft that IS the target agrees with itself, so nothing is ever
    #    rejected and every round yields k+1 tokens -- the k guesses plus the
    #    free one the target forward produced anyway
    for k in (1, 4):
        st: dict = {}
        speculative_generate(tgt, tgt, prompt, 20, k=k, temperature=0.8, stats=st)
        assert st["accept_rate"] == 1.0
        assert abs(st["tokens_per_target_pass"] - (k + 1)) < 1e-9
    print("a draft equal to the target accepts everything: k+1 tokens per pass")

    # 4. mismatched vocabularies are caught, not silently mis-decoded
    try:
        speculative_generate(tgt, GPT(**asdict(replace_v := small)), prompt, 4)  # noqa
    except AssertionError:
        pass
    other = GPT(
        **asdict(
            GPTConfig(
                vocab_size=V * 2,
                block_size=T,
                n_embed=16,
                n_head=2,
                n_layer=1,
                attention="sdpa",
            )
        )
    )
    try:
        speculative_generate(tgt, other, prompt, 4)
        raise SystemExit("should have failed")
    except AssertionError as e:
        assert "vocabulary" in str(e)

    # 5. the real thing. A 5.4M draft trained to agree with the 27M target
    try:
        t_path = CKPT_DIR / "big_2026-08-30_09-09-16.pt"
        d_path = latest_ckpt("draft")
        assert t_path.exists()
    except (FileNotFoundError, AssertionError):
        print("\nno trained checkpoints; skipping the real benchmark")
        print("ok")
        raise SystemExit

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    target, t_meta = load_checkpoint(t_path, dev)
    draft, d_meta = load_checkpoint(d_path, dev)
    n_t = sum(p.numel() for p in target.parameters())
    n_d = sum(p.numel() for p in draft.parameters())
    print(f"\ntarget {t_path.name}  {n_t / 1e6:.1f}M  val {t_meta['val_loss']:.3f}")
    print(
        f"draft  {d_path.name}  {n_d / 1e6:.1f}M  val {d_meta['val_loss']:.3f}"
        f"  accept {d_meta.get('accept', float('nan')):.3f} measured in training"
    )

    start = torch.zeros(1, 1, dtype=torch.long, device=dev)
    N = target.block_size - 1

    def timed(fn) -> tuple[float, Tensor, dict]:
        st: dict = {}
        fn(8, st)  # warm up
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(N, st)
        if dev == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, out, st

    base_s, base_out, _ = timed(
        lambda n, st: generate(target, start, n, temperature=0.0, use_cache=True)
    )
    print(
        f"\n  {'k':>2}  {'accept':>7}  {'tok/pass':>8}  {'predicted':>9}  "
        f"{'time':>7}  {'vs cache':>8}"
    )
    for k in (1, 2, 4, 8):
        secs, out, st = timed(
            lambda n, st, k=k: speculative_generate(
                target, draft, start, n, k=k, temperature=0.0, stats=st
            )
        )
        assert torch.equal(out, base_out)  # greedy: still the same tokens
        #    acceptance is a property of the two models, not of k, so one alpha
        #    should explain every row through (1 - a^(k+1)) / (1 - a). Measured
        #    against predicted is the whole theory in a single assert
        predicted = tokens_per_pass(st["accept_rate"], k)
        assert abs(predicted - st["tokens_per_target_pass"]) < 0.15, (k, predicted)
        print(
            f"  {k:>2}  {st['accept_rate']:>7.3f}  "
            f"{st['tokens_per_target_pass']:>8.2f}  {predicted:>9.2f}  "
            f"{secs:>6.2f}s  {base_s / secs:>7.2f}x"
        )
    print(f"  cache-only baseline: {base_s:.2f}s for {N} tokens")

    print("\nok")
