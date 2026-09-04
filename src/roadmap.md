# Roadmap

Updated 2026-09-04, after RoPE landed.

## Done

Everything below is built from scratch in `src/video/` — one file per topic, each
with its own `__main__` that tests it against a torch oracle.

**Model.** Linear, Embedding, LayerNorm, RMSNorm, ReLU, SiLU, softmax, dropout,
cross-entropy, multinomial. Attention three ways: per-head `MultiHeadAttention`,
`FusedQKVAttention` (one `[E -> 3E]` matmul, the checkpoint layout), and
`SDPAttention`. Dense and gated (SwiGLU) FFNs. Block, GPT, weight tying,
`1/sqrt(2·n_layer)` residual init.

**Training.** AdamW with decay groups (2D only) and betas (0.9, 0.95),
`clip_grad_norm` at 1.0, warmup + cosine decay, gradient accumulation,
`torch.compile`, bf16 autocast, memmapped `uint16` shards, checkpoint/resume,
wandb logging.

**Inference.** Temperature, top-k, top-p, min-p. KV cache through the whole
stack — attention, block, GPT, generate. Measured 5.2x on 12L/768E over 768
tokens, 2.5x on the trained 8L/512E checkpoint over 512.

**GQA / MQA.** `gqa_attention.py` is the one to build on (SDPA plus
`enable_gqa`); `fused_gqa_attention.py` is the eager artifact, kept because the
grouping is only legible when you can see the scores. `n_kv_head` runs through
`Block`, `GPT` and `GPTConfig`, and `"gqa"` is a first-class attention.
`fused_qkv_attention.py` was deliberately left at its pre-GQA state.

**Preallocated KV cache.** `kv_cache.py` — `KVCache` with two modes: grow by
copying (what the old `type KVCache = tuple[Tensor, Tensor]` did) and a
preallocated buffer written at a cursor. It kept the tuple's interface, so the
older attention files needed no edits at all. Buffers allocate on first write,
so device and dtype follow the data.

**Speculative decoding.** `speculative_decoding.py` — a 4.4M draft guesses k
tokens, the 27M target verifies all of them in one forward, and the accept /
residual-resample rule keeps the output distribution exactly the target's.
`KVCache.rollback` came with it: the first thing in the series that needs a
cache which can go backwards. The draft was trained for it, and `evaluate.py`
gained `estimate_agreement` — argmax agreement plus `sum_x min(p, q)`, which is
the acceptance rate itself — so `train.py` can select a checkpoint on `accept`
rather than `val_loss`.

**Sinusoidal positions.** `sinusoidal.py`, wired into `GPT` and `GPTConfig` as
`position="learned" | "sinusoidal"` through a `make_position` factory, matching
how `attention` / `norm` / `ffn` are chosen. Zero parameters and no checkpoint
key: the formula rebuilds itself in `__init__`.

**RoPE.** `rope.py` — `rope_tables` + `apply_rope` + a stripped-down
`RopeAttention` (no dropout, no cache, no GQA) that exists so the rotation is
legible on its own. Interleaved pairing, the paper's, matching `sinusoidal.py`'s
`2i, 2i+1` columns. Wired into `sdpa_attention.py` and `gqa_attention.py` behind
`use_rope`, and into `GPT` / `GPTConfig` as `position="rope"`, where
`make_position` returns **None** — there is nothing to add to the residual
stream, which is the whole difference from the two additive schemes.

Two facts the tests pin, both `== 0` rather than a tolerance. Rotation commutes
with the kv-head broadcast, so a `k` with `n_rep` fewer heads needs no special
handling in GQA. And the mirror pair `(m, n)` / `(n, m)` does *not* match:
distance is signed, and the sin half cancels only when `k = q`. Cached keys go
in **already rotated**, rotated once, by the position the token actually had —
`T_past` from `kv_cache.pos`. That is the invariant the next item depends on.

Note `block_size` now does double duty in `GQAttention`: it sizes the rope
tables and selects the preallocated cache, so `use_rope=True` implies buffer
mode. Harmless today; a separate flag if it ever stops being.

The old "three genuinely big remaining pieces" list is closed except for data:
SDPA, RMSNorm, SwiGLU, weight tying, and the KV cache all shipped. Optimizer
hygiene — the item that sat pending longest — is done too.

## Measured, so it does not get re-derived

- **Fold q, do not widen k/v.** With `n_kv_head < n_head` the obvious move is
  `repeat_interleave` on k and v up to `n_head`. It is correct and it rebuilds,
  once per layer per token, the tensor the small cache exists not to store.
  Folding the group into q instead — scores become `[B, nkv, n_rep*T, T_kv]` —
  is **3.2x faster at decode** (0.56 vs 1.77 ms, B=16, T_kv=2048). Widening
  keeps the storage win and hands back the entire latency win: it lands within
  noise of the MHA it was supposed to beat. `enable_gqa=True` does the same
  thing inside the kernel. `fused_gqa_attention.py` tests 8 and 9.

- **Preallocation is a shape change, not a speedup.** The copy alone is
  **26x** (`kv_cache.py` test 8). End to end through `generate` at B=1 with
  `n_kv_head=2` it is **1.00x**. The copy is a few percent of a decode step, and
  it does not grow relative to one — attention reads the cache once per step
  too, so context scales both sides. Do it for the cursor (ring buffer, sliding
  window, paged attention) and for not churning the allocator.

- **GQA and preallocation attack the same cost**, so each shrinks the other's
  payoff. Preallocation is worth 1.10x at `n_kv_head=8` and 1.00x at
  `n_kv_head=2`, B=1. Measured up to 1.37x only at B=4 with an MHA-width cache —
  i.e. in the configuration GQA just removed.

- **Speculative decoding: the algorithm works, the clock does not.** Against
  the trained pair, acceptance is **0.82 and constant in k** (0.819 / 0.820 /
  0.821 / 0.809 at k = 1 / 2 / 4 / 8) — acceptance is a property of the two
  models, not of how far ahead you guess. Tokens per target forward follow
  `(1 - a^(k+1)) / (1 - a)` to two decimals: **3.51 measured against 3.50
  predicted at k=4**. Wall clock is **0.83-0.89x — slower than plain cached
  decoding**, because at 27M params on a 4060 a target step is launch-bound, so
  four draft forwards cost more than the target forwards they save. The speedup
  needs a target expensive relative to its draft.

- **Live acceptance beats teacher-forced acceptance.** Training measured
  `accept = 0.726` on validation text; generation measured 0.82. The prediction
  was the opposite — distribution shift was supposed to *lower* it. The model's
  own output is simply more predictable than real text.

- **Divide acceptance by what was tested, not by k.** After the first rejection
  the remaining guesses are dropped untested; counting them makes acceptance
  look like it falls with k when it is flat.

- **Microbenchmarks of a cache flatter it.** Three times this session an
  isolated measurement overstated: widen-vs-MHA was 2.3x on the bare attention
  op and 1.1x in the layer; preallocation was 26x on the copy and 1.00x in
  `generate`. Isolate any component of a decode step and it looks like the
  bottleneck.

## Next, in order

**1. Ring buffer, sliding window, attention sinks.** The half of preallocation
that actually pays, and now unblocked: scores depend on `i - j`, which is what
made evicting cache entries legitimate. `kv_cache.py` has the cursor but never
wraps: capacity is `block_size` and a run still has to fit. Evicting the oldest
entries and reusing their slots is the next step. StreamingLLM's wrinkle belongs
here too — it re-indexes positions *within the cache*, which needs keys rotated
at read time rather than baked at write time, and the current layers bake at
write time. That is the one place this design will have to bend.

Also unblocked and unrun: the reverse-string probe below. Do it before or after,
but do not let it silently not happen.

**2. BPE tokenizer + FineWeb-Edu.** `src/tokenizer.py` works; port and clean it
into `video/`. Byte-level, count pairs, merge, repeat. This is also where the
chat special tokens get minted, so it has to precede SFT. Then real data at a
size that does not fit in RAM.

**3. Padding and attention masks.** *Currently missing everywhere in
`src/video/`* — checked, not assumed. Fine for pretraining on contiguous shards,
a blocker for everything after it: SFT needs the loss masked over prompt tokens
(`ignore_index`), and batched generation needs left-padding plus a real mask,
since `generate` assumes every row shares a prompt length. That makes this gate
RL as well as SFT — rollouts are batched generation. Small file. It has to land
before SFT, not during.

**4. SFT + LoRA.** Cleaner versions of `src/sft.py` and `src/lora.py`. Chat
template, loss masking, then LoRA as the parameter-efficient variant. This is
where it stops being a continuation engine.

**5. RL.** **DPO first** — a loss function over a frozen reference model, no
reward model, no rollouts, no value head, which fits the file-plus-test format.
PPO/GRPO after, and budget three episodes: sampling loop, advantage estimation,
KL control. It is the first thing in the series that can silently fail to learn.

## Worth covering, unscheduled

- **Chat templates and special tokens** — cheap, high payoff, pairs with SFT.
  The difference between a continuer and an assistant is mostly a format contract.
- **Quantization (int8/int4)** — weight-only is ~100 lines, and it is the answer
  to "how does this scale to a 7B on a 4060".
- **Online softmax / flash internals** — `sdpa_attention.py` currently says eager
  ops cannot express the tiling and leaves it there. Deriving the streaming
  recurrence closes the one place the series says "trust the kernel".
- **MoE** — routing, top-k experts, load-balancing loss. Self-contained.
- **An eval beyond val loss** — bpc is there; something task-shaped makes the SFT
  and RL episodes legible.
- **YaRN / NTK context extension** — only as a RoPE sequel, if RoPE lands well.

## Skip

**DDP / multi-GPU.** One 4060. All-reduce can be explained in three minutes
inside the gradient-accumulation episode without a setup that cannot be run.

## Pending probes

### RoPE length extrapolation

`sft.ipynb` has a reverse-string task (`hello>olleh`) that measures length
generalization directly. Trained on word lengths 3-8 with `block_size` sized for
10, the learned-position model scores **1.0 on `evaluate()` and 0.0 on
`evaluate(lmin=9, lmax=10)`** — it emits correct chunks of the reversal at the
wrong offsets.

Why it fails: position `t` must attend to `2l-1-t`, a *reflection*, so the
required offset `2l-2t-1` depends on both `t` and `l` — no single rule covers it.
But valid `(l, t)` pairs for l∈3-8 number only 33, small enough for a 32k-param
model to memorize as a lookup. Lengths 9-10 add 19 pairs with no table entry, and
the model is wrong from the very first output character.

Measured, not assumed: position rows 0-15 all train normally; only rows 16-19
stay at init with exactly zero gradient (last kept target is `t=2l-1=15` at
`l=8`, and causality stops later rows from ever reaching a kept position). So
dead rows are a minor secondary effect — they only touch the tail of `l=9,10`.
The missing lookup entries are the real cause.

RoPE has landed (`position="rope"`, needs `attention="sdpa"` or `"gqa"`), so
this is now runnable and has not been run. RoPE removes the untrained-absolute-
rows problem and makes scores depend on `t-p`, so the held-out number may improve —
but it does not hand the model a reflection, and length generalization on
reverse-and-copy is known-hard for positional encoding alone. **Treat a flat 0.0
as a fact about the task, not proof the RoPE code is broken.** Verify RoPE
separately on val loss; use this as a bonus probe.

### Cropping the KV cache

Not in the files, deliberately. Cropping the cache runs, never errors, and
returns quietly wrong logits: cached k/v are baked with their absolute positions
at write time, so a crop leaves survivors mis-numbered and colliding with the new
token. The uncached path is correct only because it crops *ids* and recomputes.
Revisit as a demo once RoPE makes it legitimate — that transition is the whole
point of sliding-window attention.
