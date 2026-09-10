# Roadmap

Updated 2026-09-09, after quantization finished.

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

**Sliding window (rung 1 of 3).** `sliding_window.py` — one function,
`sliding_window_mask(T_q, T_kv, window)`, which subsumes the cached-query shift
the two SDPA layers were already building by hand (`window=None` is exactly the
old band). Wired into `sdpa_attention.py` and `gqa_attention.py` behind
`window=W`, and through `Block` / `GPT` / `GPTConfig` as `window`, gated to
`attention="sdpa"` or `"gqa"` the same way `use_rope` is.

The three rungs are independent and were deliberately not done together: a
window is a **mask**, a ring buffer is **storage that wraps**, attention sinks
are a **policy about which slots never get evicted**. Rung 1 changes what the
model computes and nothing about what it stores — the cache still holds every
position and the kernel still walks all of them. Both attention layers keep
their mask-free fast paths alive while `T_kv <= window`, so a decode crosses a
branch boundary mid-run; that boundary is what the cache tests step over.

It asks nothing of the positional scheme — a window is correct under learned
absolute positions too (Longformer, 2020). RoPE's `i - j` property gates rungs
2 and 3, where keys are *evicted* and positions re-indexed, not this one.
Tested against an eager band-mask oracle (and, per row, against attention over
the literal `k[i-W+1:i+1]` slice — the masked columns contribute nothing).
Receptive field is `L*(W-1)+1`, proved twice: boolean matrix powers of the mask,
and a real stack of windowed layers where the out-of-reach positions move by
**exactly zero**. Mistral-7B's W=4096 over 32 layers is a 131k-token reach.

**Ring buffer (rung 2).** `ring_kv_cache.py` — `RingKVCache(KVCache)`, capacity
`W` instead of `block_size`, cursor wraps, the entry for position `p` lives in
slot `p % W`. Memory per layer stops growing with the run: measured flat at
0.50 GB from 4k to 256k context (32L/8kv/hs=128, bf16) against 32 GB for a full
cache. Wired into `GQAttention` as `ring=True` and through `Block` / `GPT` /
`GPTConfig`, gated to `attention="gqa"` — `"sdpa"` grows its cache by `torch.cat`
and never held a `KVCache` to wrap.

Three things this pass established, all measured rather than asserted:

- **Slot order is not information.** Softmax is permutation-equivariant over the
  key axis, so the ring never needs un-permuting — only the *mask* needs to know
  which position lives in which slot. `sliding_window.py` grew
  `window_mask_from_positions(q_pos, k_pos, window)` for that, and
  `sliding_window_mask` is now the contiguous special case of it.
- **Decode needs no mask at all.** Every slot a ring still holds is inside its
  window by construction, so rung 2 takes back out of the hot path the mask
  rung 1 put into it. The mask survives only for prefill and chunked verify.
- **Evicting == masking.** A ring decode reproduces a full-cache windowed decode
  to `1e-6` (`ring_kv_cache.py` test 7, `gqa_attention.py` tests 10–11, over a
  run long enough for the cursor to go round 8x). Eviction is not an
  approximation of the windowed model; it *is* the windowed model.

Two contracts broke, both on purpose. `append` no longer returns what it stores —
a prefill longer than the ring gets back every key it is entitled to and only
its tail is retained. And `rollback` refuses once the ring has wrapped, because
the entries it wants back are the ones eviction overwrote: speculative decoding
and eviction are genuinely mutually exclusive there, so it asserts rather than
returning stale keys.

`GPT.forward` had a latent bug the ring exposed: `T_past` was read off the
cache's width, which is the same number right up until a cache learns to evict.
It now asks for `pos` and falls back to the width only for the plain tuple that
`"fused"` and `"sdpa"` return.

Note `block_size` now does double duty in `GQAttention`: it sizes the rope
tables and selects the preallocated cache, so `use_rope=True` implies buffer
mode. Harmless today; a separate flag if it ever stops being.

**Online softmax.** `online_softmax.py` — the streaming `(m, l)` recurrence and
a `flash_attention` built on it, tiled over both queries and keys. Closes the
one place the series said *trust the kernel*: `sdpa_attention.py` measured the
T² blow-up and handed the fix to SDPA.

Peak memory, 4060, B=2 nh=4 hs=64, fp32:

| T | naive | flash (128×128) | SDPA |
|---|---|---|---|
| 512 | 44 MB | 23 MB | 21 MB |
| 1024 | 119 MB | 28 MB | 25 MB |
| 2048 | 413 MB | 38 MB | 33 MB |
| 4096 | 1577 MB | 58 MB | 49 MB |

**Deliberately not wired.** It changes no logits, so it earns no config knob —
the criterion the sliding window passes and this fails. Same output as `sdpa`,
slower (65 vs 4.4 ms at T=4096), more memory. It exists to be measured, not
called.

**Quantization.** `quantize.py` — `quantize` / `dequantize`, `pack_bits` /
`unpack_bits` at any width 2-8, `QuantizedLinear`, `QuantizedEmbedding`,
`quantizable`, `quantize_model`. Symmetric, scale-only, per-output-row by
default (`dim=-1`); `dim=None` is per-tensor; `bits` is the width and
`override` sets it per dotted path, which is how a mixed model is built.
Post-training, so everything is a buffer and nothing is a Parameter. Not wired
into `GPTConfig` — same criterion as online softmax. Measurements live in
`quantize.ipynb`, seven probes.

On the trained 27M checkpoint, **int8 costs +0.0001 bpc for 3.97x**. The full
curve is in *Measured* below; int6 is the quality sweet spot and int4 the
memory one.

Tied weights are quantized **once** and the buffers shared, and a tied pair
given two widths asserts rather than unsharing. Quantizing one side of a tie is
the trap: it leaves an fp32 table plus a fresh narrow copy and the pair gets
*bigger*. Measured on a toy GPT — both 142 KB, neither 236 KB, head-only 270 KB.
It works because `dim=-1` on a `[V, E]` tied weight means per-token for the
lookup and per-output-channel for `lm_head`: the same axis.

Packing is what makes a narrow width save anything — a 4-bit weight in an int8
tensor is still a whole byte. `block(bits)` gives the smallest run that lands on
a byte boundary (`8 // gcd(bits, 8)` values), and `acc_dtype` the narrowest
signed container it fits in. Smallest block is right: bigger blocks halve the
packed-side tensors but force a wider container on the unpacked-side ones, which
are one element per weight and dominate. Measured 34 MB (k=4, int32) against
64 MB (k=8, int64) at 6 bits.

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

- **Any `attn_mask` costs the flash backend.** Measured on the 4060:
  `is_causal=True` in bf16 runs `FLASH_ATTENTION`; the same call with a bool
  `attn_mask` does not, and falls to `EFFICIENT_ATTENTION`. So a sliding window
  expressed as a mask is a *slowdown* over full causal attention at short
  context, not a speedup — and the mask is itself a `[T_q, T_kv]` object, the
  T² tensor SDPA exists not to materialise (4 GB at T=65536). A real windowed
  kernel takes W as an integer and skips blocks. `sliding_window.py` test 6.

- **A fully masked tile makes `nan`, not zero.** `-inf - -inf` in the rescale,
  when the running max is still `-inf`. Substituting 0 for `m_new` there is
  exact — every term involved is `-inf`, so both exps are 0 and the state is
  unchanged. Fires on sliding windows (masked blocks come *first*), not on plain
  causal, where block 0 is always visible. `online_softmax.py` test 4.

- **The block-skip bound must be a host int.** Comparing `ki` against a CUDA
  tensor syncs once per block: 4.89 ms of pure loop overhead at T=4096 vs
  0.02 ms for a Python int, ~6% of total runtime. `q_pos.max()` is worse still
  (5.78 ms). The traffic was the bottleneck again — 528 syncs instead of 512 MB.

- **int8 weight-only is close to free, and KL is the instrument.** On the
  trained 27M: per-channel +0.0001 bpc, per-tensor +0.0005 — both buried in the
  fourth decimal. The same comparison in **KL against the fp32 logits is 1.78e-4
  vs 1.08e-3, a clean 6.1x**, from 8 sequences instead of a 1M-token sweep. Use
  Δbpc for the headline, KL for anything that has to discriminate.

- **Per-tensor buys nothing.** 3.99x against per-channel's 3.97x — the scales are
  one float per row against a whole matrix of int8, under 1% of the size, for 6x
  the accuracy. There is no regime in this model where per-tensor is right.

- **Quantize the embeddings.** Skipping them costs 3.97x -> 3.16x to save
  0.4e-4 nats. A fifth of this model is embedding.

- **The position table has the widest rows in the model** — 12.3x between its
  loudest and quietest row, against 2.7x for the token table and 1.8-8.1x for
  the blocks. Early positions appear in every window, late ones only in long
  ones. Attention spreads wider than FFN, and early blocks wider than late.
  `quantize.ipynb` probe 2; the four-weight sample in `quantize.py` missed it.

- **Quantization error is ~96% additive, so KL/MB is a budget and not a
  heuristic.** 35 per-layer KLs sum to 1.71e-4 against 1.78e-4 measured
  all-at-once. Greedy selection by KL-per-MB then predicts the total to within
  4% at every point from 1 to 34 units. For small perturbations
  `KL ~ 1/2 d'Fd`, so additivity says the per-layer logit perturbations are
  close to orthogonal — what independent rounding errors should be.

- **One bit is worth 4x, from 4 bits up.** KL is quadratic in the step size and
  a bit halves the step: 4->5 measures 4.9x, 5->6 measures 4.3x, 6->8 measures
  **16.5x against 16x predicted**. Below 4 bits it breaks (8.4x for 3->4, 14.8x
  for 2->3) because the perturbation stops being small. int2 lands at bpc 3.04,
  *worse than the untrained model's 2.96* — destroyed, not degraded.

- **The width curve, trained 27M, real packed sizes.** int4 13.2 MB / +0.0244
  bpc; **int5 16.5 / +0.0050; int6 19.7 / +0.0012**; int8 26.2 / +0.0001;
  fp32 104.0. int6 is the quality sweet spot and it is not a width anyone names.

- **Row-wise scales, then stop.** Per-tensor is 6.1x worse in KL for 0.5% less
  size, so it is never right. But **group-wise scales are not worth it here**:
  within-row spread is 1.5-1.8x against 2.7-12.3x between rows, so groups attack
  a problem 4x smaller than the one per-row already solved, at 11% more storage
  (G=64 is 4.5 effective bits/weight against 4.05). Real formats need them
  because outlier *features* appear past ~6.7B params; at 27M they do not.

- **Spread does not predict sensitivity.** The position table has the widest
  rows in the model (12.3x) and is the *second least* sensitive layer. Per-row
  scaling already neutralises spread — that is its job — so a wide ratio
  predicts how much per-row beats per-tensor, not what quantizing that layer
  costs. Two different questions.

- **Mixed precision is real but bounded by `log4` of the sensitivity spread.**
  Optimal allocation is `b_i = log4(A_i / n_i) + c` — bits scale with the log of
  **per-weight** sensitivity, so ranking on raw KL protects big layers for the
  wrong reason. Per-weight sensitivity spans 28x here, so the optimal spread is
  **2.4 bits**, and mixed buys ~18% less error at matched size (16.4 MB /
  +0.0041 against uniform int5's 16.5 / +0.0050). An earlier test of **int8/int4
  lost to uniform** — a 4-bit gap against a 2.4-bit optimum overshoots both
  ends. Size and sensitivity are uncorrelated (r = -0.05), so the tempting
  "downgrade the big insensitive layers" story is not the mechanism; bits are
  *moved*, and a swap pays whenever `KL_i > 4 KL_j`.

- **Nothing is faster, and the loss is shaped backwards.** Dequantization cost
  tracks weight size so it is paid *per forward*, while matmul cost scales with
  tokens: ~3.6x slower at decode (B=1,T=1), ~1x at batch. A real int8 kernel is
  the opposite — it helps decode, which is bandwidth-bound. **Memory does pay
  though**: total in flight (resident + transient) is fp32 104 MB, int8 42.2,
  int6 53.7, **int4 31.2**. int6 costs more in flight than int8 because a 24-bit
  block needs int32 where int4's 8-bit block fits int16.

- **`sum()` over an integer tensor promotes to int64 unless given `dtype=`.**
  It silently undid `acc_dtype` one line after it was chosen, making every
  downstream temporary int64. Fixing it took int4's transient from 60 MB to 18
  and decode from ~5.5x to ~3.6x. A conclusion about a *technique* was really a
  fact about a missing keyword argument — the reason probe 7 records it.

- **Microbenchmarks of a cache flatter it.** Three times this session an
  isolated measurement overstated: widen-vs-MHA was 2.3x on the bare attention
  op and 1.1x in the layer; preallocation was 26x on the copy and 1.00x in
  `generate`. Isolate any component of a decode step and it looks like the
  bottleneck.

## Where things go

Three files per topic, and the split is about what can *fail*, not what is slow:

- **`video/<topic>.py`**, tests under `__main__` — anything that can break
  because of a **code** change. Kept fast (`quantize.py` is 1.9s). Artifact
  dependencies are fine when guarded: check the checkpoint exists, print a skip
  line, exit clean.
- **`video/<topic>.ipynb`** — **measurements**: anything that can only change
  when an artifact does (a checkpoint's weight statistics, a dataset's token
  counts), plus sweeps too slow to run every time, plus the topic's formulas in
  LaTeX. Outputs committed; they are the archive. Cell 1 is the prelude —
  imports and shared helpers — and **a probe cell may read names from the
  prelude, never from another probe cell**, so any cell runs after a restart.
  A helper earns a place in the prelude when a second probe needs it.
- **`walkthroughs/<topic>.ipynb`** — rough derivation scratch, never re-run.

The rule caught a real case: `quantize.py`'s row-max probe took 30ms but
asserted a property of the *checkpoint*, so no code change could fail it. It
moved to the notebook, where the full 36-weight version found the 12.3x position
table the 4-weight sample had missed.


## Next, in order

**1. BPE tokenizer + FineWeb-Edu.** `src/tokenizer.py` works; port and clean it
into `video/`. Byte-level, count pairs, merge, repeat. This is also where the
chat special tokens get minted, so it has to precede SFT. Then real data at a
size that does not fit in RAM.

**2. Padding and attention masks.** *Currently missing everywhere in
`src/video/`* — checked, not assumed. Fine for pretraining on contiguous shards,
a blocker for everything after it: SFT needs the loss masked over prompt tokens
(`ignore_index`), and batched generation needs left-padding plus a real mask,
since `generate` assumes every row shares a prompt length. That makes this gate
RL as well as SFT — rollouts are batched generation. Small file. It has to land
before SFT, not during.

**3. SFT + LoRA.** Cleaner versions of `src/sft.py` and `src/lora.py`. Chat
template, loss masking, then LoRA as the parameter-efficient variant. This is
where it stops being a continuation engine.

**4. RL.** **DPO first** — a loss function over a frozen reference model, no
reward model, no rollouts, no value head, which fits the file-plus-test format.
PPO/GRPO after, and budget three episodes: sampling loop, advantage estimation,
KL control. It is the first thing in the series that can silently fail to learn.

## Worth covering, unscheduled

- **Activation quantization** — the half `quantize.py` does not do. Needs
  calibration, is where per-tensor genuinely fails, and weight-only already
  delivers the 4x. Probably not an episode.
- **Chat templates and special tokens** — cheap, high payoff, pairs with SFT.
  The difference between a continuer and an assistant is mostly a format contract.
- **Attention sinks (rung 3)** — sidestepped 2026-09-07 for quantization, not
  abandoned. Rungs 1 and 2 shipped; the plan is kept verbatim below.
- **MoE** — routing, top-k experts, load-balancing loss. Self-contained.
- **An eval beyond val loss** — bpc is there; something task-shaped makes the SFT
  and RL episodes legible.
- **YaRN / NTK context extension** — only as a RoPE sequel, if RoPE lands well.

### Attention sinks (rung 3), parked

Rungs 1 and 2 shipped — see *Sliding window* and *Ring buffer* above. Rung 3 is
attention sinks: pin the first `S` slots, ring the rest. This is the
rung that forces the bend — once generation runs past `block_size` the rope
table is exhausted, and StreamingLLM's fix is to **re-index positions within the
cache**, which needs keys stored *unrotated* and rotated at read time. Keep
write-time rotation for rungs 1–2 and introduce read-time rotation as rung 3's
lesson: rotating the whole window every step is the cost, and the cost is the
point.

Concretely, what rung 3 has to change: `RingKVCache` gains `S` pinned slots the
cursor skips, so the wrap is over `slots[S:]` rather than all of them —
`positions` already carries everything the mask needs, so that side is done.
Then `GPT.forward`'s `T_past + T <= block_size` assert becomes the binding
constraint, because it is the rope table, not the cache, that runs out. That is
where write-time rotation has to give.

Also unblocked and unrun: the reverse-string probe below. Do it before or after,
but do not let it silently not happen.

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
