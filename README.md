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

**Attention sinks (rung 3).** `attention_sinks.py` — `SinkKVCache(RingKVCache)`
pins the first `S` slots and rings the rest, capacity `S + window`. Wired into
`GQAttention` as `sinks=S` and through `Block` / `GPT` / `GPTConfig`, gated to
`ring=True` and `position="rope"`.

Pinning is four lines. What it drags in is positions, which is the bend the
roadmap parked this for. Once a run outlives `block_size` the rope table is
exhausted, so a cached key is rotated by its **rank inside the cache** rather
than by where its token sat in the text (StreamingLLM, Xiao et al. 2024). Keys
can therefore no longer be rotated once on the way in — the rank changes every
time something is evicted. **They go in unrotated and are rotated on the way
out, the whole cache, every step.** That is rung 3's cost and it is the point:
rungs 1–2 rotate one key per decode step, rung 3 rotates `S + W`.

What that buys is the assert. `GPT.forward` capped a cached run at
`T_past + T <= block_size`; for a streaming model it now caps only the chunk,
because ranks never exceed capacity. `gpt.py` test 22 decodes 40 tokens through
a `block_size` of 8, every layer still holding position 0, and `generate.py`
generates 65 on a 16-row table. The same model without sinks asserts at token 9
— and a sinkless *ring* cannot even be run that far under rope, because
write-time rotation indexes the table by absolute position and step `block_size`
finds the slice empty (`gqa_attention.py` test 13).

Three things pinned by test, since none of them is obvious:

- **Rung 3 is a no-op until something is evicted.** With fewer tokens than
  `S + W`, ranks *are* absolute positions, so a sink layer is byte-for-byte a
  plain rope layer. Without that, the oracle test below proves nothing.
- **Re-indexing preserves distances inside the window and only lies about the
  gap.** For a window key, rank distance == absolute distance exactly; for a
  sink it is compressed, which is the same approximation the mask already made.
- **The oracle.** A sink decode reproduces, to `1e-6`, a reference that keeps
  every key, picks the visible set by hand and rotates by rank
  (`attention_sinks.py` test 7 at cache level, `gqa_attention.py` test 13 at
  layer level). That is what catches `q` and `k` being ranked differently, which
  no shape check would.

Not measured here, deliberately: the *quality* claim. An untrained model has no
sink behaviour to lose, so "evicting position 0 wrecks perplexity" is the
paper's result, not this file's — it needs a checkpoint, and belongs in a
notebook. What is testable in the `.py` is the mechanism, and that is what the
tests cover.

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

**BPE + FineWeb-Edu.** `tokenizer.py` — byte-level BPE: regex pre-split, count
adjacent pairs, merge the most frequent, repeat. `train` / `encode` / `decode` /
`save` / `load`, GPT-2's pre-tokenizer pattern and GPT-4's. Two backends behind
`VIDEO_BACKEND` and the same ids either way: `OurBPETokenizer` in Python,
`FastBPETokenizer` over tiktoken. They agree because both are built from one
merge table — ours adopts GPT-2's with `from_mergeable_ranks` and hands its own
back with `mergeable_ranks`, which is also how tiktoken becomes the oracle for a
vocabulary *we* trained.

A rank table says what each id expands to but not which pair built it. BPE's own
invariant recovers that: encode a token's bytes using every lower rank and
exactly two pieces must remain, and those are the pair. 50000 merges in 0.55s.
GPT-2's byte ids are a permutation of 0-255, not the identity, so the base table
is a lookup rather than an assumption.

The series ships GPT-2's 50257 plus `<|im_start|>` and `<|im_end|>`, so
`VOCAB_SIZE` is 50259 and still fits uint16. The chat specials are minted here
because the corpus has to be tokenized with them already in the table — that is
the whole reason this episode precedes SFT. `src/tokenizer.py`'s older save
format still loads (GPT-4 pattern, identity byte ids), verified to reproduce
`bpe_ts_4096.json` id for id, so `basic.py` and every TinyStories checkpoint keep
working; `basic.py`'s `sys.path` hack into `src/` is gone.

`fineweb.py` — `ShardWriter` plus a streamed build. 990M train + 10M val tokens
of `HuggingFaceFW/fineweb-edu` `sample-10BT` at 4.59 chars/token: 1.9 GB in ten
100M-token shards, 18.3 minutes. Documents are led by `<|endoftext|>`.
`dataset.py`'s `BinDataset` now spans shards as one contiguous stream (bisect
over cumulative offsets) and takes a `data_dir`; `VIDEO_DATASET=fineweb_edu`
switches every consumer with no edit to `train.py`. TinyStories is the
single-file case and still runs. Measurements in `tokenizer.ipynb`, seven probes.

The old "three genuinely big remaining pieces" list is closed, data included:
SDPA, RMSNorm, SwiGLU, weight tying, and the KV cache all shipped. Optimizer
hygiene — the item that sat pending longest — is done too.

### Finetuning: a Task interface, and the first two tasks

`task.py` — `Packed` (a flat uint16 stream plus `mask` and `starts`), `windows()`
which opens every training window at an example start so a prompt sits at
position 0 exactly as it does at inference, `masked_targets` into
`ignore_index`, and a three-method `Task` protocol: `build`, `reward`,
`evaluate`. `reward` returns a float rather than a bool because GRPO needs the
magnitude. `concat()` mixes two tasks, and start count sets the ratio. Nothing
downstream names a dataset.

`mask.py` — document masking from eot positions, as a materialized `[B,1,T,T]`
bool and as a flex `BlockMask`. `flex_attention.py` — GQA plus rope, flex while
training and SDPA at decode, with the rope angles threaded from `GPT` instead of
tabled per layer. `cross_entropy.py` grew `ignore_index` on the hand-written
backend, keyword-only so the two aliases agree at the call site.

`sft.py` — `train.py`'s loop fed by a Task: windows instead of `get_batch`, the
loss masked to the completion, a document mask built once per batch and shared
across layers, and **two val numbers instead of one** — completion and prompt —
because masking moves them in opposite directions and one number hides half of
what SFT does. `select` defaults to the first key the task's `evaluate` returns,
so a task declares its own headline metric rather than everything agreeing on a
name.

`reverse.py` — `"cat>"` -> `"tac<eot>"`, encoded char by char so reversal is
position work and not spelling. `instruct.py` — TinyStoriesInstruct, fields in
and a story out; `Words:` appears in 66.6% of records and is the checkable part.

`tinystories.py` — the prepared corpus had no eot *token*: the boundary sat in
the stream as the three tokens that spell `<|endoftext|>`. Appending one
vocabulary entry and rewriting the boundary in place cost minutes and no
re-tokenization, because ids 0-4095 kept their meanings. 468.5M tokens, 2.12M
documents, vocab 4097. The assert that every `<|` is a boundary earned its keep
immediately — it caught 230 occurrences of a second spelling.

`basic.py` is now `pretrain` / `sft` / `sample` subcommands with `--resume`, and
a `TASKS` registry. `sft_results.ipynb` is the finetuning archive: every number
in it is read from the run's own `artifacts/logs/*.jsonl`, so re-running it
costs seconds rather than GPU hours.

## Measured, so it does not get re-derived

- **SFT works, and the scoreboard is not val loss.** Reverse, from the 123.6M
  FineWeb base: **exact match 0.40 @50 steps, 0.995 @100, 1.000 @250**, 1000
  steps in 4.6 min on the 4060 (block 256, batch 8 x accum 2, lr 3e-5, flex +
  document mask). Completion loss starts at **6.07, not ln(50259)=10.8** — that
  gap is what pretraining bought on a task the base has never seen. A 11.2M
  model from scratch needs lr 1e-3 and 200 steps for the same result. **Batch 16
  x window 256 OOMs on 8 GB**: the `[B,T,50259]` fp32 logits are ~785 MiB and the
  backward wants another, so halve the batch and accumulate.

- **Prompt loss is U-shaped, and only when the two halves are in tension.** The
  masked-out prompt loss falls, then climbs past where it started: from scratch
  **10.8 -> 6.5 @100 -> 11.7 @400**; from the pretrained base **8.73 -> 8.02 @50
  -> 9.82 @1000**, a much shallower dip because the base already models text.
  Leg one is learning the task's 27-token inventory, which helps every position
  including the untrained ones; leg two is specialisation making the model
  confidently wrong where it was never scored. The floor is ln(26) = 3.26 and
  neither run reaches it. Probed at the token level, the finetuned 123.6M model
  predicts **an early letter of the word** at every prompt position (p=0.8-0.97)
  while the small from-scratch one **echoes the current token** at p=1.00; mass
  on the 26 letters stays ~1.0, so this is certainty, not forgetting. The
  mechanism is untested — attention sinking onto position 0 is a guess.

  **On instruct there is no U-shape at all**: prompt loss drifts *down*, 4.32 ->
  4.09. The prompt is 18% of tokens rather than 50%, and it is English that
  shares vocabulary with the story, so specialising on the completion never makes
  it unpredictable.

- **On TinyStoriesInstruct, what SFT buys is the eot.** Same base, window 512,
  batch 4 x accum 4, lr 3e-5, 1000 steps in 13 min. **stop_rate 0.00 -> 0.80**,
  words present 0.15 -> 0.68, all-words 0.00 -> 0.25, stories 213 -> 154 words.
  The base is fluent and cannot stop: it echoes the Summary back and loops
  `Story:` headers until the budget runs out. 200k records is 51.8M tokens and
  the cache builds in 12s.

- **Greedy decoding measures the decoder, not just the weights.** Same instruct
  checkpoint, n=40: greedy words 0.550 / all 0.100 / **stop 0.725**; t=0.8
  p=0.95 **0.633 / 0.200 / 0.925**; t=1.0 0.567 / 0.175 / 0.775. Half the
  failures a greedy scoreboard counts belong to the decoder — the same base loops
  100% of the time at temperature 0 and 4% at 1.0. Also: `words_all` at n=20
  carries about **±0.1**, so a single eval's value is not a trend.

- **sdpa and flex are the same speed when there is no block_mask.** 4 x 200
  steps alternating in fresh processes: sdpa 163.9 / 164.0, flex 166.2 / 165.3
  ms/step, val identical to four decimals. flex falls through to the same
  `F.scaled_dot_product_attention(is_causal=True)` call, and weights transfer
  bit-identically (`max |sdpa - flex| = 0.0` from one state_dict). Pretrain with
  flex anyway, for structure and not speed: **sdpa bakes `rope_cos`/`rope_sin`
  buffers sized to `block_size` into every layer of the checkpoint, flex stores
  none**, so nothing caps a later PI/NTK/YaRN extension. One earlier timing of
  211.5 ms for sdpa was a cold-process artifact and was briefly read as flex
  being faster — a single timing from a different process is not a measurement.

- **TinyStories base, 27.0M params** (512/8/8, vocab 4097, rope, only 8%
  embedding — the opposite balance to a 50k-vocab model of the same size). 164
  ms/step at batch 32 x block 512 on the 4060, so 33000 steps = **541M tokens,
  1.14 epochs, 20 tok/param in ~1h40m**, of which 10 minutes is the 34 evals.

- **First real pretrain: 12L/768 on FineWeb-Edu.** 123.6M params, 983M tokens
  (0.99 epochs), 15000 steps x 65536 tokens, 4.4 h on a rented RTX PRO 4500
  Blackwell at $0.72/h — about $3.15. **val 3.327, bpc 1.046, ppl 27.9**,
  against the 29.9M smoke's 4.557 / 1.433 / 95.3.

  Throughput is **flat in batch size** — 62.4k tok/s at B=8, 16 and 24 alike, so
  the card saturates at B=8 and a bigger micro-batch only costs memory (+8.4 GB
  per +8, which is the `B x T x V` logits tensor; B=32 OOMs at 32 GB). 4.25x the
  4060's 14.7k tok/s on the same config.

  Prose is fluent and facts are not: it gets mitochondria right and states that
  photosynthesis produces carbon dioxide, in the same register. At ~8 tokens per
  parameter against Chinchilla's 20 it is compute-starved, not broken;
  `sample-10BT` holds 10x the tokens used.

- **Compression is linear in log2(vocab), at ~0.42 chars/token per doubling.**
  Truncating GPT-2's ranks at N gives exactly the vocabulary it had after N-256
  merges, so one table sweeps the whole range. On FineWeb-Edu: 1.99 chars/token
  at 512, 2.87 at 2k, 3.31 at 4k, 3.75 at 8k, 4.13 at 16k, 4.44 at 32k, **4.59
  at 50k**. The per-doubling gain is near constant (0.46, 0.42, 0.44, 0.43, 0.38,
  0.31), decaying only at the top — not the sharp diminishing return it looks
  like on a linear axis. But the rows cost linearly: 8k -> 50k is 6.1x the
  embedding for 22% fewer tokens. A smaller vocabulary is therefore not a free
  saving either — it shrinks the embedding and lengthens every sequence.

- **Adopt the table, do not retrain it.** `train()` recounts every pair over the
  whole corpus once per merge: 14.7s / 40.4s / 90.4s for vocab 512 / 1024 / 2048
  on **1 MB**. Extrapolated to 50257 that is ~42 minutes for one megabyte,
  against the hundreds of GB GPT-2's table was learned on. The unique-chunk
  `Counter` is the only reason even the small ones finish.

- **The shard build is download-bound.** Streaming steady state is 1.12M tok/s;
  one tiktoken thread encodes 5.3M. So `num_threads=1` — which also sidesteps
  tiktoken's pool and `datasets`' parquet reader tearing each other down at
  interpreter exit (SIGABRT or a hang, at random, *after* every byte is on disk;
  the build ends with `os._exit(0)` for that reason). Four parallel connections
  summed to 3.25 MB/s, the same as one: the link is saturated, so parallel
  downloads buy nothing either.

- **Measure over a window longer than the startup cost.** The first throughput
  number for that stream was 0.14M tok/s and 8x wrong — the window was dominated
  by 15.2s of one-time stream startup latency. It nearly went into a comment as
  the justification for a design choice that happens to be right anyway.

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

Padding and masking shipped as `mask.py` plus the flex `BlockMask` path; SFT as
`task.py` + `sft.py`; LoRA as `lora.py`; the two-task mix as `joint.py`. Results
are in `src/video/sft_results.ipynb` and `lora_results.ipynb`. What follows
assumes the 27M TinyStories base.

**1. SFT and LoRA — done.** Full SFT and a rank-8 adapter on qkv+proj
(196,608 params, **0.72%**) both learn instruct. Once *both* arms get their own
lr sweep, LoRA is ~4% worse on completion loss and drifts ~15% less on
TinyStories val; compared at a single lr the gap looks twice that, which is the
trap. Reverse is the opposite — **no lr lets LoRA match full SFT**, and at 3e-3
it costs the base **+3.80 nats against instruct's +0.115**. An lr does not
transfer between tasks on one base; sweep per task, on completion loss *and* a
base-loss check, never on the task scoreboard alone. Two other findings worth
keeping: the adapter's loss floor is set by **rank, not lr** (0.1175 vs 0.1179
across a 3x lr gap), and a generated scoreboard at n=40 carries ±0.1, enough to
keep the wrong checkpoint — hence `--select comp`.

**2. Joint LoRA — done, and it clears the old track.** One rank-8 adapter on the
67/33 instruct+reverse mix reaches **reverse exact_match 1.000** while its
instruct scores stay within one standard error of the instruct-only adapter:
reverse comes essentially free. The old track got 0.969 with the same adapter
size on a same-size base. `joint.py` sets the mix by **start count**, because
windows open at starts, and it does so by dropping starts rather than resizing a
task — nothing in the `Task` protocol exposes its own size.

**3. DPO — fix the reward first.** `Instruct.reward` scores only the `Words:`
line, so it ranks a plot-faithful story that misses one word **below** a
word-stuffed one that ignores the Summary entirely. As an SFT scoreboard that
costs nothing, because nothing optimises it. As a DPO/GRPO objective it is
disqualifying: the model chases what is measured, and this measures three words.

  Field coverage over 3000 val records: `Summary` 100%, `Words` **61.4%**,
  `Features` 49.2%, `Random sentence` 29.6%. Two free string-match checks the
  reward currently ignores — the `Random sentence` appears **verbatim** in its
  story **100%** of the time, and `Features: Dialogue` implies a quote in the
  story 90.8% of the time. The Summary is **not** checkable by string match at
  all, which is precisely why the model learned to ignore it.

  So: widen `reward` to a composite over the checkable fields before building
  the rung on it. DPO goes on instruct and **not** on reverse, which is solved
  to 1.000 and has no preference to express. Preference pairs stay free — sample
  K, rank, take best against worst.

  **Reward widened** (2026-09-22): `instruct.checks` returns every check a
  prompt supports — `words` (fraction), `sentence` (Random sentence verbatim,
  case-folded), `dialogue` (a `"` when Features asks for Dialogue), `stop` — and
  `reward` is `stop x mean(content checks)`, or just `stop` for the 20% of
  prompts that are Summary-only. The stop gate kills the run-to-budget,
  list-every-word policy. Gold ceiling over 3000 val records: words 0.984,
  sentence 1.000, dialogue 0.908. Open cost: 11.4% of gold stories exceed the
  256-token eval budget and so would fail the gate (3.7% exceed 384).

  **Summary, pooled with Words, rarity-weighted** (same day). `words` is now one
  set: the `Words:` line plus the Summary's non-stopword words, stemmed
  (wants/wanted, cats/cat; car does not match careful). Each word counts by its
  idf over 20k train stories (lily 1.39, dog 2.49, stitches 6.41), so missing
  the plot costs more than missing a name. Zeroed if the story shares a 12-word
  run with the Summary (pasting it in otherwise wins; fires on 0.6% of gold).
  `reward` = points / points available: stop, sentence, dialogue 1 each, words
  2, and a story that never stops scores 0 (the gate stays). Why the weighting: two SFT samples on
  the waffle prompt (one drops "bitten/hospital/stitches", the other "toy car/
  meet") both hit 9/12 words and tied; weighted, **0.720 vs 0.849**, the right
  way round. Gold words 0.767 vs **0.052** against a random story; gold reward
  0.860, only 5.7% exactly 1.0 — read it as a ranking, not a pass rate. Still
  words, not plot.

  **After the DPO loop works: replace it with PMI** — `log p(story | prompt) -
  log p(story | prompt without Summary)` per story token, under the frozen SFT
  model (the DPO reference, so no extra model). Picks own summary over a
  swapped one 100/100, no ties within a group, Spearman 0.08 with the string
  checks (independent signal). Needs the same copy guard: pasting the summary
  scores PMI 1.105 vs gold 0.047.

**4. A code corpus — a search, not a build.** The intended analogue,
`nampdn-ai/tiny-codes`, is **gated**. Of what was verified to load:
`codeparrot-clean` is raw GitHub Python (Django views, not tiny),
`flytech/python-codes-25k` is toy interactive scripts that cannot be unit-tested,
and MBPP is 974 problems — eval-sized, not a pretraining corpus. There is no
drop-in "TinyStories for code", so this phase begins with finding or assembling
one, and may conclude that 27M is the wrong scale for it. **Gate: measure pass@1
before building any of the RL rung on it.** If a from-scratch code model scores
~0, there is no gradient for DPO or GRPO and the phase ends there, which is a
finding rather than a failure.

**5. Competitive programming on a real model: sandbox, then GRPO.** The data is
better than expected — `deepmind/code_contests` loads, is **61% Codeforces**, and
ships `description`, `cf_rating`, `cf_index`, `cf_tags`, 80-100 **generated tests
per problem**, and `incorrect_solutions` on 79% of problems, which are free
preference pairs. From a 1500-row stream — 891 Codeforces, 101 of them rated
<= 1000 — extrapolating over the full set gives **roughly 7,700 Codeforces
problems and ~900 rated <= 1000**, statements averaging ~430 tokens. The total
row count is taken on trust, so treat both figures as order-of-magnitude. Python, not
C++: solution counts are equal (42k vs 40k in a 300-problem sample) and C++ adds
a compile step, which is a failure mode a small model will hit constantly.

  This is a track change rather than a rung — a different tokenizer, weights that
  are not ours, and ~2k context that the from-scratch models do not have. The
  lesson moves from owning the weights to owning the algorithm. Build in order:

  - **the sandbox** — subprocess, timeout, no network, memory cap. It is the
    reward function for everything in phases 4 and 5, so it lands first.
  - **pass@1 on Qwen2.5-Coder 1.5B and 7B**, over the rated <= 1000 band and over
    MBPP. Pick the band where it falls in 20-50%: higher and there is nothing to
    amplify, zero and there is no gradient. Contamination is real here — those
    solutions are on GitHub — so hold out recent problems as a clean test set.
  - **GRPO** on the band that measurement chooses. DPO is also available without
    sampling, straight from `incorrect_solutions`.

**6. Chat and QA.** Multi-turn masking: `doc_ids` generalises from eot boundaries
to scoring only the assistant's turns, and `<|im_start|>`/`<|im_end|>` are
already in the 50259 vocab because the tokenizer episode minted them for this.

## Worth covering, unscheduled

- **MoE** — routing, top-k experts, load-balancing loss. Self-contained.
- **PI / NTK / YaRN context extension** — **built and measured in
  `src/minimal_experiments/rope6.py` (NTK) and `rope7.py` (PI, NTK, YaRN in one
  table)**; integration into `src/video/` is pending, still as a RoPE sequel. All
  three live entirely in the angle computation — attention, mask, KV cache and
  `generate` are untouched — which is what makes them a small episode. The whole
  family is three lines of `theta` shrink per pair, fastest to slowest, at x8:

      pi    8.00 8.00 8.00 8.00 8.00 8.00 8.00 8.00
      ntk   1.00 1.35 1.81 2.44 3.28 4.42 5.94 8.00
      yarn  1.00 1.00 1.80 4.13 7.00 8.00 8.00 8.00

  PI is the control and it earns its place: slowing the fast pairs costs **1.92
  -> 2.95** inside the trained range at x2 alone, against NTK's 1.92 -> 2.09 at
  x8 for a better far field. Frequency-selectivity is worth ~1.4 nats, measured,
  not asserted.

  The open question is that all of this is **zero-shot**, which is the case only
  NTK was designed for — PI and YaRN are both published *with* a fine-tune at the
  extended length, and YaRN's pinning of well-covered pairs exists precisely to
  make that fine-tune cheap and non-destructive. A frozen model sees the cost and
  none of the benefit, and YaRN duly loses to NTK here. The experiment that
  settles it: **fine-tune ~100 steps at 4096 with each scaling on**, then re-run
  the per-position table. Worth doing after BPE+FineWeb rather than on the char
  model, where long context means something.

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
