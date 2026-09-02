# Roadmap

Updated 2026-09-02, after the KV cache landed.

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

The old "three genuinely big remaining pieces" list is closed except for data:
SDPA, RMSNorm, SwiGLU, weight tying, and the KV cache all shipped. Optimizer
hygiene — the item that sat pending longest — is done too.

## Next, in order

**1. Sinusoidal positions.** One short file, and it exists to earn one idea:
positions as frequencies. It is the bridge to RoPE, not a destination.

**2. RoPE.** Do this now, while the cache is fresh. RoPE does not just replace
learned positions — it changes what the cache holds. Rotated keys go in, and
scores come to depend on `i - j`, which is what makes *cropping* the cache
correct. Sliding-window attention and attention sinks become reachable in the
same breath. Deferred, this costs a re-explanation of the whole cache.

Payoff test is already written: the reverse-string probe below.

**3. GQA / MQA.** The sequel to the cache, and the highest-value item that was
not on the original list. The cache costs `2 · n_layer · B · T · E` in memory;
sharing k/v across query heads cuts it 4-8x. Small diff to the two attention
files. Only lands as a lesson *after* someone has felt the memory cost, which is
why it goes here and not earlier.

**4. BPE tokenizer + FineWeb-Edu.** `src/tokenizer.py` works; port and clean it
into `video/`. Byte-level, count pairs, merge, repeat. This is also where the
chat special tokens get minted, so it has to precede SFT. Then real data at a
size that does not fit in RAM.

**5. Padding and attention masks.** *Currently missing everywhere in
`src/video/`* — checked, not assumed. Fine for pretraining on contiguous shards,
a blocker for everything after it: SFT needs the loss masked over prompt tokens
(`ignore_index`), and batched generation needs left-padding plus a real mask,
since `generate` assumes every row shares a prompt length. Small file. It has to
land before SFT, not during.

**6. SFT + LoRA.** Cleaner versions of `src/sft.py` and `src/lora.py`. Chat
template, loss masking, then LoRA as the parameter-efficient variant. This is
where it stops being a continuation engine.

**7. RL.** **DPO first** — a loss function over a frozen reference model, no
reward model, no rollouts, no value head, which fits the file-plus-test format.
PPO/GRPO after, and budget three episodes: sampling loop, advantage estimation,
KL control. It is the first thing in the series that can silently fail to learn.

## Worth covering, unscheduled

- **Chat templates and special tokens** — cheap, high payoff, pairs with SFT.
  The difference between a continuer and an assistant is mostly a format contract.
- **Speculative decoding** — the finale of the inference thread (cache → GQA →
  spec decode). Draft proposes k, target verifies in one forward. Lossless, which
  is the surprising part.
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

Re-run both after swapping in RoPE. RoPE removes the untrained-absolute-rows
problem and makes scores depend on `t-p`, so the held-out number may improve —
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
