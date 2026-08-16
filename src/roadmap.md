Here's the map after those two steps.

## Immediately after scaling — training hygiene

The 10M-param run takes real minutes on your 4060, which changes what you need:

- **Checkpointing** — `torch.save` model + optimizer state periodically. A 40-minute run you can't resume is a bad time.
- **Mixed precision** — `torch.autocast` with `bfloat16`. Your 4060 supports it natively; roughly 2x faster and halves memory. Nearly free.
- **`torch.compile(model)`** — one line, another sizable speedup.
- **LR warmup + cosine decay**, and **gradient clipping** at 1.0. At `n_embed=384` a constant LR leaves performance on the table and occasionally blows up.
- **A config dataclass.** You'll have ~12 globals by then. This is the point where they stop being manageable.

Expect **~1.48 val loss** on Shakespeare at that size. Output becomes recognizably English-shaped — real words, plausible dialogue structure, no meaning.

## Then the three genuinely big remaining pieces

**1\. BPE tokenizer, from scratch.** This is the other half of "how LLMs are built" and you haven't touched it. Byte-level, count adjacent pairs, merge the most frequent, repeat. ~150 lines. Take vocab to ~4096 and sequences get ~4x shorter, so the same `block_size` covers 4x more text. It's also where you'll finally understand tokenization pathologies — why models can't count letters in a word.

**2\. Real data.** TinyStories or a FineWeb-Edu sample. Shakespeare is 1MB; you'll be memorizing it. This brings a new set of problems that are their own lesson: data doesn't fit in RAM, so you need `np.memmap` over pre-tokenized `uint16` shards, and a separate tokenize-once script.

**3\. Modern architecture.** Your model is 2017 GPT. What actually changed since:

- `F.scaled_dot_product_attention` — flash attention, one call replacing your whole `Head.forward`, much faster and less memory
- **RoPE** instead of learned position embeddings — relative positions, extrapolates past training length
- **RMSNorm** instead of LayerNorm — cheaper, works as well
- **SwiGLU** instead of ReLU MLP
- **Weight tying** between the token embedding and `lm_head`
- **KV cache** — your `generate` currently recomputes all previous tokens every step; caching makes it linear instead of quadratic

Each is a small, independent swap. Do them one at a time and measure.

## After that

Sampling controls (temperature, top-k, top-p), gradient accumulation for larger effective batches, then post-training: supervised finetuning on an instruction dataset, LoRA, and optionally DPO. That's the step where it stops being a text continuation engine and starts behaving like an assistant.

## Pending

### optimizer hygiene

The one item still outstanding from early on is optimizer hygiene — you're running AdamW with default wd=0.01 applied to norms and embeddings alike, no gradient clipping, and betas (0.9, 0.999). Param groups with decay only on 2D params, clip*grad_norm*(1.0), betas (0.9, 0.95) is a single run and likely a real gain. It's the cheapest unclaimed win on the board.

### RoPE length-extrapolation test

`sft.ipynb` has a reverse-string task (`hello>olleh`) that measures length generalization directly. Trained on word lengths 3–8 with `block_size` sized for 10, the learned-position model scores **1.0 on `evaluate()` and 0.0 on `evaluate(lmin=9, lmax=10)`** — it emits correct chunks of the reversal at the wrong offsets.

Why it fails: position `t` must attend to `2l-1-t`, a *reflection*, so the required offset `2l-2t-1` depends on both `t` and `l` — no single rule covers it. But valid `(l, t)` pairs for l∈3–8 number only 33, small enough for a 32k-param model to memorize as a lookup. Lengths 9–10 add 19 unseen pairs, and position-embedding rows 9–19 only ever held pad, so they sit near init with no learned relation to rows 3–8. Nothing to interpolate from.

Re-run both after swapping in RoPE. RoPE removes the untrained-absolute-rows problem and makes scores depend on `t-p`, so the held-out number may improve — but it does not hand the model a reflection, and length generalization on reverse-and-copy is known-hard for positional encoding alone. **Treat a flat 0.0 as a fact about the task, not proof the RoPE code is broken.** Verify RoPE separately on val loss; use this as a bonus probe.
