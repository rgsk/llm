# scratch roadmap

Pure-Python (stdlib-only) rebuild of the `src/video` basic GPT slice, as of commit
`0d9ae23`. torch appears only in tests, as the oracle.

**Target:** same result as `video/basic.py` — train on TinyStories, print loss/bpc,
sample text.

## Done

- strided `Tensor`: flat storage, shape, strides, storage offset, `tolist`,
  `transpose`, `reshape`, `contiguous`, `expand`
- broadcasting, `+ - * /`, batched `@`
- autograd: `topo`, `backward`, broadcast-aware backward, matmul backward,
  differentiable views
- A1 `sum` / `mean(dim, keepdim)` (t18)
- A2 `_unop`: `neg`, `exp`, `log`, `sqrt`, `relu`; `/` (t19)
- A3 `masked_fill` (t20)
- A4 `cat` (t21)
- A5 int-id lookup `weight[ids]`, negative ids, scatter-add backward (t22)
- slicing: `x[1]`, `x[:, -1]`, `x[::2]` as views, backward into the base (t23)

## Not needed at this scope

- backward through `max` in softmax: softmax is shift-invariant, so treat it as a
  constant

## A. Engine (remaining)

6. `no_grad`

## B. nn

`Parameter`, `Module` (dotted `named_parameters`, `train/eval`, `apply`,
`zero_grad`, `state_dict`), `ModuleList`, `Sequential`, `Linear`, `ResidualProj`,
`Embedding`, `ReLU`, `LayerNorm`

## C. Functional

`softmax`, `log_softmax`, `cross_entropy`. Loss at init ≈ ln 4096 = 8.318.

## D. Model

`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPT` (init + weight tying).
Integration test vs a torch mirror with identical state_dict keys: logits, loss,
every grad.

## E. Speed

Contiguous fast paths for matmul and `_binop`. Re-run the suite, time one step.

## F. Training

`AdamW` + `decay_groups` (wd 0.1, betas 0.9/0.95), `clip_grad_norm`, `get_lr`,
data loading (`mmap` + `random.Random`), `estimate_loss` / bpc, train loop.
One AdamW step vs `torch.optim.AdamW`.

## G. Finish

`generate` (greedy, temperature, top-k/top-p), JSON checkpoint, BPE decode
(merges JSON; prompts are raw byte ids, `"\n"` = 10), `basic.py`.

## Out of scope

dropout, buffers, RMSNorm, fused qkv / SDPA, gated FFN, amp / compile, KV cache,
RoPE, GQA, quantization, loading torch `.pt` files, the `regex` BPE encoder.

## Cost (measured 2026-09-13)

| path | rate |
|---|---|
| matmul via `_offset` | 0.69 µs / mult-add fwd |
| matmul flat-list ikj | 62 ns (~11×) |
| `_binop` via `_offset` | 1.1–1.8 µs / element fwd+bwd |
| flat `zip` | 23 ns fwd (~40×) |

Estimated per step (confirm on the assembled model):

| config | now | fast paths | 300 steps |
|---|---|---|---|
| video small_cfg (V4096 T32 E64 L3 B32) | ~1000 s | ~90 s | ~7.5 h |
| **tiny (V4096 T16 E32 nh2 L2 B8)** | ~49 s | **~4 s** | **~21 min** |

`lm_head` is ~80% of matmul work at V=4096.
