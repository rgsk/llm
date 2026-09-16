Right — this was five episodes in a row on the same axis, and they're easy to blur together. Here's the whole arc in plain terms.

## The thing everything is fighting

When the model generates text one token at a time, every new token has to look back at every token before it. To avoid recomputing all that from scratch each step, you save each token's key/value pair in a **KV cache**.

That fixes speed and creates the real problem: **the cache grows with every single token.** A long chat means gigabytes per layer. From here on, every episode is about making that cache smaller.

There are two different ways to attack it:

- **GQA/MQA** — shrink it by a constant factor (fewer key/value heads). Divides by 8. Still grows.
- **The three rungs** — stop it growing with the run _at all_. This is the part you're asking about.

## Rung 1 — sliding window: a **mask**

"Each token only looks at the last W tokens." You hide the far-away columns. Done.

The thing people always get wrong, and why it got its own episode: **this saves zero memory.** The cache still holds every key, the kernel still walks past every one of them — they just contribute nothing to the answer. It changes _what the model computes_, not what it stores.

The nice surprise: a window doesn't cap what the _model_ can see. Token 100 in layer 1 sees 97–100. In layer 2 it sees layer-1 outputs at 97–100 — and each of those already looked back to 94. Stack it up and the reach is `L*(W-1)+1`. Mistral ships a 4096 window over 32 layers, which reaches \~131k tokens.

## Rung 2 — ring buffer: **storage** that wraps

Now actually stop storing what you stopped looking at. The cache is exactly `W` wide, forever. Position `p` lives in slot `p % W`. Writing token 100 into a 4-slot ring overwrites token 96 — which just fell out of the window anyway.

Two things fell out of this one:

**Slot order stops being position order.** After wrapping, your slots hold `[100, 97, 98, 99]`. And it turns out _that's completely fine_ — softmax doesn't care what order the keys are in. Shuffle the keys and values the same way and the answer is identical. So nothing ever needs un-shuffling; only the _mask_ needs to know which position is sitting in which slot.

**Decode stops needing a mask entirely.** Everything the ring still holds is inside the window by construction. So rung 2 _removes_ from the hot path the mask that rung 1 put into it.

And the punchline: throwing keys away gives **bit-for-bit the same answer** as keeping them and masking. It's not an approximation of the windowed model — it _is_ the windowed model, stored honestly.

## Why RoPE had to land first

Worth saying, because it's the hidden dependency. With old-style absolute positions, a key is stamped "I am token 5000." If you start throwing keys away and renumbering the survivors, that stamp becomes a lie.

RoPE makes a score depend on the **distance** between query and key, not on where either one sits. That's what makes eviction and renumbering legal at all. Rung 1 doesn't need it; rungs 2 and 3 can't exist without it.

## Rung 3 — attention sinks: a **policy**

The pure ring has a bug: it eventually throws away token 0. And models turn out to care enormously about the first few tokens — not for their _content_, but as a dump. Softmax weights must add to 1, so when a head has nothing relevant to look at, it parks that weight on the first few tokens. Evict them and all that weight gets redistributed onto real keys, poisoning everything downstream.

Fix: **pin the first S slots. Never evict them. Ring over the rest.** Cache = `S + W`.

_(That dump explanation is the paper's finding — we have not measured it in this repo, and can't on an untrained model. That's the notebook probe I flagged.)_

### The part that made it a real episode

If you generate forever, your position counter hits 10000, 10001... but the RoPE table only has `block_size` rows. **You run out of positions.**

StreamingLLM's fix is lovely: stop numbering keys by where they were in the text. Number them by their **rank inside the cache**.

```
cache actually holds positions:  [0, 1, 5000, 5001, 5002, 5003]
what we tell the model they are: [0, 1,    2,    3,    4,    5]
```

Ranks can never exceed the cache size, so the table never runs out. **`block_size` stops being a limit on how long you can generate and becomes a limit on how big your cache is.**

What's preserved and what's a lie: the gaps _inside the window_ are exact — 5000–5003 becomes 2–5, same spacing. The only lie is the jump across the gap: a sink that's really 5003 tokens back gets told it's 5 back. Which is fine, because the model is using it as a dump, not reading its position.

**The cost:** you can't rotate a key once when you write it anymore, because its rank changes every time something gets evicted. So keys go into the cache raw and get rotated **every single step, the whole cache**. One rotation per step became `S + W` per step. That's the bill, and it's the point of the episode.

## The whole thing in one table

| Rung       | What it is                                             | What it changes  | What it costs                     |
| ---------- | ------------------------------------------------------ | ---------------- | --------------------------------- |
| 1 — window | what you **look at** (a mask)                          | the math         | no memory saved                   |
| 2 — ring   | what you **store**                                     | memory goes flat | can't rewind the cache            |
| 3 — sinks  | what you **never throw away** (a policy) + renumbering | generate forever | rotate the whole cache every step |

They were deliberately three separate episodes because they're genuinely independent, and nearly everyone conflates them — "sliding window attention" gets used to mean all three at once.
