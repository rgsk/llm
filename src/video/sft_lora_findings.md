# The SFT + LoRA track, in plain terms

## The setup

We have a small model — 27 million parameters — that was trained on TinyStories, a corpus of simple children's stories written with a 4-year-old's vocabulary. It writes fluent little stories and knows when to stop. That's all it knows.

Finetuning is the step where you take that general model and teach it to do a _specific_ job. We used two jobs, deliberately very different:

**Reverse** is a toy: given `cat>`, write `tac`. It exists as a unit test for the training loop. If the loop works, a model solves this; if the score is low, you have a bug, not a research finding.

**Instruct** is the real job. You give the model a set of fields — some required words, a plot summary, sometimes a sentence that must appear verbatim — and it writes a story satisfying them. This is a miniature version of what instruction-tuning does to a real chatbot.

Then we compared two ways of doing that finetuning: **full finetuning**, where all 27M weights change, and **LoRA**, where you freeze the whole model and train a tiny add-on instead — in our case 196,608 numbers, **0.72%** of the model.

## Part 1 — does the loop work? (reverse)

It works. The model goes from 0% to **100% correct by step 300**, in 113 seconds.

The first genuinely interesting thing showed up here, and it's about how to read a loss number at all.

Loss is measured in nats, and the meaningless baseline — pure guessing — is `ln(vocabulary size)`. Our vocabulary is 4097 words, so guessing scores **8.32**. Our model starts at **8.11** on the reversal task. That's only 0.21 better than guessing. Pretraining on children's stories bought us almost _nothing_ for spelling words backwards.

Compare: a bigger model trained on web text starts 4.76 nats below its own guessing baseline on the same task. It has seen spelling, acrostics, letter-by-letter text. TinyStories has seen none of that.

**The aha:** you cannot compare loss numbers across models with different vocabularies. 8.11 and 6.07 look miles apart, but relative to their own baselines, 8.11 is nearly useless knowledge and 6.07 is a big head start. Always read a starting loss against `ln(V)`.

And a stranger one. The prompt half — the random letters `qxzv` — starts at **9.70**, which is _above_ the 8.32 guessing line. The model is **worse than random** there. A model that has only ever read children's stories is confidently certain that `qxzv` cannot happen. Its knowledge is actively harmful on that input.

## Part 2 — the real job (instruct)

Full finetuning works well. Completion loss drops from 1.5043 to **1.0945**. The fraction of stories containing _every_ required word goes from 0% to **42%**. Story length grows from 56 words to 153, matching the corpus.

**The aha here is what finetuning bought.** On a web-text base, the big win from instruction-tuning is teaching the model to _stop_ — those models ramble until they hit the token budget. Our TinyStories base already stops 58% of the time, because every story it ever read ended. So finetuning didn't buy termination; it bought **format and constraint-following**. Same algorithm, same data, completely different lesson, purely because the starting model was different.

There's also a cost nobody advertises. The loss on the _prompt_ tokens — which are masked out and never trained on — gets **worse**. The model specialises into a story-writing machine and becomes confidently wrong about everything else. That's forgetting, visible in miniature.

## Part 3 — LoRA

LoRA freezes the model and trains a small correction beside it. Mechanically: instead of changing the weight matrix `W`, you learn two thin matrices `A` and `B` and compute `W·x + (B·A)·x`. `B` starts at zero, so at step 0 the model is bit-for-bit identical to the original. `A` starts random, so `B` has something to learn from.

That last detail matters more than it looks. At step 0, `A`'s gradient is exactly **zero** — the gradient reaches `A` through `B`, and `B` is zeros. So only `B` moves on the first step, and `A` only starts learning once `B` is nonzero. If you initialised _both_ at zero, neither would ever move. The test that pins this is the clearest explanation of the design in the whole file.

Now the findings, which is where it got interesting.

### Aha 1 — the learning rate does not transfer between tasks

The learning rate is how big a step you take. Too small, nothing learns; too large, you wreck the model.

Same base, same adapter, same 0.72% of weights, same learning rate of 3e-3:

- on **instruct**, it costs the base model **0.115 nats** of its original ability
- on **reverse**, it costs **3.80 nats** — the model is essentially destroyed

**33× different, from the task alone.**

The reason makes sense in hindsight. Instruct is a _format_ the model is already most of the way to — it can already write stories, it just needs to learn the field layout. Reverse is a genuinely new _mechanism_ — reflecting positions — that the base has no prior for. That mechanism has to be forced into the attention weights, which is exactly where the adapter lives. So reverse demands a violent change and gets one.

The practical rule: sweep the learning rate per task. Never inherit one, not even from the same model on a different job.

### Aha 2 — fewer parameters is not a smaller change

This is the one that genuinely surprised me. LoRA is sold as the gentle option — you're only touching 0.72% of the model, so surely you can't do much damage.

You absolutely can. Those 196,608 numbers sit on the attention weights in _every single layer_. At a hot learning rate they move attention as far as they like. The reverse adapter at 3e-3 scored a perfect 1.000 on its task while pushing the base model's loss from 1.36 to **5.16** — a model that can reverse strings and has forgotten how to write.

Parameter count tells you about _memory and compute_. It tells you nothing about how much damage a step can do.

### Aha 3 — the scoreboard has a ceiling and the loss doesn't

On reverse, full finetuning and LoRA both hit `exact_match` **1.000**. By that metric they're identical models.

They aren't. Full finetuning's loss goes to **0.0000**; LoRA's bottoms out at **0.1175** and stops. That's the rank-8 bottleneck — the point where a low-rank correction runs out of room.

And here's the confirmation: at a 3× different learning rate, LoRA floors at **0.1179**. Essentially the same number. **The floor is set by the rank, not the learning rate** — no amount of tuning moves it.

A task scored only by its own scoreboard would have reported these as the same model. The loss was the only thing that could see the difference.

### Aha 4 — comparing at one setting is not a comparison

This is my own mistake, and it's the most useful thing in the track.

I measured full finetuning at one learning rate and LoRA at a properly swept one, and reported "LoRA forgets **4.6× less**." It looked like a strong result for LoRA.

Then I swept both. The learning rate I'd given full finetuning was its best setting for _story quality_ and its worst for _gentleness_. When full finetuning gets to pick a gentler setting, the gap collapses: LoRA forgets about **15% less**, not 4.6×, and full finetuning actually wins on loss.

The real relationship is a trade curve for each method, and which one "wins" depends entirely on where you stand on each curve. Comparing two methods at one setting each tells you almost nothing — and the direction of the error is always flattering to whichever one you tuned.

### Aha 5 — a noisy metric will silently keep the wrong model

The training loop saves whichever checkpoint scores best. It was scoring on "fraction of stories with all required words," measured on 40 samples — which swings about ±0.1 run to run.

So it kept **step 600** because that metric randomly peaked there, and threw away step 1800, which was genuinely better on loss. Both arms of the comparison were affected. We were comparing two models neither of which was the best its run produced.

The fix is one word: select on completion loss, which is measured the same way every time. `--select comp`.

Worth noting how this hid: nothing errors, nothing looks wrong, the run completes and writes a file. You only catch it by asking "which step did it actually save, and was that the best one?"

## Part 4 — both jobs at once

The last experiment: one adapter, trained on a mix of 67% instruct and 33% reverse.

It works, and better than expected. **One rank-8 adapter — 196,608 numbers — solves reverse perfectly (1.000) while doing instruct within noise of an adapter trained on instruct alone.** Reverse comes essentially free.

You can see it in the output. The same weights produce:

```
cat>    -> tac
zebra>  -> arbez
puzzle> -> elzzup
```

and terminated, coherent children's stories.

The mixing detail is neat: training windows open at example boundaries, so the ratio between two datasets is controlled by _how many boundaries_ each contributes, not how many tokens. To hit 67/33 we just drop boundaries from the bigger one. Its tokens stay in the stream as context; they're simply never used as a starting point.

Also: the mix tolerated a **hotter** learning rate than either task alone. Not sure why. Possibly the two objectives pulling in different directions act as a regulariser. Untested, and I'd want to measure before believing it.

## Part 5 — the reward is broken, and that blocks what's next

You spotted this one. A sample that followed the plot exactly — waffle, dog bite, hospital, stitches, 4 of 5 plot elements — scored **0.67**, while a story that ignored the plot entirely but contained all three required words scored **1.00**.

Our reward only checks whether the required words appear. It cannot tell a faithful story from a word-stuffed one.

Right now that's harmless: it's a report card, and nothing optimises it. It becomes **disqualifying** at the next stage, where we train the model to make that number go up. The model will chase exactly what we measure, so it would learn to stuff three words into anything and stop trying to follow the plot. The score would improve while the stories got worse.

What's checkable, measured over 3000 records: the `Random sentence` field appears **verbatim** in its story **100%** of the time — a perfect target we're currently ignoring. `Features: Dialogue` implies a quote in the story 90.8% of the time. The **Summary cannot be checked by string matching at all** — which is precisely why the model learned to ignore it.

So the reward needs widening before DPO, and now we know the fix is cheap.

## A side finding: the vocabulary is an odd number

Our vocabulary is 4097 — 4096 merges plus one end-of-text token. That single extra column costs **1.6% of every training step**, and it's entirely one matrix multiply: the final projection from hidden state to vocabulary runs **34.5% slower** at 4097 than at 4096, because it falls off the GPU's aligned fast path.

Padding to 4104 (a multiple of 8) recovers almost all of it, for 3,584 dead parameters that the model just learns to ignore. On the 100-minute pretraining run that's about 1.6 minutes.

I checked the mechanism rather than assuming it — isolated the matmul, confirmed the delta accounts for the whole step's slowdown.

## The through-line

Almost every real finding here was a **measurement correcting an assumption**, and three of them corrected assumptions I had stated confidently in writing:

- LoRA is the gentle option → only at a matched learning rate, and the margin is small
- fewer parameters means a smaller change → no, it means less memory
- a good score means a good model → the scoreboard saturates; the loss is what sees the ceiling
- one lr transfers across tasks → 33× wrong
- the run saved the best model → it saved whatever a ±0.1 metric peaked on

The pattern: whenever a comparison looked clean and favourable, it was because only one side had been tuned. The fix is boring and works — sweep both arms, select on a stable metric, and check the thing you're not optimising.
