# ops magic — roadmap

Libraries that change how the code *reads*, not just what it does. Walkthroughs live in
[rough.ipynb](rough.ipynb), one small piece at a time.

Status: `[x]` done · `[ ]` todo

---

## einops

The core idea: shape ops written as a sentence about **named axes**.
`rearrange` keeps elements, `reduce` drops them, `repeat` adds them.

- [x] **part 1 — `rearrange`**
  `lhs -> rhs`, reorder axes, merge with `( )` on the right, split with `( )` on the left.
  Key rule: order inside parentheses matters — leftmost name is the slow/outer axis.
  Silent-but-wrong `(h d)` vs `(d h)` is the bug to watch for.
- [x] **part 2 — `reduce` / `repeat`**
  Collapse an axis with `mean/sum/max/min/prod`; `1` on the right = `keepdim=True`.
  Split-then-reduce *is* pooling. `repeat` tiles — but prefer broadcasting when you can.
- [ ] **part 3 — `einsum`**
  `einops.einsum(q, k, "b h t d, b h s d -> b h t s")` — einsum with real names instead of
  `bhtd,bhsd->bhts`. Attention scores, weighted sums, batched outer products.
- [ ] **part 4 — `pack` / `unpack`**
  Concatenate ragged things along one axis and get back the recipe to split them again.
  Prepending a CLS token, packing q/k/v into one tensor, then unpacking.
- [ ] **part 5 — layers + `...`**
  `einops.layers.torch.Rearrange` / `Reduce` as `nn.Module`s inside `nn.Sequential`.
  Ellipsis for "whatever leading batch dims exist". Brief look at `EinMix`.
- [ ] **capstone** — rewrite multi-head attention from `attention.ipynb` in einops, diff the
  two side by side.

## jaxtyping

Same named-axis idea, moved into function signatures — and actually enforced at runtime.

- [ ] `Float[Tensor, "b h t d"]` annotations, consistency checking across arguments
- [ ] hooking up `beartype` / `typeguard` so violations raise instead of decorate
- [ ] where it pays off: module boundaries, not every helper

## omegaconf

Kills the 200-line `argparse` block in a training script.

- [ ] YAML → dot access (`cfg.model.n_heads`)
- [ ] interpolation: `${base_lr}`, `${model.name}_${data.split}`
- [ ] merging base config + CLI overrides; `OmegaConf.structured` against a dataclass
- [ ] when to graduate to Hydra (multirun sweeps, per-run output dirs)

## later / maybe

- [ ] **safetensors** — checkpoint format that's fast and can't execute code on load.
      Basically a drop-in replacement for `torch.save`.
- [ ] **tensordict** — nested dicts of tensors you can index/reshape/`.to(device)` as one
      object. Mostly an RL thing: replay buffers, rollout batches.
- [ ] **tyro** — dataclass or function signature → typed CLI. Lighter than Hydra when there's
      no config composition to do.
- [ ] **wandb** — the non-obvious parts: `wandb.watch(model)` for gradient histograms,
      artifacts for checkpoint versioning.
- [ ] **einx** — einops-style notation generalized further; only worth it if einops starts
      feeling limiting.
