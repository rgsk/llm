```-
                              ms/step   vs fp32
fp32,  old attn                 275.5    1.00x
fp32 + tf32,  old attn          235.1    1.20x
bf16,  old attn                 194.3    1.42x
bf16 + tf32,  old attn          194.6    1.41x   ← no gain over bf16
bf16,  CSA                      149.0    1.85x
bf16 + compile, old attn         90.0    3.06x
bf16,  flash                     80.2    3.43x
bf16 + compile, CSA              81.0    3.40x
bf16 + compile, flash            65.9    4.18x   ←

```

Two things that row tells you:

**tf32 alone is worth 1.20x** — one line, no autocast, no numerics to think about. Cheapest win available if you ever run without bf16.

**tf32 + bf16 = nothing extra** (194.3 → 194.6, i.e. noise). They're not additive: autocast already routes matmuls through bf16 tensor cores, so there's no fp32 matmul left for TF32 to accelerate. Keep the line anyway — `full_val_loss` and `generate` run outside autocast, and tf32 helps there.


                        ms/step   peak mem
    compile + CSA              80.5     2.3 GB
    compile + flash            65.1     1.6 GB
    

**0.7 GB saved, ~30%.** And the arithmetic confirms where it comes from: your score tensor is `[64, 6, 256, 256]` = 25.2M elements ≈ 50 MB in bf16, and autograd retains a couple of those per layer for the backward pass. 6 layers × ~100 MB ≈ 0.6 GB — which is what disappeared.

That's the number that matters more than the speed, because it's the one that scales badly. Attention memory grows as **T²** in the CSA version and roughly **linearly** in flash. Concretely, at `block_size=512`:

*   CSA: the 0.7 GB of score tensors becomes ~2.8 GB, on top of everything else growing — you'd be at or past your 7.6 GB
*   flash: no score tensors at all, so you'd land somewhere near 2.5-3 GB

