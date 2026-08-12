                                  ms/step   speedup
    fp32,  old attn                 275.5     1.00x
    fp32 + tf32,  old attn          235.1     1.20x
    bf16,  old attn                 194.3     1.42x
    bf16 + tf32,  old attn          199.6     1.41x
    bf16,  CSA                      149.0     1.85x
    bf16 + compile, old attn         90.0     3.06x
    bf16 + compile, CSA              81.0     3.40x
    

Two things that row tells you:

**tf32 alone is worth 1.20x** — one line, no autocast, no numerics to think about. Cheapest win available if you ever run without bf16.

**tf32 + bf16 = nothing extra** (199.4 → 199.6, i.e. noise). They're not additive: autocast already routes matmuls through bf16 tensor cores, so there's no fp32 matmul left for TF32 to accelerate. Keep the line anyway — `full_val_loss` and `generate` run outside autocast, and tf32 helps there.