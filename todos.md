### Important

That's the model complete end to end. What's left is packaging: checkpoint.py and basic.py as the entrypoint — then the deferred pieces (dropout, RMSNorm, SwiGLU, fused QKV, SDPA, KV cache) that make it checkpoint-compatible with main.py.
