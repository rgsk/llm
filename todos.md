### Important

takes W as an integer, in online softmax

FlexAttention — zero kernel authoring, autograd works, so it's the one you'd actually train with. mask_mod(b, h, i, j) -> bool compiles to block-sparse Triton.

see if rope + kv cache just works
