### Important

takes W as an integer

FlexAttention — zero kernel authoring, autograd works, so it's the one you'd actually train with. mask_mod(b, h, i, j) -> bool compiles to block-sparse Triton.
