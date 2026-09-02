"""Types shared by more than one layer -- kept here so importing an alias never
drags in a module that imports you back."""

from torch import Tensor

# k and v for every position attended so far, each [B, nh, T_kv, hs]
type KVCache = tuple[Tensor, Tensor]
