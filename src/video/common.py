"""Types shared by more than one layer -- kept here so importing an alias never
drags in a module that imports you back."""

from kv_cache import KVCache

# KVCache was a type alias here:
#
#     type KVCache = tuple[Tensor, Tensor]
#
# It stopped being one when it grew a preallocated buffer and a cursor, and
# moved to kv_cache.py. Re-exported from here because the attention files that
# import it from common still hold -- the class kept the tuple's interface, so
# none of them noticed.
__all__ = ["KVCache"]
