check doc_block_mask in src/video/mask.py and compare flex_attention with F.scaled_dot_product_attention

we have to see this in action, a possible way to reduce kv cache allocations, when we maintain in full before sliding window -
Apply it to the cache: preallocate, and double capacity when full. That's amortized O(T), log₂(T) copies total, no ceiling, no new config field.
