class CrossSelectiveAttnCache:
    def __init__(self):
        self.k = None   # (batch_size, num_heads, context_len, head_dim)
        self.v = None   # (batch_size, num_heads, context_len, head_dim)
        self.gate = None    # (batch_size, num_heads, context_len)
        self.valid_mask = None  # (batch_size, num_heads, context_len)