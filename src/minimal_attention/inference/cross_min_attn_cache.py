class CrossMinAttnCache:
    def __init__(self):
        self.k = None   # (batch_size, num_heads, context_len, head_dim)
        self.v = None   # (batch_size, num_heads, context_len, head_dim)
        self.log_gate = None    # (batch_size, num_heads, context_len)