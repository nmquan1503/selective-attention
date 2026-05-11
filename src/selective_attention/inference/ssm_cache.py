class SSMCache:
    def __init__(self):
        self.h = None   # (batch_size, seq_len, num_heads, head_dim)
        self.conv_ctx = None    # (batch_size, conv_dim, conv_kernel - 1)
