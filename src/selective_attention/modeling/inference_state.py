class LayerState:
    def __init__(self):
        self.ssm_h = None   # (batch_size, seq_len, num_heads, head_dim)
        self.ssm_conv_ctx = None    # (batch_size, conv_dim, conv_kernel - 1)

        self.mlconv_in_ctx = None   # (batch_size, mlconv_in_channels, mlconv_radius + 1)
        self.mlconv_out_ctx = None  # (batch_size, mlconv_radius + 1, mlconv_out_channels)

        self.k_rot = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.v = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.log_gate = None    # (batch_size, compressed_len)
        self.lengths = None # (batch_size,)
        self.write_idx = 0
        self.valid_mask = None  # (batch_size, compressed_len)

class InferenceState:
    def __init__(self, num_layers):
        self.layers = [LayerState() for _ in range(num_layers)]
        self.step = 0