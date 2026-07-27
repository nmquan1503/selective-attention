import torch

class MinAttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.v = None   # (batch_size, num_heads, compressed_len, head_dim)
        self.log_gate = None    # (batch_size, num_heads, compressed_len)
        self.write_idx = 0

        self.k_fast = None
        self.v_fast = None
        self.log_gate_fast = None
        self.cu_seqlens_k = None
        self.write_pos = None

    def build_kv(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        log_gate: torch.Tensor,
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, compressed_len, head_dim)
            v: (batch_size, num_heads, compressed_len, head_dim)
            log_gate: (batch_size, num_heads, compressed_len)
        """
        self.k_rot = k_rot.contiguous()
        self.v = v.contiguous()
        self.log_gate = log_gate.contiguous()
        self.write_idx = self.k_rot.shape[2]

    def update_kv(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        log_gate: torch.Tensor,
    ):
        """
        Args:
            k_rot: (batch_size, self.num_heads, self.head_dim)
            v: (batch_size, self.num_heads, self.head_dim)
            log_gate: (batch_size, num_heads)
        """
        self.k_rot[:, :, self.write_idx, :] = k_rot
        self.v[:, :, self.write_idx, :] = v
        self.log_gate[:, :, self.write_idx] = log_gate
        self.write_idx += 1

    def build_fast(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        log_gate: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        write_pos: torch.Tensor
    ):
        self.k_fast = k
        self.v_fast = v
        self.log_gate_fast = log_gate
        self.cu_seqlens_k = cu_seqlens_k
        self.write_pos = write_pos