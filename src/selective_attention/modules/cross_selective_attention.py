import torch
import torch.nn as nn

from ..inference import CrossSelectiveAttnCache

class CrossSelectiveMHA(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor | None = None,
        log_gate: torch.Tensor | None = None,
        cache: CrossSelectiveAttnCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            context: (batch_size, context_len, model_dim)
            log_gate: (batch_size, context_len)

        Returns:
            hidden_states: (batch_size, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        context_len = context.shape[1]
        device = hidden_states.device
        is_infer = cache is not None and cache.k is not None

        q = self.q_proj(hidden_states)
        
        if is_infer:
            k = cache.k
            v = cache.v
            log_gate = cache.log_gate

        else:
            k = self.k_proj(context)
            v = self.v_proj(context)
            cache.k = k
            cache.v = v
            cache.log_gate = log_gate

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        attn_matrix = (q @ k.transpose(-2, -1)) / scale
        attn_matrix = attn_matrix + log_gate[:, None, None, :]

        attn_weights = torch.softmax(attn_matrix, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        hidden_states = self.out_proj(out)

        return hidden_states