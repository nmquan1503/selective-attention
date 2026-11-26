import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import CrossAttnCache, InferenceState, GenerationConfig

class CrossMHA(nn.Module):
    def __init__(self, layer_idx, dim, head_dim):
        super().__init__()
        self.layer_idx = layer_idx

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_valid_mask: torch.Tensor | None = None, 
        cache: CrossAttnCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            masks: (batch_size, seq_len)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        context_len = context.shape[1]
        device = hidden_states.device
        is_infer = not self.training
 
        q = self.q_proj(hidden_states)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        attn_matrix = (q @ k.transpose(-2, -1)) / scale
        if context_valid_mask is not None:
            attn_matrix = attn_matrix.masked_fill(~context_valid_mask[:, None, None, :], float("-inf"))
        attn_weights = F.softmax(attn_matrix, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)

        if cache is not None:
            cache.build(k, v, context_valid_mask)

        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        cache: CrossAttnCache, 
        state: InferenceState,
        gen_cfg: GenerationConfig,
    ):
        """
        Args:
            hidden_states: (batch_size, model_dim)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """

        batch_size, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = cache.k
        v = cache.v

        q = q.view(batch_size, self.num_heads, self.head_dim)

        scale = self.head_dim ** 0.5
        attn_matrix = (q.unsqueeze(2) @ k.transpose(-2, -1)) / scale
        if cache.mask is not None:
            attn_matrix = attn_matrix.masked_fill(~cache.mask[:, None, None, :], float("-inf"))
        attn_matrix = F.softmax(attn_matrix, dim=-1)
        out = attn_matrix @ v
        out = out.squeeze(2).view(batch_size, self.dim)

        return self.out_proj(out)
