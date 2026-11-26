import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import AttnCache, InferenceState, GenerationConfig
from .rope import RoPE

class MHA(nn.Module):
    def __init__(
        self,
        layer_idx, 
        dim, 
        head_dim, 
        is_causal=False,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.is_causal = is_causal

        self.rope = RoPE(self.head_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        masks: torch.Tensor | None = None, 
        cache: AttnCache | None = None,
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            masks: (batch_size, seq_len)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = hidden_states.shape
        device = hidden_states.device
        is_infer = cache is not None

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(seq_len, device=device)
        q_rot, k_rot = self.rope(q, k, positions, mode="seq")

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot @ k_rot.transpose(-2, -1)) / scale
        if self.is_causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            if masks is not None:
                attn_mask = masks[:, None, None, :] & causal_mask[None, None, :, :]
            else:
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            if masks is not None:
                attn_mask = masks[:, None, None, :]
            else:
                attn_mask = torch.ones(batch_size, 1, 1, seq_len, device=device, dtype=torch.bool)

        attn_matrix = attn_matrix.masked_fill(~attn_mask, float("-inf"))
        attn_weights = F.softmax(attn_matrix, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)
        
        if is_infer and self.is_causal:
            cache_mask = masks if masks is not None else torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
            cache.build(k_rot, v, cache_mask)
        
        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        cache: AttnCache, 
        state: InferenceState,
        gen_cfg: GenerationConfig,
    ):
        """
        Args:
            hidden_states: (batch_size, model_dim)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """

        batch_size, dim = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        q_rot, k_rot = self.rope(q, k, state.lengths, mode="pos")

        cache.update(k_rot, v)
        L = cache.seq_len

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot.unsqueeze(2) @ cache.k_rot.transpose(-2, -1)) / scale
        attn_matrix = attn_matrix.masked_fill(
            ~cache.mask[:, None, None, :],
            float("-inf")
        )

        attn_matrix = F.softmax(attn_matrix, dim=-1)
        out = attn_matrix @ cache.v
        out = out.squeeze(2).view(batch_size, self.dim)

        return self.out_proj(out)
