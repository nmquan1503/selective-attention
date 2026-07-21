from dataclasses import dataclass
from typing import List
import torch

@dataclass
class GenerationConfig:
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    max_new_tokens: int = 256
    attn_gate_thresholds: torch.Tensor | List | None = None    # (num_layers, num_heads)
    cross_attn_gate_thresholds: torch.Tensor | List | None = None  # (num_layers, num_heads)
    enc_attn_gate_thresholds: torch.Tensor | List | None = None    # (num_layers, num_heads)
    cache_update_interval: int = 100