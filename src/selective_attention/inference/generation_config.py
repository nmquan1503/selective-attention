from dataclasses import dataclass
from typing import List

@dataclass
class GenerationConfig:
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    max_new_tokens: int = 256
    attn_gate_thresholds: List[float] | None = None
    cross_attn_gate_thresholds: List[float] | None = None
    enc_attn_gate_thresholds: List[float] | None = None
    cache_update_interval: int = 100