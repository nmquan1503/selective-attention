from dataclasses import dataclass

@dataclass
class GenerationConfig:
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    max_new_tokens: int = 256
    attn_gate_threshold: float = 0.01
    cache_update_interval: int = 100