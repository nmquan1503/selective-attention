from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    attn_gate_threshold: float = 0.01
    cache_update_interval: int = 100
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int