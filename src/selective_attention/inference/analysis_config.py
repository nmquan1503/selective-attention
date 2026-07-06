from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    gate_attn_num_bins: int = 100