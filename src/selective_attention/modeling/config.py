from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 32000
    model_dim: int = 512
    head_dim: int = 64
    ssm_state_dim: int = 64
    ssm_conv_kernel_size: int = 4
    ssm_num_groups: int = 1
    ssm_chunk_size: int = 256
    mlconv_radius: int = 2
    num_layers: int = 4
    dropout_rate: float = 0.15
    device: str | None = "cuda"
