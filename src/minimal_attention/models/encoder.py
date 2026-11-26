import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict

from ..modules import BiBlock, RMSNorm
from ..inference import AnalysisConfig

@dataclass
class EncoderConfig:
    vocab_size: int = 32000
    model_dim: int = 512
    head_dim: int = 64
    ssm_state_dim: int = 64
    ssm_conv_kernel_size: int = 4
    ssm_num_groups: int = 1
    ssm_chunk_size: int = 256
    num_layers: int = 4
    dropout_rate: float = 0.15
    device: str | None = "cuda"

class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()

        if cfg is None:
            cfg = EncoderConfig()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList([
            BiBlock(
                layer_idx=layer_idx,
                model_dim=cfg.model_dim,
                head_dim=cfg.head_dim,
                ssm_state_dim=cfg.ssm_state_dim,
                ssm_conv_kernel_size=cfg.ssm_conv_kernel_size,
                ssm_num_groups=cfg.ssm_num_groups,
                ssm_chunk_size=cfg.ssm_chunk_size,
                dropout_rate=cfg.dropout_rate,
                device=cfg.device
            )
            for layer_idx in range(cfg.num_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            lengths: (batch_size,)
        
        Returns:
            hidden_states: (batch_size, seq_len, model_dim)
        """

        hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                lengths=lengths,
            )
        
        return hidden_states

    def warmup(self, batch_size: int = 2):
        device = self.cfg.device
        seq_len = max(2, self.cfg.ssm_chunk_size)
        input_ids = torch.randint(
            0, self.cfg.vocab_size,
            (batch_size, seq_len),
            device=device,
            dtype=torch.long
        )
        lengths = torch.full(
            (batch_size,),
            fill_value=seq_len,
            device=device,
            dtype=torch.long
        )

        if self.training:
            hidden_states = self.forward(input_ids=input_ids, lengths=lengths)
            loss = hidden_states.float().mean()
            loss.backward()
            self.zero_grad(set_to_none=True)

        self.eval()
        with torch.no_grad():
            self.forward(
                input_ids=input_ids,
                lengths=lengths,
            )

        torch.cuda.synchronize(device)