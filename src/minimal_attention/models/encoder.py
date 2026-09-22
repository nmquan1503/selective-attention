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
    attn_log_gate_penalty: float = 2.0
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
                attn_log_gate_penalty=cfg.attn_log_gate_penalty,
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
        attn_gate_thresholds: torch.Tensor | None = None,
        analysis_cfg: AnalysisConfig | None = None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            lengths: (batch_size,)
        
        Returns:
            hidden_states: (batch_size, seq_len, model_dim)
        """

        stats = None
        if analysis_cfg is not None:
            layers_stats = [{} for _ in range(self.cfg.num_layers)]
            overall_stats = {}
            stats = {
                "layers": layers_stats,
                "overall": overall_stats
            }

        hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                lengths=lengths,
                attn_gate_threshold=attn_gate_thresholds[layer_idx] if attn_gate_thresholds is not None else None,
                analysis_cfg=analysis_cfg,
                stats=stats["layers"][layer_idx] if stats is not None else None
            )

        if analysis_cfg is not None:
            total_head_tokens = 0
            kept_head_tokens = 0

            for layer_stats in stats["layers"]:
                key = "non_causal_attn_gate_analysis"
                if key in layer_stats:
                    total_head_tokens += layer_stats[key]["total_head_tokens"]
                    kept_head_tokens += layer_stats[key]["kept_head_tokens"]

            stats["overall"]["kept_ratio"] = kept_head_tokens / max(total_head_tokens, 1)

            return hidden_states, stats
        
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
                attn_gate_thresholds=torch.full(
                    (self.cfg.num_layers, self.cfg.model_dim // self.cfg.head_dim),
                    0.5, device=device
                ),
            )

        torch.cuda.synchronize(device)

    def compute_attn_gate_threshold(
        self,
        inputs: List[torch.Tensor],
        lengths: List[torch.Tensor],
        mass_threshold: float,
        analysis_cfg: AnalysisConfig
    ):
        """
        Args:
            inputs: List[(batch_size, seq_len)]
            lengths: List[(batch_size,)]

        Returns:
            attn_gate_threshold: (num_layers, num_heads)
        """
        num_bins = analysis_cfg.gate_attn_num_bins
        num_heads = self.cfg.model_dim // self.cfg.head_dim
        num_layers = self.cfg.num_layers
        gate_thresholds = [None for _ in range(num_layers)]

        for layer_idx in range(num_layers):
            mass = torch.zeros(num_heads, num_bins, device=self.cfg.device)
            count = torch.zeros(num_heads, num_bins, device=self.cfg.device)
            freq = torch.zeros(num_heads, num_bins, device=self.cfg.device)

            with torch.inference_mode():
                for ip, l in zip(inputs, lengths):
                    ip = ip.to(self.cfg.device)
                    l = l.to(self.cfg.device)
                    _, stats_dict = self.forward(ip, l, gate_thresholds, analysis_cfg)
                    gate_analysis = stats_dict["layers"][layer_idx]["non_causal_attn_gate_analysis"]
                    mass += gate_analysis["attn_mass"]
                    count += gate_analysis["attn_count"]
                    freq += gate_analysis["gate_freq"]

            mass_mean = mass / count.clamp(min=1)
            min_freq = 30
            above_threshold = (mass_mean >= mass_threshold) & (freq >= min_freq)
            has_exceeding_bin = above_threshold.any(dim=-1)
            first_bin = above_threshold.float().argmax(dim=-1)
            calibrated_threshold = first_bin.float() / num_bins
            calibrated_threshold = calibrated_threshold.masked_fill(
                ~has_exceeding_bin,
                1.1,
            )
            gate_thresholds[layer_idx] = calibrated_threshold
        
        return torch.stack(gate_thresholds)