import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
from copy import deepcopy

from ..modules import RMSNorm, CausalBlock
from ..inference import InferenceState, CausalBlockCache, GenerationConfig, AnalysisConfig

@dataclass
class CausalLMConfig:
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

class CausalLM(nn.Module):
    def __init__(self, cfg: CausalLMConfig | None = None):
        super().__init__()
        
        if cfg is None:
            cfg = CausalLMConfig

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList([
            CausalBlock(
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

        self.norm = RMSNorm(cfg.model_dim)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def warmup(self, batch_size: int = 2):
        device = self.cfg.device
        seq_len = max(2, self.cfg.ssm_chunk_size)
        input_ids = torch.randint(
            0,self.cfg.vocab_size, 
            (batch_size, seq_len), 
            device=device, 
            dtype=torch.long
        )

        if self.training:
            logits = self.forward(input_ids=input_ids)
            loss = logits.float().mean()
            loss.backward()
            self.zero_grad(set_to_none=True)

        self.eval()
        self.generate(input_ids, GenerationConfig(
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            max_new_tokens=1,
            attn_gate_thresholds=torch.full(
                (self.cfg.num_layers, self.cfg.model_dim // self.cfg.head_dim), 
                0.5, device=self.cfg.device
            ),
        ))

        torch.cuda.synchronize(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        attn_gate_thresholds: torch.Tensor | List | None = None,
        cache: list[CausalBlockCache] | None = None,
        analysis_cfg: AnalysisConfig | None = None,
        stats: List[dict] | None = None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            lengths: (batch_size,)
            attn_gate_thresholds: (num_layers, num_heads)
        
        Returns:
            (batch_size, seq_len, vocab_size)
        """
        is_infer = cache is not None

        hidden_states = self.embedding(input_ids)
        
        for layer_idx, layer in enumerate(self.layers):
            attn_gate_threshold = (
                attn_gate_thresholds[layer_idx] 
                    if attn_gate_thresholds is not None 
                    else None
            )
            hidden_states, _ = layer(
                hidden_states=hidden_states, 
                lengths=lengths,
                attn_gate_threshold=attn_gate_threshold,
                cache=cache[layer_idx] if is_infer else None,
                analysis_cfg=analysis_cfg,
                stats=stats[layer_idx] if stats is not None else None
            )
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

    def step(
        self, 
        input_ids: torch.Tensor,
        cache: list[CausalBlockCache], 
        state: InferenceState, 
        gen_cfg: GenerationConfig,
        analysis_cfg: AnalysisConfig | None = None,
        stats: List[dict] | None = None
    ):
        """
        Args:
            input_ids: (batch_size,)
        
        Returns:
            logits: (batch_size, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.step(
                hidden_states, 
                cache[layer_idx], 
                state, 
                gen_cfg,
                analysis_cfg,
                stats[layer_idx] if stats is not None else None
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        gen_cfg: GenerationConfig,
        analysis_cfg: AnalysisConfig | None = None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:    
            seq_ids: (batch_size, seq_len + num_new_token)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        cache = [CausalBlockCache() for _ in range(self.cfg.num_layers)]
        lengths = (input_ids != gen_cfg.pad_token_id).sum(dim=1)
        state = InferenceState(lengths)
        stats = None
        if analysis_cfg is not None:
            layers_stats = [{} for _ in range(self.cfg.num_layers)]
            overall_stats = {}
            stats = {"layers": layers_stats, "overall": overall_stats}

        # if gen_cfg.attn_gate_thresholds is None:
        #     gen_cfg.attn_gate_thresholds = [0.0] * self.cfg.num_layers

        last_indices = lengths - 1
        logits = self.forward(
            input_ids, 
            lengths, 
            gen_cfg.attn_gate_thresholds, 
            cache, 
            analysis_cfg, 
            stats["layers"] if stats is not None else None
        )
        logits = logits[torch.arange(batch_size, device=device), last_indices]
        
        seq_ids = torch.full(
            (batch_size, seq_len + gen_cfg.max_new_tokens),
            fill_value=gen_cfg.pad_token_id,
            dtype=input_ids.dtype,
            device=device
        )
        seq_ids[:, :seq_len] = input_ids
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(gen_cfg.max_new_tokens):
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)

            seq_ids[:, seq_len + state.step] = next_token
            finished |= (next_token == gen_cfg.eos_token_id)
            
            if finished.all():
                break

            logits = self.step(
                next_token, 
                cache, 
                state, 
                gen_cfg, 
                analysis_cfg, 
                stats["layers"] if stats is not None else None
            )
            state.update()

        eos_mask = (seq_ids == gen_cfg.eos_token_id)
        first_eos = eos_mask.float().cumsum(dim=1) >= 1
        seq_ids = torch.where(first_eos, gen_cfg.eos_token_id, seq_ids)

        if analysis_cfg is not None:
            num_heads = self.cfg.model_dim // self.cfg.head_dim
            max_seq_len = seq_len + min(state.step + 1, gen_cfg.max_new_tokens)
            max_cache_slots = max_seq_len * num_heads * self.cfg.num_layers

            total_kept_slots = 0
            for layer_idx in range(self.cfg.num_layers):
                log_gate = cache[layer_idx].attn_cache.log_gate
                if log_gate is not None:
                    write_idx = cache[layer_idx].attn_cache.write_idx
                    total_kept_slots += (~torch.isinf(log_gate[:, :, :write_idx])).sum().item()

            stats["overall"]["kept_ratio"] = total_kept_slots / max_cache_slots
            return seq_ids, stats

        return seq_ids

    def compute_attn_gate_threshold(
        self, 
        inputs: List[torch.Tensor],
        mass_threshold: float,
        gen_cfg: GenerationConfig,
        analysis_cfg: AnalysisConfig
    ):
        """
        Args:
            inputs: List[(batch_size, seq_len)]
        Returns:
            gate_threshold: (num_layers, num_heads)
        """
        num_bins = analysis_cfg.gate_attn_num_bins
        num_heads = self.cfg.model_dim // self.cfg.head_dim
        num_layers = self.cfg.num_layers
        gate_thresholds = [
            None
            for _ in range(num_layers)
        ]
        for layer_idx in range(num_layers):
            mass = torch.zeros(num_heads, num_bins, device=self.cfg.device)
            count = torch.zeros(num_heads, num_bins, device=self.cfg.device)
            freq = torch.zeros(num_heads, num_bins, device=self.cfg.device)

            current_gen_cfg = deepcopy(gen_cfg)
            current_gen_cfg.attn_gate_thresholds = gate_thresholds

            with torch.inference_mode():
                for ip in inputs:
                    ip = ip.to(self.cfg.device)
                    _, stats_dict = self.generate(ip, current_gen_cfg, analysis_cfg)
                    gate_analysis = stats_dict["layers"][layer_idx]["causal_attn_gate_analysis"]
                    mass += gate_analysis["attn_mass"]
                    count += gate_analysis["attn_count"]
                    freq += gate_analysis["gate_freq"]

            mass_mean = mass / count.clamp(min=1)
            min_freq = (1.0 / num_bins) * 0.1
            freq = freq / freq.sum(dim=-1, keepdim=True).clamp(min=1)
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
