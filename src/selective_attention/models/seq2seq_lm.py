import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict

from ..modules import BiBlock, CrossBlock, SSM, RMSNorm
from ..inference import CrossBlockCache, InferenceState, GenerationConfig, AnalysisConfig

@dataclass
class Seq2SeqLMConfig:
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

class Seq2SeqLM(nn.Module):
    def __init__(self, cfg: Seq2SeqLMConfig | None = None):
        super().__init__()

        if cfg is None:
            cfg = Seq2SeqLMConfig()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.norm1 = RMSNorm(cfg.model_dim)
        self.norm2 = RMSNorm(cfg.model_dim)
        self.norm3 = RMSNorm(dim=cfg.ssm_state_dim, num_groups=cfg.model_dim * 2 // cfg.head_dim)
        self.norm4 = RMSNorm(cfg.model_dim)
        self.encoder_layers = nn.ModuleList([
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
        self.ssm = SSM(
            layer_idx=self.cfg.num_layers,
            model_dim=cfg.model_dim,
            state_dim=cfg.ssm_state_dim,
            conv_kernel_size=cfg.ssm_conv_kernel_size,
            head_dim=cfg.head_dim,
            num_groups=cfg.ssm_num_groups,
            chunk_size=cfg.ssm_chunk_size,
            dropout_rate=cfg.dropout_rate,
            device=cfg.device
        )
        self.decoder_layers = nn.ModuleList([
            CrossBlock(
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
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.dropout = nn.Dropout(cfg.dropout_rate)
    
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
            lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
            decoder_input_ids = torch.randint(
                0, self.cfg.vocab_size,
                (batch_size, 2),
                device=device, dtype=torch.long
            )
            logits = self.forward(
                input_ids=input_ids,
                lengths=lengths,
                decoder_input_ids=decoder_input_ids
            )
            loss = logits.float().mean()
            loss.backward()
            self.zero_grad(set_to_none=True)

        self.eval()
        self.generate(input_ids, GenerationConfig(
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            max_new_tokens=2,
            enc_attn_gate_thresholds=torch.full(
                (self.cfg.num_layers, self.cfg.model_dim // self.cfg.head_dim), 
                0.5, device=self.cfg.device
            ),
            attn_gate_thresholds=torch.full(
                (self.cfg.num_layers, self.cfg.model_dim // self.cfg.head_dim), 
                0.5, device=self.cfg.device
            ),
            cross_attn_gate_thresholds=torch.full(
                (self.cfg.num_layers, self.cfg.model_dim // self.cfg.head_dim), 
                0.5, device=self.cfg.device
            )
        ))

        if str(device) == "cuda":
            torch.cuda.synchronize(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        enc_attn_gate_thresholds: torch.Tensor | None = None,
        attn_gate_thresholds: torch.Tensor | None = None,
        cross_attn_gate_thresholds: torch.Tensor | None = None,
        cache: list[CrossBlockCache] | None = None,
        analysis_cfg: AnalysisConfig | None = None,
        stats: List[Dict] | None = None
    ):
        """
        Args:
            input_ids: (batch_size, enc_seq_len)
            lengths: (batch_size,)
            decoder_input_ids: (batch_size, dec_seq_len)
        
        Returns:
            logits: (batch_size, dec_seq_len, vocab_size)
        """
        enc_seq_len = input_ids.size(1)
        device = input_ids.device
        is_infer = not self.training

        enc_hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.encoder_layers):
            enc_hidden_states, _ = layer(
                hidden_states=enc_hidden_states,
                lengths = lengths,
                attn_gate_threshold=enc_attn_gate_thresholds[layer_idx] if enc_attn_gate_thresholds is not None else None,
                analysis_cfg=analysis_cfg,
                stats=stats[layer_idx] if stats is not None else None
            )
        res = enc_hidden_states
        enc_hidden_states = self.norm1(enc_hidden_states)
        enc_hidden_states, ssm_hiddens = self.ssm(
            hidden_states=enc_hidden_states,
            lengths=lengths
        )
        enc_hidden_states = res + self.dropout(enc_hidden_states)

        enc_hidden_states = self.norm2(enc_hidden_states)
        ssm_hiddens = self.norm3(ssm_hiddens.transpose(1, 2)).transpose(1, 2)
        dec_hidden_states = self.embedding(decoder_input_ids)
        for layer_idx, layer in enumerate(self.decoder_layers):
            dec_hidden_states = layer(
                hidden_states=dec_hidden_states,
                context=enc_hidden_states,
                context_lengths=lengths,
                ssm_hiddens=ssm_hiddens,
                self_attn_gate_threshold=attn_gate_thresholds[layer_idx] if attn_gate_thresholds is not None else None,
                cross_attn_gate_threshold=cross_attn_gate_thresholds[layer_idx] if cross_attn_gate_thresholds is not None else None,
                cache=cache[layer_idx] if cache is not None else None,
                analysis_cfg=analysis_cfg,
                stats=stats[layer_idx] if stats is not None else None
            )
        
        dec_hidden_states = self.norm4(dec_hidden_states)
        logits = self.lm_head(dec_hidden_states)

        return logits

    def step(
        self,
        input_ids: torch.Tensor,
        cache: list[CrossBlockCache],
        state: InferenceState,
        gen_cfg: GenerationConfig,
        analysis_cfg: AnalysisConfig | None = None,
        stats: List[Dict] | None = None
    ):
        """
        Args:
            input_ids: (batch_size,)

        Returns:
            logits: (batch_size, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.decoder_layers):
            hidden_states = layer.step(
                hidden_states=hidden_states,
                cache=cache[layer_idx],
                state=state,
                gen_cfg=gen_cfg,
                analysis_cfg=analysis_cfg,
                stats=stats[layer_idx] if stats is not None else None
            )
        hidden_states = self.norm4(hidden_states)
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
            seq_ids: (batch_size, num_new_tokens)
        """

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        cache = [CrossBlockCache() for _ in range(self.cfg.num_layers)]
        state = InferenceState(
            lengths=torch.ones(batch_size, dtype=torch.long, device=device)
        )
        if analysis_cfg is not None:
            layers_stats = [{} for _ in range(self.cfg.num_layers)]
            stats = {"layers": layers_stats, "overall": {}}
        else:
            stats = None

        lengths = (input_ids != gen_cfg.pad_token_id).sum(dim=1)
        seq_ids = torch.full(
            (batch_size, gen_cfg.max_new_tokens),
            fill_value=gen_cfg.pad_token_id,
            dtype=input_ids.dtype,
            device=device
        )
        seq_ids[:, 0] = gen_cfg.bos_token_id 
        logits = self.forward(
            input_ids=input_ids,
            lengths=lengths,
            decoder_input_ids=seq_ids[:, :1],
            enc_attn_gate_thresholds=gen_cfg.enc_attn_gate_thresholds,
            attn_gate_thresholds=gen_cfg.attn_gate_thresholds,
            cross_attn_gate_thresholds=gen_cfg.cross_attn_gate_thresholds,
            cache=cache,
            analysis_cfg=analysis_cfg,
            stats=stats["layers"] if stats is not None else None
        ).squeeze(1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(gen_cfg.max_new_tokens - 1):
            next_token = torch.argmax(logits, dim=-1)
            seq_ids[:, state.step + 1] = next_token
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
            max_cached_tokens = (seq_len + min(state.step() + 2, gen_cfg.max_new_tokens)) * self.cfg.num_layers
            total_kept_tokens = sum(
                cache[layer_idx].cross_attn_cache.k.shape[2] + cache[layer_idx].attn_cache.write_idx
                for layer_idx in range(self.cfg.num_layers)
            )
            stats["overall"]["token_kept_ratio"] = total_kept_tokens / max_cached_tokens

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
            enc_gate_threshold: (num_layers, num_heads)
            cross_gate_threshold: (num_layers, num_heads)
            dec_gate_threshold: (num_layers, num_heads)
        """ 
        num_bins = analysis_cfg.gate_attn_num_bins
        num_heads = self.cfg.model_dim // self.cfg.head_dim
        
        enc_mass = torch.zeros(self.cfg.num_layers, num_heads, num_bins, device=self.cfg.device)
        enc_count = torch.zeros(self.cfg.num_layers, num_heads, num_bins, device=self.cfg.device)
        
        cross_mass = torch.zeros(self.cfg.num_layers, num_heads, num_bins, device=self.cfg.device)
        cross_count = torch.zeros(self.cfg.num_layers, num_heads, num_bins, device=self.cfg.device)
        
        dec_mass = torch.zeros(self.cfg.num_layers, num_heads, num_bins, device=self.cfg.device)
        dec_count = torch.zeros(self.cfg.num_layers, num_heads, num_bins, device=self.cfg.device)
        
        with torch.inference_mode():
            for ip in inputs:
                ip = ip.to(self.cfg.device)
                _, stats_dict = self.generate(ip, gen_cfg, analysis_cfg)
                layers_stats = stats_dict["layers"]
                for layer_idx in range(self.cfg.num_layers):
                    enc_analysis = layers_stats[layer_idx]["non_causal_attn_gate_analysis"]
                    enc_mass[layer_idx] += enc_analysis["attn_mass"]
                    enc_count[layer_idx] += enc_analysis["attn_count"]

                    cross_analysis = layers_stats[layer_idx]["cross_attn_gate_analysis"]
                    cross_mass[layer_idx] += cross_analysis["attn_mass"]
                    cross_count[layer_idx] += cross_analysis["attn_count"]
                
                    dec_analysis = layers_stats[layer_idx]["causal_attn_gate_analysis"]
                    dec_mass[layer_idx] += dec_analysis[layer_idx]["attn_mass"]
                    dec_count[layer_idx] += dec_analysis[layer_idx]["attn_count"]
        
        enc_mass_mean = enc_mass / enc_count.clamp(min=1)
        enc_above_threshold = enc_mass_mean >= mass_threshold
        enc_has_exceeding_bin = enc_above_threshold.any(dim=-1)
        enc_first_bin = enc_above_threshold.float().argmax(dim=-1)
        enc_gate_threshold = enc_first_bin.float() / num_bins
        enc_gate_threshold = enc_gate_threshold.masked_fill(~enc_has_exceeding_bin, 1.0)

        cross_mass_mean = cross_mass / cross_count.clamp(min=1)
        cross_above_threshold = cross_mass_mean >= mass_threshold
        cross_has_exceeding_bin = cross_above_threshold.any(dim=-1)
        cross_first_bin = cross_above_threshold.float().argmax(dim=-1)
        cross_gate_threshold = cross_first_bin.float() / num_bins
        cross_gate_threshold = cross_gate_threshold.masked_fill(~cross_has_exceeding_bin, 1.0)

        dec_mass_mean = dec_mass / dec_count.clamp(min=1)
        dec_above_threshold = dec_mass_mean >= mass_threshold
        dec_has_exceeding_bin = dec_above_threshold.any(dim=-1)
        dec_first_bin = dec_above_threshold.float().argmax(dim=-1)
        dec_gate_threshold = dec_first_bin.float() / num_bins
        dec_gate_threshold = dec_gate_threshold.masked_fill(~dec_has_exceeding_bin, 1.0)

        return enc_gate_threshold, cross_gate_threshold, dec_gate_threshold