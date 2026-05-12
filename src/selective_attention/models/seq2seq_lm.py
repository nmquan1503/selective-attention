import torch
import torch.nn as nn
from dataclasses import dataclass

from ..modules import BiBlock, CrossBlock, SSM, RMSNorm
from ..inference import CrossBlockCache, InferenceState, GenerationConfig

@dataclass
class Seq2SeqLMConfig:
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

class Seq2SeqLM(nn.Module):
    def __init__(self, cfg: Seq2SeqLMConfig | None = None):
        super().__init__()

        if cfg is None:
            cfg = Seq2SeqLMConfig()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.norm1 = RMSNorm(cfg.model_dim)
        self.norm2 = RMSNorm(cfg.model_dim)
        self.encoder_layers = nn.ModuleList([
            BiBlock(
                model_dim=cfg.model_dim,
                head_dim=cfg.head_dim,
                ssm_state_dim=cfg.ssm_state_dim,
                ssm_conv_kernel_size=cfg.ssm_conv_kernel_size,
                ssm_num_groups=cfg.ssm_num_groups,
                ssm_chunk_size=cfg.ssm_chunk_size,
                mlconv_radius=cfg.mlconv_radius,
                dropout_rate=cfg.dropout_rate,
                device=cfg.device
            )
            for _ in range(cfg.num_layers)
        ])
        self.ssm = SSM(
            model_dim=cfg.model_dim,
            state_dim=cfg.ssm_state_dim,
            conv_kernel_size=cfg.ssm_conv_kernel_size,
            num_groups=cfg.ssm_num_groups,
            chunk_size=cfg.ssm_chunk_size,
            dropout_rate=cfg.dropout_rate,
            device=cfg.device
        )
        self.gate_conv = nn.Conv1d(
            in_channels=cfg.model_dim,
            out_channels=1,
            kernel_size=2 * cfg.mlconv_radius + 1,
            padding=cfg.mlconv_radius
        )
        self.decoder_layers = nn.ModuleList([
            CrossBlock(
                model_dim=cfg.model_dim,
                head_dim=cfg.head_dim,
                ssm_state_dim=cfg.ssm_state_dim,
                ssm_conv_kernel_size=cfg.ssm_conv_kernel_size,
                ssm_num_groups=cfg.ssm_num_groups,
                ssm_chunk_size=cfg.ssm_chunk_size,
                mlconv_radius=cfg.mlconv_radius,
                dropout_rate=cfg.dropout_rate,
                device=cfg.device
            )
            for _ in range(cfg.num_layers)
        ])
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        cache: list[CrossBlockCache] | None = None
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

        enc_hidden_states = self.embedding(input_ids)
        for layer in self.encoder_layers:
            enc_hidden_states, _ = layer(
                hidden_states=enc_hidden_states,
                lengths = lengths   
            )
        enc_hidden_states = self.norm1(enc_hidden_states)
        enc_hidden_states, ssm_hiddens = self.ssm(
            hidden_states=enc_hidden_states,
            lengths=lengths
        )
        enc_gate = torch.sigmoid(self.gate_conv(
            enc_hidden_states.transpose(1, 2)
        ).squeeze(1))
        idx = torch.arange(enc_seq_len, device=device)
        pad_mask = idx[None, :] >= lengths[:, None]
        enc_log_gate = torch.log(enc_gate.clamp(min=1e-12))
        enc_log_gate = enc_log_gate.masked_fill(pad_mask, float("-inf"))

        dec_hidden_states = self.embedding(decoder_input_ids)
        for layer_idx, layer in enumerate(self.decoder_layers):
            dec_hidden_states = layer(
                hidden_states=dec_hidden_states,
                context=enc_hidden_states,
                context_log_gate=enc_log_gate,
                ssm_hiddens=ssm_hiddens,
                cache=cache[layer_idx] if cache is not None else None
            )
        
        dec_hidden_states = self.norm2(dec_hidden_states)
        logits = self.lm_head(dec_hidden_states)

        return logits

    def step(
        self,
        input_ids: torch.Tensor,
        cache: list[CrossBlockCache],
        state: InferenceState,
        gen_cfg: GenerationConfig
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
                gen_cfg=gen_cfg
            )
        hidden_states = self.norm2(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        gen_cfg: GenerationConfig
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
        for layer_idx in range(1, self.cfg.num_layers):
            cache[layer_idx].cross_attn_cache = cache[0]
        state = InferenceState()

        lengths = (input_ids != gen_cfg.pad_token_id).sum(dim=1)
        seq_ids = torch.full(
            (batch_size, gen_cfg.max_new_tokens),
            fill_value=gen_cfg.pad_token_id,
            dtype=input_ids.dtype,
            device=device
        )
        seq_ids[:, 1] = gen_cfg.bos_token_id 
        logits = self.forward(
            input_ids=input_ids,
            lengths=lengths,
            decoder_input_ids=seq_ids[:, :1],
            cache=cache
        ).squeeze(1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(gen_cfg.max_new_tokens - 1):
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            seq_ids[:, state.step + 1] = next_token
            finished |= (next_token == gen_cfg.eos_token_id)

            if finished.all():
                break
        
            logits = self.step(next_token, cache, state, gen_cfg)
            state.step += 1
        
        eos_mask = (seq_ids == gen_cfg.eos_token_id)
        first_eos = eos_mask.float().cumsum(dim=1) >= 1
        seq_ids = torch.where(first_eos, gen_cfg.eos_token_id, seq_ids)

        return seq_ids
