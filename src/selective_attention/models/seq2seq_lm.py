import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

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

        self.generate(input_ids, GenerationConfig(
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            max_new_tokens=1,
            enc_attn_gate_thresholds=[0.5] * self.cfg.num_layers,
            attn_gate_thresholds=[0.5] * self.cfg.num_layers,
            cross_attn_gate_thresholds=[0.5] * self.cfg.num_layers
        ))

        if str(device) == "cuda":
            torch.cuda.synchronize(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        enc_attn_gate_thresholds: List[float] | None = None,
        attn_gate_thresholds: List[float] | None = None,
        cross_attn_gate_thresholds: List[float] | None = None,
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
        is_infer = not self.training

        enc_hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.encoder_layers):
            enc_hidden_states, _ = layer(
                hidden_states=enc_hidden_states,
                lengths = lengths,
                attn_gate_threshold=enc_attn_gate_thresholds[layer_idx] if enc_attn_gate_thresholds is not None else None
            )
        enc_hidden_states = self.norm1(enc_hidden_states)
        enc_hidden_states, ssm_hiddens = self.ssm(
            hidden_states=enc_hidden_states,
            lengths=lengths
        )

        dec_hidden_states = self.embedding(decoder_input_ids)
        for layer_idx, layer in enumerate(self.decoder_layers):
            dec_hidden_states = layer(
                hidden_states=dec_hidden_states,
                context=enc_hidden_states,
                context_lengths=lengths,
                ssm_hiddens=ssm_hiddens,
                self_attn_gate_threshold=attn_gate_thresholds[layer_idx] if attn_gate_thresholds is not None else None,
                cross_attn_gate_threshold=cross_attn_gate_thresholds[layer_idx] if cross_attn_gate_thresholds is not None else None,
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
        state = InferenceState(
            lengths=torch.ones(batch_size, dtype=torch.long, device=device)
        )

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
            cache=cache
        ).squeeze(1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(gen_cfg.max_new_tokens - 1):
            next_token = torch.argmax(logits, dim=-1)
            seq_ids[:, state.step + 1] = next_token
            finished |= (next_token == gen_cfg.eos_token_id)

            if finished.all():
                break
        
            logits = self.step(next_token, cache, state, gen_cfg)
            state.update()
        
        eos_mask = (seq_ids == gen_cfg.eos_token_id)
        first_eos = eos_mask.float().cumsum(dim=1) >= 1
        seq_ids = torch.where(first_eos, gen_cfg.eos_token_id, seq_ids)

        return seq_ids
