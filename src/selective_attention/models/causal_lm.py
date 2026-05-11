import torch
import torch.nn as nn
from dataclasses import dataclass

from ..modules import RMSNorm, Block
from ..inference import InferenceState, BlockCache, GenerationConfig

@dataclass
class CausalLMConfig:
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

class CausalLM(nn.Module):
    def __init__(self, cfg: CausalLMConfig | None = None):
        super().__init__()
        
        if cfg is None:
            cfg = CausalLMConfig

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList([
            Block(
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
            for layer_idx in range(cfg.num_layers)
        ])

        self.norm = RMSNorm(cfg.model_dim)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        cache: list[BlockCache] | None = None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            lengths: (batch_size,)
        
        Returns:
            (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, _ = layer(
                hidden_states, 
                lengths=lengths, 
                cache=cache[layer_idx] if cache is not None else None
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def step(
        self, 
        input_ids: torch.Tensor,
        cache: list[BlockCache], 
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
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.step(hidden_states, cache[layer_idx], state, gen_cfg)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, gen_cfg: GenerationConfig):
        """
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:    
            seq_ids: (batch_size, seq_len + num_new_token)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        cache = [BlockCache() for _ in range(self.cfg.num_layers)]
        state = InferenceState()
        lengths = (input_ids != gen_cfg.pad_token_id).sum(dim=1)
        last_indices = lengths - 1

        logits = self.forward(input_ids, lengths, cache)
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
            finished |= (next_token.squeeze(1) == gen_cfg.eos_token_id)
            
            if finished.all():
                break

            logits = self.step(next_token.squeeze(1), cache, state, gen_cfg)
            state.step += 1

        eos_mask = (seq_ids == gen_cfg.eos_token_id)
        first_eos = eos_mask.float().cumsum(dim=1) >= 1
        seq_ids = torch.where(first_eos, gen_cfg.eos_token_id, seq_ids)

        return seq_ids