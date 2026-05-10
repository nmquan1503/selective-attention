import torch
import torch.nn as nn

from selective_attention.modeling.block import Block
from selective_attention.modeling.rms_norm import RMSNorm
from selective_attention.modeling.config import Config
from selective_attention.modeling.inference_state import InferenceState
from selective_attention.modeling.generation_config import GenerationConfig

class CausalLM(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        
        if cfg is None:
            cfg = Config()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList([
            Block(
                layer_idx=layer_idx,
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
        state: InferenceState | None = None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            lengths: (batch_size,)
        
        Returns:
            (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, lengths=lengths, state=state)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def step(self, input_ids: torch.Tensor, state: InferenceState, gen_cfg: GenerationConfig):
        """
        Args:
            input_ids: (batch_size,)
        
        Returns:
            logits: (batch_size, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer.step(hidden_states, state, gen_cfg)
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
        
        state = InferenceState()
        lengths = (input_ids != gen_cfg.pad_token_id).sum(dim=1)
        last_indices = lengths - 1

        logits = self.forward(input_ids, lengths, state)
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

            logits = self.step(next_token.squeeze(1), state, gen_cfg)
            state.step += 1

        eos_mask = (seq_ids == gen_cfg.eos_token_id)
        first_eos = eos_mask.float().cumsum(dim=1) >= 1
        seq_ids = torch.where(first_eos, gen_cfg.eos_token_id, seq_ids)

        return seq_ids