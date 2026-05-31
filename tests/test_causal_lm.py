import random
import numpy as np
import torch

from selective_attention.models import CausalLM, CausalLMConfig
from selective_attention.inference import (
    CausalBlockCache,
    InferenceState,
    GenerationConfig,
)

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = CausalLMConfig(
        vocab_size=32000,
        model_dim=512,
        head_dim=64,
        ssm_state_dim=128,
        ssm_conv_kernel_size=4,
        ssm_num_groups=1,
        ssm_chunk_size=256,
        attn_conv_kernel_size=4,
        num_layers=8,
        dropout_rate=0.0,
        device=device,
    )

    model = CausalLM(cfg).to(device)
    model.eval()
    return model, cfg


def build_batch(batch_size, seq_len, lengths, vocab_size, device):
    x = torch.full((batch_size, seq_len), 0, dtype=torch.long, device=device)

    for b in range(batch_size):
        l = lengths[b].item()
        x[b, :l] = torch.randint(10, vocab_size, (l,), device=device)

    return x


def test_forward_step():
    print("test_forward_step")

    model, cfg = build_model()
    device = cfg.device

    batch_size = 4
    seq_len = 256

    lengths = torch.tensor([256, 220, 180, 140], device=device)
    input_ids = build_batch(batch_size, seq_len, lengths, cfg.vocab_size, device)

    prefill = torch.tensor([32, 64, 96, 48], device=device)
    steps = 64

    gen_cfg = GenerationConfig(
        max_new_tokens=steps,
        pad_token_id=0,
        eos_token_id=-1,
        bos_token_id=-1,
        attn_gate_thresholds=[0.5] * 8,
    )

    full_logits = model.forward(
        input_ids=input_ids,
        lengths=lengths,
        attn_gate_thresholds=[0.5] * 8,
        cache=None,
    )

    cache = [CausalBlockCache() for _ in range(cfg.num_layers)]
    state = InferenceState(prefill.clone())

    _ = model.forward(
        input_ids=input_ids,
        lengths=prefill,
        attn_gate_thresholds=[0.5] * 8,
        cache=cache,
    )

    total_mean = 0
    worst = 0
    argmax_err = 0

    for i in range(steps):
        cur = input_ids[torch.arange(batch_size, device=device), prefill + i]

        step_logits = model.step(
            input_ids=cur,
            cache=cache,
            state=state,
            gen_cfg=gen_cfg,
        )

        pos = state.lengths
        ref_logits = full_logits[torch.arange(batch_size, device=device), pos]

        diff = (step_logits - ref_logits).abs()

        mean_d = diff.mean().item()
        max_d = diff.max().item()

        total_mean += mean_d
        worst = max(worst, max_d)

        if not torch.equal(step_logits.argmax(-1), ref_logits.argmax(-1)):
            argmax_err += 1

        print(f"step {i} mean={mean_d:.6f} max={max_d:.6f}")

        state.update()

    print("summary")
    print("mean", total_mean / steps)
    print("worst", worst)
    print("argmax_err", argmax_err)

    assert total_mean / steps < 1e-5
    assert argmax_err == 0


def test_generate_forward():
    print("test_generate_forward")

    model, cfg = build_model()
    device = cfg.device

    batch_size = 4
    prompt_len = 192

    lengths = torch.tensor([48, 96, 144, 192], device=device)
    input_ids = build_batch(batch_size, prompt_len, lengths, cfg.vocab_size, device)

    max_new = 128

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new,
        pad_token_id=0,
        eos_token_id=-1,
        bos_token_id=-1,
        attn_gate_thresholds=[0.5] * 8,
    )

    gen = model.generate(input_ids, gen_cfg)

    final_len = lengths + max_new
    max_len = final_len.max().item()

    clean = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)

    for b in range(batch_size):
        l = lengths[b].item()

        clean[b, :l] = input_ids[b, :l]
        clean[b, l:l + max_new] = gen[b, prompt_len:prompt_len + max_new]

    full_logits = model.forward(
        input_ids=clean,
        lengths=final_len,
        attn_gate_thresholds=[0.5] * 8,
        cache=None,
    )

    mismatch = 0

    for i in range(max_new):
        pos = lengths + i
        logits_pos = pos - 1

        pred = full_logits[torch.arange(batch_size, device=device), logits_pos].argmax(-1)
        true = clean[torch.arange(batch_size, device=device), pos]

        mismatch += (pred != true).sum().item()

        if i % 16 == 0:
            print(f"step {i} mismatch {mismatch}")

    assert mismatch == 0


if __name__ == "__main__":
    test_forward_step()
    test_generate_forward()
    print("all tests passed")