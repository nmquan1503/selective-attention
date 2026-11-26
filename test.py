import torch

from minimal_attention.models import Seq2SeqLM, Seq2SeqLMConfig
# =====================================================

def main():
    torch.manual_seed(42)

    cfg = Seq2SeqLMConfig(
        vocab_size=1000,
        model_dim=128,
        head_dim=32,
        num_layers=2,
        dropout_rate=0.0,
        device="cpu",   # đổi thành "cuda" nếu muốn
    )

    model = Seq2SeqLM(cfg)
    model.eval()

    device = cfg.device
    batch_size = 2
    enc_seq_len = 10
    dec_seq_len = 5

    # Seed lại để input tất định
    torch.manual_seed(26)

    input_ids = torch.randint(
        0, cfg.vocab_size, (batch_size, enc_seq_len),
        device=device, dtype=torch.long
    )
    decoder_input_ids = torch.randint(
        0, cfg.vocab_size, (batch_size, dec_seq_len),
        device=device, dtype=torch.long
    )
    lengths = torch.full((batch_size,), enc_seq_len, device=device, dtype=torch.long)

    print("input_ids        :", input_ids.shape)
    print("decoder_input_ids:", decoder_input_ids.shape)
    print("lengths          :", lengths.shape)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            lengths=lengths,
            decoder_input_ids=decoder_input_ids,
        )

    print("logits shape     :", logits.shape)
    print("logits dtype     :", logits.dtype)
    print("logits sum       :", logits.float().sum().item())
    print("logits mean      :", logits.float().mean().item())
    print("logits std       :", logits.float().std().item())

    # In một vài giá trị đầu để so sánh giữa các lần chạy
    print("\nlogits[0, 0, :5] =", logits[0, 0, :5].tolist())
    print("logits[0, -1, :5]=", logits[0, -1, :5].tolist())


if __name__ == "__main__":
    main()