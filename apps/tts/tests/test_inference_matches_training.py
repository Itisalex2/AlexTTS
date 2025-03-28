import torch
from ..transformer import TTSTransformer, TTSTransformerArgs
from ..tokenizer import DacTokenizer, MisakiTokenizer, create_dac_tokenizer_model


def test_inference_matches_training():
    misaki_tokenizer = MisakiTokenizer()
    dac_model = create_dac_tokenizer_model()
    dac_tokenizer = DacTokenizer(dac_model)

    args = TTSTransformerArgs(
        text_vocab_size=misaki_tokenizer.vocab_size,
        audio_vocab_size=dac_tokenizer.vocab_size,
    )

    model = TTSTransformer(args).cuda()

    batch_size = 1
    text_seq_len = 50
    audio_seq_len = 100
    num_quantizers = args.num_quantizers

    text_tokens = torch.randint(
        0, args.text_vocab_size, (batch_size, text_seq_len)
    ).cuda()  # [B, text_T]
    audio_tokens = torch.randint(
        0, args.audio_vocab_size, (batch_size, num_quantizers, audio_seq_len)
    ).cuda()  # [B, Q, audio_T]

    logits_forward = model(text_tokens, audio_tokens)  # [B, Q, T, V]

    # Teacher forcing inference
    logits_step = []
    for i in range(audio_seq_len):
        current_audio = audio_tokens[:, :, : i + 1]
        step_logits = model(text_tokens, current_audio)  # [B, Q, i+1, V]
        step_logit = step_logits[:, :, -1, :]
        logits_step.append(step_logit)

    logits_step = torch.stack(logits_step, dim=2)  # [B, Q, T, V]

    relative_error = (
        (logits_forward - logits_step).abs() / logits_forward.abs()
    ).mean()

    assert relative_error < 1e-5, "Relative error too high"


if __name__ == "__main__":
    test_inference_matches_training()
