import pytest
import torch
from unittest.mock import MagicMock
from ..generate import GenerateConfig, TTSGenerator
from ..transformer import TTSTransformer


@pytest.fixture
def mock_components():
    config = GenerateConfig(temperature=0.7, top_k=50, top_p=0.9)

    transformer = MagicMock(spec=TTSTransformer)
    transformer.num_quantizers = 9
    vocab_size = 1026
    transformer.return_value = torch.randn(1, 9, 1, vocab_size)  # Mock logits

    if config.device.startswith("cuda"):
        transformer.return_value = transformer.return_value.to(config.device)

    text_tokenizer = MagicMock()
    text_tokenizer.encode.return_value = [1, 2, 3]

    audio_tokenizer = MagicMock()
    audio_tokenizer.bos_id = 1024
    audio_tokenizer.eos_id = 1025
    audio_tokenizer.vocab_size = vocab_size
    frequency = 16000
    audio_tokenizer.decode.return_value = torch.randn(1, frequency)  # 1s audio

    return transformer, text_tokenizer, audio_tokenizer, config


def test_greedy_sampling(mock_components):
    transformer, text_tok, audio_tok, config = mock_components
    config.temperature = 0.0

    vocab_size = 1026
    mock_logits = torch.full((1, 9, 1, vocab_size), -float("inf"))
    mock_logits[:, :, :, 0] = 100
    transformer.return_value = mock_logits.to(config.device)

    generator = TTSGenerator(transformer, text_tok, audio_tok, config)
    generator.generate("test")

    generated = generator.audio_tokenizer.decode.call_args[0][0]
    assert torch.all(generated[:, :, :] == 0), (
        "Non-zero tokens after BOS removed in post processing"
    )


def test_temperature_sampling(mock_components):
    transformer, text_tok, audio_tok, config = mock_components
    config.temperature = 0.7
    config.top_k = None
    config.top_p = None

    vocab_size = 1024
    mock_logits = torch.full((1, 9, 1, vocab_size), -float("inf"))
    mock_logits[:, :, :, 0] = 1.0
    mock_logits[:, :, :, 1] = 2.0  # Will become highest after temperature scaling
    mock_logits = mock_logits.to(config.device)

    transformer.return_value = mock_logits

    torch.manual_seed(42)
    generator = TTSGenerator(transformer, text_tok, audio_tok, config)
    generator.generate("test")

    generated = generator.audio_tokenizer.decode.call_args[0][0]
    assert torch.any(generated == 1), (
        "Should sample token 1 with these logits/temperature"
    )

    assert not torch.all(generated == 0), (
        "Temperature sampling should not be purely greedy"
    )


def test_token_range(mock_components):
    transformer, text_tok, audio_tok, config = mock_components
    generator = TTSGenerator(transformer, text_tok, audio_tok, config)

    vocab_size = audio_tok.vocab_size
    generator.transformer.return_value = torch.randn(
        1, 9, 100, vocab_size, device=config.device
    )

    generator.generate("test")
    generated = generator.audio_tokenizer.decode.call_args[0][0]

    assert (generated >= 0).all() and (generated < vocab_size).all()


def test_output_creation(mock_components):
    transformer, text_tok, audio_tok, config = mock_components
    generator = TTSGenerator(transformer, text_tok, audio_tok, config)

    generator.generate("test")

    minimum_reasonable_size = 1024

    assert config.output_path.exists()
    assert config.output_path.stat().st_size > minimum_reasonable_size


def test_eos_handling(mock_components):
    transformer, text_tok, audio_tok, config = mock_components
    generator = TTSGenerator(transformer, text_tok, audio_tok, config)

    vocab_size = 1026  # 1024 + BOS + EOS
    eos_id = audio_tok.eos_id
    seq_lengths = []

    def mock_forward(*args, **kwargs):
        current_length = args[1].shape[-1]
        seq_lengths.append(current_length)

        # Create logits that predict EOS on 5th token
        logits = torch.full((1, 9, 1, vocab_size), -1e9, device=config.device)
        if current_length >= 5:
            # Set EOS probability high for one codebook
            logits[:, 0, :, eos_id] = 100
        else:
            logits[:, :, :, 100] = 100
        return logits

    transformer.side_effect = mock_forward

    generator.generate("test")
    generated_tokens = audio_tok.decode.call_args[0][0]

    assert len(seq_lengths) < config.max_tokens, "Should have stopped early"

    final_length = seq_lengths[-1]

    assert not torch.any(generated_tokens == eos_id), "EOS token not removed"

    assert generated_tokens.shape[-1] == final_length - 1, (
        f"Expected {final_length - 1} tokens, got {generated_tokens.shape[-1]}"
    )
