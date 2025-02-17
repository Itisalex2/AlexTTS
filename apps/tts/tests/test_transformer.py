import gc

import pytest
import torch
from torch import nn

from ..transformer import TTSTransformer, TTSTransformerArgs


@pytest.fixture
def transformer() -> TTSTransformer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = TTSTransformerArgs(
        dim=32,
        n_layers=2,
        n_heads=4,
        text_vocab_size=20,
        audio_vocab_size=30,
        num_quantizers=3,
        max_seqlen=64,
    )

    transformer = TTSTransformer(args).to(device)
    return transformer


@pytest.fixture(autouse=True)
def clear_cuda_memory_after_test():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def test_instantiation(transformer: TTSTransformer):
    assert isinstance(transformer, nn.Module), (
        "Transformer should be an instance of nn.Module."
    )


def test_transformer_inference(transformer: TTSTransformer):
    device = next(transformer.parameters()).device
    batch_size = 4
    text_length = 5
    audio_length = 6
    num_quantizers = transformer.num_quantizers
    audio_vocab_size = transformer.audio_vocab_size

    text_tokens = torch.randint(
        low=0,
        high=transformer.text_vocab_size,
        size=(batch_size, text_length),
        device=device,
        dtype=torch.long,
    )

    audio_tokens = torch.randint(
        low=0,
        high=audio_vocab_size,
        size=(batch_size, num_quantizers, audio_length),
        device=device,
        dtype=torch.long,
    )

    logits = transformer(text_tokens, audio_tokens)
    assert isinstance(logits, torch.Tensor), "Output should be a Tensor."

    expected_shape = (batch_size, num_quantizers, audio_length, audio_vocab_size)
    assert logits.shape == expected_shape, (
        f"Logits shape mismatch! Expected {expected_shape}, got {logits.shape}."
    )


def test_transformer_training(transformer: TTSTransformer):
    device = next(transformer.parameters()).device

    batch_size = 2
    text_length = 3
    audio_length = 4
    num_quantizers = transformer.num_quantizers
    audio_vocab_size = transformer.audio_vocab_size

    text_tokens = torch.randint(
        low=0,
        high=transformer.text_vocab_size,
        size=(batch_size, text_length),
        device=device,
        dtype=torch.long,
    )
    audio_tokens = torch.randint(
        low=0,
        high=audio_vocab_size,
        size=(batch_size, num_quantizers, audio_length),
        device=device,
        dtype=torch.long,
    )

    target = torch.randint(
        low=0,
        high=audio_vocab_size,
        size=(batch_size, num_quantizers, audio_length),
        device=device,
        dtype=torch.long,
    )

    loss, logits = transformer(text_tokens, audio_tokens, target=target)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."
    assert loss.dim() == 0, "Loss must be a scalar (0D tensor)."

    expected_shape = (batch_size, num_quantizers, audio_length, audio_vocab_size)
    assert logits.shape == expected_shape, (
        f"Logits shape mismatch! Expected {expected_shape}, got {logits.shape}."
    )

    assert not torch.isnan(loss).any(), "Loss contains NaNs."
    assert not torch.isinf(loss).any(), "Loss is infinite."

    loss.backward()
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"Param '{name}' got no gradients after backward."
            )
