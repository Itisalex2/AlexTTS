import gc
from pathlib import Path

import dac
import pytest
import torch
from audiotools import AudioSignal
from torch import Tensor

from ..tokenizer import DacTokenizer

TEST_DIR = Path(__file__).parent
SAMPLE_AUDIO_DIR = TEST_DIR / "sample_audio"


@pytest.fixture
def model() -> dac.DAC:
    model_type = "16khz"
    try:
        model_path = dac.utils.download(model_type=model_type)
    except Exception as e:
        raise RuntimeError(f"Failed to download or load the DAC model: {e}")
    model = dac.DAC.load(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model


@pytest.fixture
def tokenizer(model) -> DacTokenizer:
    tok = DacTokenizer(model)
    return tok


@pytest.fixture(autouse=True)
def clear_cuda_memory_after_test():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def test_instantiation(tokenizer: DacTokenizer):
    assert tokenizer is not None
    assert tokenizer.pad_id == 0


def test_encode_from_file(tokenizer):
    audio_file_path = SAMPLE_AUDIO_DIR / "sa1.wav"
    codes = tokenizer.encode(audio_file_path)
    assert isinstance(codes, Tensor)
    assert codes is not None
    assert isinstance(codes, torch.Tensor)
    assert codes.dim() == 3, (
        f"Expected codes to have 3 dimensions, but got {codes.dim()}"
    )
    B, N, T = codes.shape
    assert B > 0, "The number of Batches (B) should be greater than 0"
    assert N > 0, "The number of codebooks (N) should be greater than 0"
    assert T > 0, "The number of time steps (T) should be greater than 0"
    assert not torch.is_floating_point(codes), (
        "Codes tensor should contain integer values"
    )


def test_encode_from_tensor(tokenizer):
    sample_rate = 44100
    duration_seconds = 1
    num_samples = sample_rate * duration_seconds
    audio_tensor = torch.randn(1, 1, num_samples)  # Mono audio
    codes = tokenizer.encode(audio_tensor)
    assert isinstance(codes, Tensor)
    assert codes is not None


def test_encode_with_incorrect_tensor_dimensions(tokenizer):
    sample_rate = 44100
    duration_seconds = 1
    num_samples = sample_rate * duration_seconds
    incorrect_audio_tensor = torch.randn(1, num_samples)  # Missing batch dimension

    with pytest.raises(ValueError, match="Expected tensor with 3 dimensions."):
        tokenizer.encode(incorrect_audio_tensor)


def test_encode_with_unsupported_input_type(tokenizer):
    unsupported_input = 12345

    with pytest.raises(
        TypeError,
        match="Unsupported audio input type. Expected file path or torch.Tensor.",
    ):
        tokenizer.encode(unsupported_input)


def test_bos_token_addition(tokenizer):
    audio_file_path = SAMPLE_AUDIO_DIR / "sa1.wav"

    codes = tokenizer.encode(audio_file_path, add_bos=True, add_eos=False)

    assert torch.all(codes[:, :, 0] == tokenizer.bos_id), (
        "BOS token not found at start of all codebooks"
    )

    assert not torch.any(codes[:, :, 1:] == tokenizer.bos_id), (
        "BOS token should not appear in middle of sequence"
    )


def test_eos_token_addition(tokenizer):
    audio_file_path = SAMPLE_AUDIO_DIR / "sa1.wav"

    codes = tokenizer.encode(audio_file_path, add_bos=False, add_eos=True)

    assert torch.all(codes[:, :, -1] == tokenizer.eos_id), (
        "EOS token not found at end of all codebooks"
    )

    assert not torch.any(codes[:, :, :-1] == tokenizer.eos_id), (
        "EOS token appears in middle of sequence"
    )


def test_full_token_wrapping(tokenizer):
    audio_file_path = SAMPLE_AUDIO_DIR / "sa1.wav"

    codes = tokenizer.encode(audio_file_path, add_bos=False, add_eos=False)
    original_length = codes.shape[-1]

    codes_with_wrap = tokenizer.encode(audio_file_path, add_bos=True, add_eos=True)

    assert codes_with_wrap.shape[-1] == original_length + 2, (
        f"Expected length {original_length + 2}, got {codes_with_wrap.shape[-1]}"
    )

    assert torch.all(codes_with_wrap[:, :, 0] == tokenizer.bos_id), "BOS missing"
    assert torch.all(codes_with_wrap[:, :, -1] == tokenizer.eos_id), "EOS missing"

    assert torch.all(codes_with_wrap[:, :, 1:-1] == codes), (
        "Original codes modified between BOS and EOS"
    )


def test_decode_from_dacfile(tokenizer):
    dac_file_path = SAMPLE_AUDIO_DIR / "sa2.dac"
    reconstructed_signal = tokenizer.decode(dac_file_path)
    assert isinstance(reconstructed_signal, AudioSignal)
    assert reconstructed_signal.audio_data is not None


def test_decode_from_tensor(tokenizer):
    audio_file_path = SAMPLE_AUDIO_DIR / "sa2.wav"
    audio_signal = AudioSignal(audio_file_path)
    compressed = tokenizer.model.compress(audio_signal)
    codes = compressed.codes  # Shape: [B, Q, T]

    device = next(tokenizer.model.parameters()).device
    codes = codes.to(device)

    decoded_output = tokenizer.decode(codes)
    assert isinstance(decoded_output, torch.Tensor)
    assert decoded_output.dim() == 3
