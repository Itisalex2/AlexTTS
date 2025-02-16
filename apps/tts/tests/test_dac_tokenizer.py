import pytest
import torch
import dac
import gc
from ..tokenizer import DacTokenizer
from pathlib import Path
from audiotools import AudioSignal

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


def test_encode_from_file(tokenizer):
    audio_file_path = SAMPLE_AUDIO_DIR / "sa1.wav"
    compressed = tokenizer.encode(audio_file_path)
    assert isinstance(compressed, dac.DACFile)
    assert compressed.codes is not None
    codes = compressed.codes
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
    compressed = tokenizer.encode(audio_tensor)
    assert isinstance(compressed, dac.DACFile)
    assert compressed.codes is not None


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


def test_decode_from_dacfile(tokenizer):
    dac_file_path = SAMPLE_AUDIO_DIR / "sa1.dac"
    reconstructed_signal = tokenizer.decode(dac_file_path)
    assert isinstance(reconstructed_signal, AudioSignal)
    assert reconstructed_signal.audio_data is not None


def test_decode_from_tensor(tokenizer):
    audio_file_path = SAMPLE_AUDIO_DIR / "sa1.wav"
    audio_signal = AudioSignal(audio_file_path)
    compressed = tokenizer.model.compress(audio_signal)
    codes = compressed.codes  # Shape: [B, N, T]

    # Move codes to cuda if available
    device = next(tokenizer.model.parameters()).device
    codes = codes.to(device)

    z, _, _ = tokenizer.model.quantizer.from_codes(
        codes
    )  # Shape: [B, D, T] where D is 1024 = latent_dim = encoder_dim * (2 ** len(encoder_rates))
    decoded_output = tokenizer.decode(z)
    assert isinstance(decoded_output, torch.Tensor)
    assert decoded_output.dim() == 3
