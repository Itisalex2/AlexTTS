import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from torch import Tensor

from ..data import apply_audio_tokenizer


class DummyDacTokenizer:
    def encode(self, audio_input: torch.Tensor) -> Tensor:
        codes = torch.tensor([0, 1, 2]).unsqueeze(0).unsqueeze(0)

        return codes


@pytest.fixture
def dummy_audio_dataset() -> DatasetDict:
    audio_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    data = {
        "train": Dataset.from_dict(
            {
                "text": ["dummy"],
                "audio": [
                    {"array": audio_array, "path": "dummy.wav", "sampling_rate": 16000}
                ],
            }
        )
    }
    return DatasetDict(data)


def test_apply_audio_tokenizer_adds_audio_tokens(dummy_audio_dataset):
    dummy_tokenizer = DummyDacTokenizer()
    processed_ds = apply_audio_tokenizer(dummy_audio_dataset, dummy_tokenizer)
    sample = processed_ds["train"][0]

    assert "audio_tokens" in sample, "The sample should have an 'audio_tokens' field."

    expected_codes = [[0, 1, 2]]
    assert sample["audio_tokens"] == expected_codes, (
        f"Expected {expected_codes} but got {sample['audio_tokens']}."
    )
