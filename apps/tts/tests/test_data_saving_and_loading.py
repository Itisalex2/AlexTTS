import pytest
import torch
from datasets import DatasetDict
import shutil
import numpy as np
from ..data import save_to_pt_files, TTSDataset, TTSCollator


@pytest.fixture
def mock_dataset():
    return DatasetDict(
        {
            "train": [
                {
                    "text_tokens": [1, 2, 3],
                    "audio_tokens": [[4, 5, 6], [7, 8, 9]],
                    "text": "hello world",
                    "audio": {
                        "array": np.array([0.1, 0.2, 0.3]),
                        "sampling_rate": 16000,
                    },
                }
            ],
            "validation": [
                {
                    "text_tokens": [10, 11, 12],
                    "audio_tokens": [[13, 14, 15], [16, 17, 18]],
                    "text": "test audio",
                    "audio": {
                        "array": np.array([0.4, 0.5, 0.6]),
                        "sampling_rate": 16000,
                    },
                }
            ],
        }
    )


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "test_data"
    yield data_dir

    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture
def dataset_on_disk(mock_dataset, temp_data_dir):
    splits = ["train", "validation"]
    save_to_pt_files(mock_dataset, splits, temp_data_dir)
    return temp_data_dir


def test_save_to_pt_files(mock_dataset, temp_data_dir):
    splits = ["train", "validation"]

    save_to_pt_files(mock_dataset, splits, temp_data_dir)

    for split in splits:
        split_dir = temp_data_dir / split
        assert split_dir.exists(), f"Directory for {split} was not created"

    for split in splits:
        sample_path = temp_data_dir / split / "sample_0.pt"
        assert sample_path.exists(), f"Sample file for {split} was not created"

        saved_data = torch.load(sample_path, weights_only=False)

        expected_keys = {
            "text_tokens",
            "audio_tokens",
            "text",
            "audio",
            "sampling_rate",
        }
        assert all(key in saved_data for key in expected_keys), (
            "Missing keys in saved data"
        )

        original_sample = mock_dataset[split][0]
        assert torch.equal(
            saved_data["text_tokens"],
            torch.tensor(original_sample["text_tokens"], dtype=torch.long),
        )
        assert torch.equal(
            saved_data["audio_tokens"],
            torch.tensor(original_sample["audio_tokens"], dtype=torch.long),
        )
        assert saved_data["text"] == original_sample["text"]
        assert np.array_equal(saved_data["audio"], original_sample["audio"]["array"])
        assert saved_data["sampling_rate"] == original_sample["audio"]["sampling_rate"]


def test_save_to_pt_files_empty_dataset(temp_data_dir):
    empty_dataset = DatasetDict({"train": [], "validation": []})
    splits = ["train", "validation"]

    save_to_pt_files(empty_dataset, splits, temp_data_dir)

    for split in splits:
        split_dir = temp_data_dir / split
        assert split_dir.exists(), f"Directory for {split} was not created"


def test_save_to_pt_files_invalid_path():
    invalid_dataset = DatasetDict(
        {
            "train": [
                {
                    "text_tokens": [1, 2, 3],
                    "audio_tokens": [[4, 5, 6]],
                    "text": "test",
                    "audio": {"array": np.array([0.1, 0.2]), "sampling_rate": 16000},
                }
            ]
        }
    )

    with pytest.raises(Exception):
        save_to_pt_files(invalid_dataset, ["train"], "/nonexistent/path")


def test_tts_dataset_initialization(dataset_on_disk):
    split = "train"
    dataset = TTSDataset(split, dataset_on_disk)
    assert len(dataset) == 1, "Dataset should have one sample"

    sample = dataset[0]
    print(sample)
    expected_keys = {"text_tokens", "audio_tokens", "text", "audio", "sampling_rate"}
    assert all(key in sample for key in expected_keys), "Missing keys in dataset sample"


def test_tts_dataset_getitem(dataset_on_disk):
    split = "train"
    dataset = TTSDataset(split, dataset_on_disk)
    sample = dataset[0]

    assert isinstance(sample["text_tokens"], torch.Tensor)
    assert isinstance(sample["audio_tokens"], torch.Tensor)
    assert isinstance(sample["text"], str)
    assert isinstance(sample["audio"], np.ndarray)
    assert isinstance(sample["sampling_rate"], int)

    assert torch.equal(sample["text_tokens"], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(
        sample["audio_tokens"], torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.long)
    )
    assert sample["text"] == "hello world"
    assert np.array_equal(sample["audio"], np.array([0.1, 0.2, 0.3]))
    assert sample["sampling_rate"] == 16000


def test_tts_collator():
    text_pad_id = audio_pad_id = 0
    collator = TTSCollator(text_pad_id, audio_pad_id)

    batch = [
        {
            "text_tokens": torch.tensor([1, 2, 3], dtype=torch.long),
            "audio_tokens": torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.long),
            "text": "hello world",
            "audio": np.array([0.1, 0.2, 0.3]),
            "sampling_rate": 16000,
        },
        {
            "text_tokens": torch.tensor([10, 11], dtype=torch.long),
            "audio_tokens": torch.tensor([[13, 14], [16, 17]], dtype=torch.long),
            "text": "test audio",
            "audio": np.array([0.4, 0.5]),
            "sampling_rate": 16000,
        },
    ]

    collated = collator(batch)

    assert isinstance(collated, dict)
    assert "text_tokens" in collated
    assert "audio_tokens" in collated

    assert collated["text_tokens"].shape == (2, 3)
    assert collated["audio_tokens"].shape == (2, 2, 3)

    assert collated["text_tokens"].dtype == torch.long
    assert collated["audio_tokens"].dtype == torch.long

    assert torch.equal(collated["text_tokens"][0], torch.tensor([1, 2, 3]))
    assert torch.equal(collated["text_tokens"][1], torch.tensor([10, 11, 0]))


def test_tts_collator_empty_batch():
    text_pad_id = audio_pad_id = 0
    collator = TTSCollator(text_pad_id, audio_pad_id)
    with pytest.raises(RuntimeError):
        collator([])
