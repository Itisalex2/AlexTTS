import pytest
from ..data import (
    preprocess_text,
    get_useful_fields,
    map_punctuation_in_dataset,
    apply_text_tokenizer,
)
from datasets import DatasetDict, Dataset


@pytest.fixture
def sample_dataset():
    data = {
        "train": Dataset.from_dict(
            {
                "text": ["hello", "world"],
                "audio": [1.0, 2.0],
                "extra": [True, False],
                "metadata": ["a", "b"],
            }
        )
    }
    return DatasetDict(data)


@pytest.fixture
def punctuation_dataset():
    data = {
        "train": Dataset.from_dict(
            {
                "text": [
                    "Hello<COMMA> world<PERIOD>",
                    "Test<EXCLAMATIONPOINT> OK<QUESTIONMARK>",
                    "No punctuation here",
                ],
                "audio": [1.0, 2.0, 3.0],
            }
        )
    }
    return DatasetDict(data)


# A dummy tokenizer for testing purposes.
class DummyTokenizer:
    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        # For this dummy, we'll simply convert each character's ordinal value.
        tokens = [ord(c) for c in text]
        if add_bos:
            tokens.insert(0, 100)
        if add_eos:
            tokens.append(101)
        return tokens


@pytest.fixture
def dummy_dataset() -> DatasetDict:
    data = {
        "train": Dataset.from_dict(
            {
                "text": ["hello", "world", "test case"],
                "audio": [1.0, 2.0, 3.0],
                "extra": [True, False, True],
            }
        )
    }
    return DatasetDict({"train": data["train"]})


def test_get_useful_fields_keeps_specified_columns(sample_dataset):
    columns_to_keep = ["text", "audio"]
    result = get_useful_fields(sample_dataset, columns_to_keep)

    assert set(result["train"].column_names) == set(columns_to_keep)
    assert "extra" not in result["train"].column_names
    assert "metadata" not in result["train"].column_names


def test_get_useful_fields_raises_error_for_missing_column(sample_dataset):
    columns_to_keep = ["text", "nonexistent_column"]

    with pytest.raises(ValueError) as exc_info:
        get_useful_fields(sample_dataset, columns_to_keep)

    assert "Column 'nonexistent_column' not found in dataset" in str(exc_info.value)


def test_get_useful_fields_keeps_data_intact(sample_dataset):
    columns_to_keep = ["text", "audio"]
    result = get_useful_fields(sample_dataset, columns_to_keep)

    assert result["train"]["text"] == sample_dataset["train"]["text"]
    assert result["train"]["audio"] == sample_dataset["train"]["audio"]


def test_get_useful_fields_with_empty_columns_list(sample_dataset):
    columns_to_keep = []
    result = get_useful_fields(sample_dataset, columns_to_keep)

    assert len(result["train"].column_names) == 0


def test_get_useful_fields_with_all_columns(sample_dataset):
    columns_to_keep = ["text", "audio", "extra", "metadata"]
    result = get_useful_fields(sample_dataset, columns_to_keep)

    assert set(result["train"].column_names) == set(
        sample_dataset["train"].column_names
    )


def test_process_text():
    punctuation_mappings = {
        "<PERIOD>": ".",
        "<COMMA>": ",",
        "<QUESTIONMARK>": "?",
        "<EXCLAMATIONPOINT>": "!",
    }

    text = "Hello<COMMA> world<EXCLAMATIONPOINT> How are you<QUESTIONMARK> This is great<PERIOD>"
    expected_output = "Hello, world! How are you? This is great."

    result = preprocess_text(text, punctuation_mappings)
    assert result == expected_output

    text = "Hi<COMMA><EXCLAMATIONPOINT> Testing<PERIOD><QUESTIONMARK>"
    expected_output = "Hi,! Testing.?"

    result = preprocess_text(text, punctuation_mappings)
    assert result == expected_output

    text = "Hello world"
    expected_output = "Hello world"

    result = preprocess_text(text, punctuation_mappings)
    assert result == expected_output


def test_map_punctuation_in_dataset_replaces_tokens(punctuation_dataset):
    mappings = {
        "<COMMA>": ",",
        "<PERIOD>": ".",
        "<QUESTIONMARK>": "?",
        "<EXCLAMATIONPOINT>": "!",
    }
    processed_ds = map_punctuation_in_dataset(punctuation_dataset, mappings)

    assert processed_ds["train"][0]["text"] == "Hello, world."
    assert processed_ds["train"][1]["text"] == "Test! OK?"
    # A sample with no punctuation tokens should remain unchanged.
    assert processed_ds["train"][2]["text"] == "No punctuation here"


def test_map_punctuation_in_dataset_no_mapping(punctuation_dataset):
    mappings = {}
    processed_ds = map_punctuation_in_dataset(punctuation_dataset, mappings)
    for i, sample in enumerate(punctuation_dataset["train"]):
        assert processed_ds["train"][i]["text"] == sample["text"]


def test_map_punctuation_in_dataset_multiple_occurrences():
    data = {
        "train": Dataset.from_dict(
            {"text": ["Repeat<COMMA> Repeat<COMMA> Repeat<COMMA>"], "audio": [1.0]}
        )
    }
    ds = DatasetDict(data)
    mappings = {"<COMMA>": ","}
    processed_ds = map_punctuation_in_dataset(ds, mappings)
    assert processed_ds["train"][0]["text"] == "Repeat, Repeat, Repeat,"


def test_map_punctuation_in_dataset_missing_text_field():
    data = {"train": Dataset.from_dict({"audio": [1.0, 2.0]})}
    ds = DatasetDict(data)
    mappings = {"<COMMA>": ","}
    with pytest.raises(KeyError):
        _ = map_punctuation_in_dataset(ds, mappings)


def test_apply_text_tokenizer_replaces_text(dummy_dataset):
    tokenizer = DummyTokenizer()
    processed_ds = apply_text_tokenizer(dummy_dataset, tokenizer)

    for i, sample in enumerate(dummy_dataset["train"]):
        expected_tokens = tokenizer.encode(sample["text"], add_bos=True, add_eos=True)
        assert processed_ds["train"][i]["text"] == expected_tokens


def test_apply_text_tokenizer_preserves_non_text_fields(dummy_dataset):
    tokenizer = DummyTokenizer()
    processed_ds = apply_text_tokenizer(dummy_dataset, tokenizer)

    original_audio = dummy_dataset["train"]["audio"]
    processed_audio = processed_ds["train"]["audio"]

    assert original_audio == processed_audio
    assert "extra" in processed_ds["train"].column_names
