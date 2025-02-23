from datasets import load_dataset, DatasetDict
from typing import Dict
from lingua.tokenizer import Tokenizer
from .tokenizer import MisakiTokenizer


def get_useful_fields(dataset: DatasetDict, columns_to_keep: list[str]) -> DatasetDict:
    all_columns = dataset["train"].column_names
    for col in columns_to_keep:
        if col not in all_columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
    return dataset.remove_columns(columns_to_remove)


def preprocess_text(text: str, punctuation_mappings: Dict[str, str]) -> str:
    processed_text = text
    for tag, symbol in punctuation_mappings.items():
        processed_text = processed_text.replace(tag, symbol)

    return processed_text


def map_punctuation_in_dataset(
    dataset: DatasetDict, punctuation_mappings: Dict[str, str]
) -> DatasetDict:
    def process_sample(sample):
        sample["text"] = preprocess_text(sample["text"], punctuation_mappings)
        return sample

    return dataset.map(process_sample)


def apply_text_tokenizer(dataset: DatasetDict, tokenizer: Tokenizer) -> DatasetDict:
    def process_sample(sample):
        sample["text"] = tokenizer.encode(sample["text"], add_bos=True, add_eos=True)
        return sample

    return dataset.map(process_sample)


def main():
    # For the gigaspeech dataset ONLY
    punctuation_mappings = {
        "<COMMA>": ",",
        "<PERIOD>": ".",
        "<QUESTIONMARK>": "?",
        "<EXCLAMATIONPOINT>": "!",
    }
    misakiToknizer = MisakiTokenizer()

    gigaspeech = load_dataset("speechcolab/gigaspeech", "xs")
    gigaspeech = DatasetDict(
        {
            "train": gigaspeech["train"].select(range(10)),
            "validation": gigaspeech["validation"].select(range(10)),
            "test": gigaspeech["test"].select(range(10)),
        }
    )
    COLUMNS_TO_KEEP = ["text", "audio"]
    gigaspeech = get_useful_fields(gigaspeech, COLUMNS_TO_KEEP)
    gigaspeech = map_punctuation_in_dataset(gigaspeech, punctuation_mappings)
    gigaspeech = apply_text_tokenizer(gigaspeech, misakiToknizer)

    print(gigaspeech["train"][0:10])


if __name__ == "__main__":
    main()
