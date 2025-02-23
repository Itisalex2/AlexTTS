from datasets import load_dataset, DatasetDict
from typing import Dict
from lingua.tokenizer import Tokenizer
from .tokenizer import MisakiTokenizer, DacTokenizer, create_dac_tokenizer_model
import torch


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
        sample["text_tokens"] = tokenizer.encode(
            sample["text"], add_bos=True, add_eos=True
        )
        return sample

    return dataset.map(process_sample)


def apply_audio_tokenizer(dataset: DatasetDict, tokenizer: Tokenizer) -> DatasetDict:
    def process_sample(sample):
        audio_tensor = torch.tensor(sample["audio"]["array"])
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

        assert audio_tensor.dim() == 3, "Tensor input must have 3 dimensions"
        tokenized_audio = tokenizer.encode(audio_tensor).codes.squeeze(0)
        assert tokenized_audio.dim() == 2, (
            f"Codes should have 2 dimensions: (num_codebooks, timesteps), but it has size {tokenized_audio.size()}"
        )

        sample["audio_tokens"] = tokenized_audio

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
    AUDIO_SAMPLING_RATE = "16khz"
    misaki_tokenizer = MisakiTokenizer()
    dac_model = create_dac_tokenizer_model(AUDIO_SAMPLING_RATE)
    dac_tokenizer = DacTokenizer(dac_model)

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
    gigaspeech = apply_text_tokenizer(gigaspeech, misaki_tokenizer)
    gigaspeech = apply_audio_tokenizer(gigaspeech, dac_tokenizer)

    # print(gigaspeech["train"][0])


if __name__ == "__main__":
    main()
