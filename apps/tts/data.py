from datasets import load_dataset, DatasetDict
from typing import Dict, List
from lingua.tokenizer import Tokenizer
from .tokenizer import MisakiTokenizer, DacTokenizer, create_dac_tokenizer_model
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


class TTSDataset(Dataset):
    def __init__(self, split: str, data_dir: Path):
        self.data_dir = data_dir / split
        self.file_paths = list(self.data_dir.glob("*.pt"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=False)
        return {
            "text_tokens": data["text_tokens"],
            "audio_tokens": data["audio_tokens"],
            "text": data["text"],
            "audio": data["audio"],
            "sampling_rate": data["sampling_rate"],
        }


class TTSCollator:
    def __init__(self, text_pad_id, audio_pad_id):
        self.text_pad_id = text_pad_id
        self.audio_pad_id = audio_pad_id

    def __call__(self, batch):
        text_tokens = [sample["text_tokens"] for sample in batch]
        text_padded = pad_sequence(
            text_tokens, batch_first=True, padding_value=self.text_pad_id
        )

        audio_tokens = [
            sample["audio_tokens"].permute(1, 0) for sample in batch
        ]  # Permute: [Q, T] -> [T, Q] for each sample. Result: [B, T, Q]
        audio_padded = pad_sequence(
            audio_tokens, batch_first=True, padding_value=self.audio_pad_id
        )  # [B, T, Q]
        audio_padded = audio_padded.permute(0, 2, 1)  # [B, Q, T]

        text_mask = (text_padded != self.text_pad_id).float()
        audio_mask = (audio_padded != self.audio_pad_id).all(dim=1).float()
        combined_mask = torch.cat([text_mask, audio_mask], dim=1)

        return {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "attention_mask": combined_mask,
            "texts": [sample["text"] for sample in batch],
            "audio": [sample["audio"] for sample in batch],
        }


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


def save_to_pt_files(dataset: DatasetDict, splits: List[str], data_dir: Path):
    for split in splits:
        (data_dir / split).mkdir(parents=True, exist_ok=True)

    for split in splits:
        for idx, sample in enumerate(dataset[split]):
            text_tokens = torch.tensor(sample["text_tokens"], dtype=torch.long)
            audio_tokens = torch.tensor(sample["audio_tokens"], dtype=torch.long)

            torch.save(
                {
                    "text_tokens": text_tokens,
                    "audio_tokens": audio_tokens,
                    "text": sample["text"],
                    "audio": sample["audio"]["array"],
                    "sampling_rate": sample["audio"]["sampling_rate"],
                },
                data_dir / split / f"sample_{idx}.pt",
            )


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
    DATA_DIR = Path("data")
    SPLITS = ["train", "validation", "test"]

    gigaspeech = load_dataset("speechcolab/gigaspeech", "xs")
    gigaspeech = DatasetDict(
        {
            "train": gigaspeech["train"].select(range(7)),
            "validation": gigaspeech["validation"].select(range(7)),
            "test": gigaspeech["test"].select(range(7)),
        }
    )
    COLUMNS_TO_KEEP = ["text", "audio"]
    gigaspeech = get_useful_fields(gigaspeech, COLUMNS_TO_KEEP)
    gigaspeech = map_punctuation_in_dataset(gigaspeech, punctuation_mappings)
    gigaspeech = apply_text_tokenizer(gigaspeech, misaki_tokenizer)
    gigaspeech = apply_audio_tokenizer(gigaspeech, dac_tokenizer)

    save_to_pt_files(gigaspeech, SPLITS, DATA_DIR)


if __name__ == "__main__":
    main()
