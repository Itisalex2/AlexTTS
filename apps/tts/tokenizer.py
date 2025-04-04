import logging
import string
from pathlib import Path
from typing import Dict, List, Tuple, Union

import dac
import torch
from audiotools import AudioSignal
from misaki import en
from torch import Tensor

from lingua.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class MisakiTokenizer(Tokenizer):
    def __init__(self, use_transformer: bool = False, british: bool = False):
        self.phonemizer = en.G2P(trf=use_transformer, british=british, fallback=None)
        self.phoneme_dict = self._build_phoneme_dict()
        self.punctuation_dict = self._build_puncutation_dict()
        self.whitespace_dict = {w: i + 300 for i, w in enumerate(string.whitespace)}
        self.special_tokens_dict = self._build_special_tokens_dict()
        self.pad_id = 0
        self.vocab_size = 306

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        phonemes, _ = self.phonemizer(text)
        tokens = self._phoneme_to_int(
            phonemes
        )  # Phonemes include punctuation for Misaki

        if add_bos:
            tokens.insert(0, self.special_tokens_dict["SOS"])
        if add_eos:
            tokens.append(self.special_tokens_dict["EOS"])

        return tokens

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

    def get_token_offsets(self, text: str) -> Tuple[List[str], List[int]]:
        raise NotImplementedError

    def _phoneme_to_int(self, phonemes: str) -> List[int]:
        ids = []
        for p in phonemes:
            if p in self.phoneme_dict:
                ids.append(self.phoneme_dict[p])
            elif p in self.punctuation_dict:
                ids.append(self.punctuation_dict[p])
            elif p in self.whitespace_dict:
                ids.append(self.whitespace_dict[p])
            elif p in self.special_tokens_dict:
                ids.append(self.special_tokens_dict[p])
            else:
                raise Exception(
                    f"Character: {p} not in phoneme, puncutation, whitespace, or special tokens dictionary!"
                )

        return ids

    def _build_puncutation_dict(self) -> Dict[str, int]:
        dict = {p: i + 200 for i, p in enumerate(string.punctuation)}
        assert len(dict) < 50

        # Characters that aren't in the standard punctuation dictionary
        dict["“"] = 250
        dict["”"] = 251
        dict["❓"] = 252
        dict["—"] = 253

        return dict

    def _build_phoneme_dict(self) -> Dict[str, int]:
        phoneme_dict = {
            # Stress Marks
            "ˈ": 1,
            "ˌ": 2,
            # Shared IPA Consonants
            "b": 3,
            "d": 4,
            "f": 5,
            "h": 6,
            "j": 7,
            "k": 8,
            "l": 9,
            "m": 10,
            "n": 11,
            "p": 12,
            "s": 13,
            "t": 14,
            "v": 15,
            "w": 16,
            "z": 17,
            "ɡ": 18,
            "ŋ": 19,
            "ɹ": 20,
            "ʃ": 21,
            "ʒ": 22,
            "ð": 23,
            "θ": 24,
            # Consonant Clusters
            "ʤ": 25,
            "ʧ": 26,
            # Shared IPA Vowels
            "ə": 27,
            "i": 28,
            "u": 29,
            "ɑ": 30,
            "ɔ": 31,
            "ɛ": 32,
            "ɜ": 33,
            "ɪ": 34,
            "ʊ": 35,
            "ʌ": 36,
            # Shared Dipthong Vowels
            "A": 37,
            "I": 38,
            "W": 39,
            "Y": 40,
            # Custom Vowel
            "ᵊ": 41,
            # American-only Phonemes
            "æ": 42,
            "O": 43,
            "ᵻ": 44,
            "ɾ": 45,
            # British-only Phonemes
            "a": 46,
            "Q": 47,
            "ɒ": 48,
            "ː": 49,
            # Extra that isn't in https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
            "ɐ": 50,
        }

        return phoneme_dict

    def _build_special_tokens_dict(self) -> Dict[str, int]:
        dict = {"SOS": 100, "EOS": 101}
        return dict


class DacTokenizer(Tokenizer):
    def __init__(self, model: dac.DAC):
        self.model = model
        self.pad_id = 0
        self.bos_id = 1024
        self.eos_id = 1025
        self.vocab_size = 1024 + 1 + 1

    def encode(
        self,
        audio_input: Union[str, Path, Tensor],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Tensor:
        if isinstance(audio_input, (str, Path)):
            signal = AudioSignal(audio_input)
        elif isinstance(audio_input, Tensor):
            if audio_input.dim() != 3:
                raise ValueError("Expected tensor with 3 dimensions.")
            signal = AudioSignal(audio_input, sample_rate=self.model.sample_rate)
        else:
            raise TypeError(
                "Unsupported audio input type. Expected file path or torch.Tensor."
            )

        signal.to(self.model.device)
        x = self.model.preprocess(
            signal.audio_data, signal.sample_rate
        )  # [B, 1, T] mono
        _, codes, _, _, _ = self.model.encode(x)  # [B, Q, T] codes

        if add_bos:
            bos_tokens = torch.full((codes.size(0), codes.size(1), 1), self.bos_id).to(
                self.model.device
            )
            codes = torch.cat([bos_tokens, codes], dim=-1)
        if add_eos:
            eos_tokens = torch.full((codes.size(0), codes.size(1), 1), self.eos_id).to(
                self.model.device
            )
            codes = torch.cat([codes, eos_tokens], dim=-1)

        return codes

    def decode(
        self, audio_input: Union[str, Path, dac.DACFile, Tensor]
    ) -> Union[AudioSignal, Tensor]:
        if isinstance(audio_input, (str, Path, dac.DACFile)):
            reconstructed_signal = self.model.decompress(audio_input)  # Audio signal
            return reconstructed_signal
        elif isinstance(audio_input, Tensor):
            if audio_input.dim() != 3:
                raise ValueError(f"Codes must be 3D [B, Q, T], got {audio_input.shape}")

            z, _, _ = self.model.quantizer.from_codes(audio_input)
            return self.model.decode(z)
        else:
            raise TypeError(
                "Unsupported compressed input type. Expected dac.DACFile or torch.Tensor."
            )

    def get_token_offsets(self):
        raise NotImplementedError


def create_dac_tokenizer_model(model_type: str = "16khz") -> dac.DAC:
    try:
        model_path = dac.utils.download(model_type=model_type)
    except Exception as e:
        raise RuntimeError(f"Failed to download or load the DAC model: {e}")
    model = dac.DAC.load(model_path)
    print("DAC model sample rate:", model.sample_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model
