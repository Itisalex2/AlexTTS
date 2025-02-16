from lingua.tokenizer import Tokenizer
from misaki import en
from typing import List, Tuple, Dict, Union
from audiotools import AudioSignal
from pathlib import Path
from torch import Tensor
import string
import logging
import dac


logger = logging.getLogger(__name__)


class MisakiTokenizer(Tokenizer):
    def __init__(self, use_transformer: bool = False, british: bool = False):
        self.phonemizer = en.G2P(trf=use_transformer, british=british, fallback=None)
        self.phoneme_dict = self._build_phoneme_dict()
        self.punctuation_dict = self._build_puncutation_dict()
        self.whitespace_dict = {w: i + 300 for i, w in enumerate(string.whitespace)}
        self.special_tokens_dict = self._build_special_tokens_dict()

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
        print(phonemes)
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
                    f"Character: {p} not in phoneme or puncutation dictionary!"
                )

        return ids

    def _build_puncutation_dict(self) -> Dict[str, int]:
        dict = {p: i + 200 for i, p in enumerate(string.punctuation)}
        assert len(dict) < 50

        dict["“"] = 250
        dict["”"] = 251
        dict["❓"] = 252

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
        dict = {"SOS": 100, "EOS": 101, "PAD": 102}
        return dict


class DacTokenizer(Tokenizer):
    def __init__(self, model: dac.DAC):
        self.model = model

    def encode(self, audio_input: Union[str, Path, Tensor]) -> dac.DACFile:
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
        compressed = self.model.compress(signal)
        return compressed

    def decode(
        self, compressed_input: Union[str, Path, dac.DACFile, Tensor]
    ) -> Union[AudioSignal, Tensor]:
        if isinstance(compressed_input, (str, Path, dac.DACFile)):
            reconstructed_signal = self.model.decompress(
                compressed_input
            )  # Audio signal
        elif isinstance(compressed_input, Tensor):
            if compressed_input.dim() != 3:
                raise ValueError("Expected tensor with 3 dimensions.")
            compressed_input = compressed_input.float()
            reconstructed_signal = self.model.decode(compressed_input)  # Tensor
        else:
            raise TypeError(
                "Unsupported compressed input type. Expected dac.DACFile or torch.Tensor."
            )

        return reconstructed_signal

    def get_token_offsets(self):
        raise NotImplementedError
