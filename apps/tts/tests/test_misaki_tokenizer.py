import pytest
from ..tokenizer import MisakiTokenizer


@pytest.fixture
def tokenizer() -> MisakiTokenizer:
    tok = MisakiTokenizer(use_transformer=False)
    return tok


def test_phoneme_dict(tokenizer: MisakiTokenizer):
    assert tokenizer.phoneme_dict.get("ˈ") == 1
    assert tokenizer.phoneme_dict.get("b") == 3
    assert tokenizer.phoneme_dict.get("ʤ") == 25
    assert tokenizer.phoneme_dict.get("ə") == 27
    assert tokenizer.phoneme_dict.get("A") == 37
    assert tokenizer.phoneme_dict.get("ᵊ") == 41
    assert tokenizer.phoneme_dict.get("æ") == 42
    assert tokenizer.phoneme_dict.get("a") == 46


def test_punctuation_dict(tokenizer: MisakiTokenizer):
    assert tokenizer.punctuation_dict.get("!") == 200
    assert tokenizer.punctuation_dict.get(",") == 211
    assert tokenizer.punctuation_dict.get("~") == 231


def test_whitespace_dict(tokenizer: MisakiTokenizer):
    assert tokenizer.whitespace_dict.get(" ") == 300
    assert tokenizer.whitespace_dict.get("\t") == 301
    assert tokenizer.whitespace_dict.get("\n") == 302
    assert tokenizer.whitespace_dict.get("\r") == 303
    assert tokenizer.whitespace_dict.get("\x0b") == 304
    assert tokenizer.whitespace_dict.get("\x0c") == 305


def test_special_tokens_dict(tokenizer: MisakiTokenizer):
    assert tokenizer.special_tokens_dict.get("SOS") == 100
    assert tokenizer.special_tokens_dict.get("EOS") == 101
    assert tokenizer.special_tokens_dict.get("PAD") == 102


def test_encode(tokenizer: MisakiTokenizer):
    text = "[Misaki](/misˈɑki/) is a G2P engine."
    ids = [
        100,
        10,
        28,
        13,
        1,
        30,
        8,
        28,
        300,
        34,
        17,
        300,
        50,
        300,
        25,
        1,
        28,
        14,
        27,
        12,
        1,
        28,
        300,
        1,
        32,
        11,
        25,
        27,
        11,
        213,
        101,
    ]
    assert tokenizer.encode(text) == ids

    text = """ 
    Lorem Ipsum comes from a latin text written in 45BC by Roman statesman, lawyer, scholar, and philosopher, Marcus Tullius Cicero. The text is titled "de Finibus Bonorum et Malorum" which means "The Extremes of Good and Evil". The most common form of Lorem ipsum is the following: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. The text is a corrupted version of the original and therefore does not mean anything in particular. The book however where it originates discusses the philosophical views of Epicureanism, Stoicism, and the Platonism of Antiochus of Ascalon. Lorem ipsum is widely in use since the 14th century and up to today as the default dummy "random" text of the typesetting and web development industry. In fact not only it has survived the test of time but it thrived and can be found in many software products, from Microsoft Word to WordPress. 
    """

    assert tokenizer.encode(text) is not None
