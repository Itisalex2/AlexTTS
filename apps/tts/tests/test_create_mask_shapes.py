import pytest
import torch

from ..transformer import TTSTransformer, TTSTransformerArgs


@pytest.fixture
def transformer() -> TTSTransformer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = TTSTransformerArgs(
        dim=32,
        n_layers=2,
        n_heads=4,
        text_vocab_size=20,
        audio_vocab_size=30,
        num_quantizers=3,
        max_seqlen=64,
    )
    model = TTSTransformer(args).to(device)
    return model


@pytest.mark.parametrize(
    "batch_size,text_len,audio_len",
    [
        (1, 3, 4),
        (2, 5, 6),
        (3, 1, 1),
    ],
)
def test_create_mask_shapes(
    transformer: TTSTransformer, batch_size, text_len, audio_len
):
    device = next(transformer.parameters()).device
    num_quantizers = transformer.num_quantizers

    text_tokens = torch.randint(
        low=0,
        high=transformer.text_vocab_size,
        size=(batch_size, text_len),
        device=device,
        dtype=torch.long,
    )

    audio_tokens = torch.randint(
        low=0,
        high=transformer.audio_vocab_size,
        size=(batch_size, num_quantizers, audio_len),
        device=device,
        dtype=torch.long,
    )

    mask = transformer._create_mask(text_tokens, audio_tokens)
    assert mask.dtype == torch.bool, "Mask should be a boolean tensor"

    total_len = text_len + audio_len
    expected_shape = (batch_size, 1, total_len, total_len)
    assert mask.shape == expected_shape, (
        f"Mask shape mismatch. Expected {expected_shape}, got {mask.shape}."
    )

    assert mask.any(), "Mask should have at least some True values."
    assert not mask.all(), (
        "Mask should not be all True; there should be some padding or causal constraints."
    )


def test_padding_mask_values(transformer: TTSTransformer):
    device = next(transformer.parameters()).device

    text_pad_id = transformer.text_pad_id
    audio_pad_id = transformer.audio_pad_id

    text_tokens = torch.tensor(
        [[1, 2, text_pad_id, text_pad_id], [text_pad_id, 3, 4, text_pad_id]],
        dtype=torch.long,
        device=device,
    )

    audio_tokens = torch.tensor(
        [
            [
                [5, 6, audio_pad_id],
                [7, 8, audio_pad_id],
                [9, 10, audio_pad_id],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
        ],
        dtype=torch.long,
        device=device,
    )

    text_mask = transformer._create_text_padding_mask(text_tokens)
    expected_text_mask = torch.tensor(
        [[True, True, False, False], [False, True, True, False]],
        dtype=torch.bool,
        device=device,
    )
    assert torch.equal(text_mask, expected_text_mask), (
        f"Text padding mask mismatch!\nExpected:\n{expected_text_mask}\nGot:\n{text_mask}"
    )

    audio_mask = transformer._create_audio_padding_mask(audio_tokens)
    expected_audio_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ],
        dtype=torch.bool,
        device=device,
    )
    assert torch.equal(audio_mask, expected_audio_mask), (
        f"Audio padding mask mismatch!\nExpected:\n{expected_audio_mask}\nGot:\n{audio_mask}"
    )


def test_final_mask_causality(transformer: TTSTransformer):
    """
    Ensures that the final mask from _create_mask respects:
    - Padding (False where padded)
    - Causality for audio portion
    """
    device = next(transformer.parameters()).device
    text_pad_id = transformer.text_pad_id

    text_tokens = torch.tensor([[1, 2, text_pad_id]], device=device)
    audio_tokens = torch.tensor([[[5, 6], [7, 8], [9, 10]]], device=device)

    mask_4d = transformer._create_mask(text_tokens, audio_tokens)
    assert mask_4d.shape == (1, 1, 5, 5), (
        f"Should be (B=1, 1, T=5, T=5). Got {mask_4d.shape}"
    )

    mask_2d = mask_4d[0, 0]  # shape (5, 5)

    assert not mask_2d[2].any(), (
        "Row 2 should be entirely False because text index 2 is PAD."
    )
    assert not mask_2d[:, 2].any(), (
        "Column 2 should be entirely False because text index 2 is PAD."
    )

    assert mask_2d[4, 3].item() == 1, (
        "Position (row=4,col=3) should be True if col=3 < row=4 in audio causal block."
    )
    assert mask_2d[3, 4].item() == 0, (
        "Position (row=3,col=4) should be False for causal reasons."
    )
