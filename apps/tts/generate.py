import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
import torchaudio
from tqdm import tqdm

from lingua.tokenizer import Tokenizer

from .tokenizer import DacTokenizer, MisakiTokenizer, create_dac_tokenizer_model
from .transformer import TTSTransformer, TTSTransformerArgs
from .train import TrainerState, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    output_path: Path = Path("generated_audio.wav")
    model_path: Optional[Path] = None


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs - sorted_probs > p
    sorted_probs[mask] = 0.0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
    _, unsort_indices = torch.sort(sorted_indices, dim=-1)
    modified_probs = torch.gather(sorted_probs, dim=-1, index=unsort_indices)
    return modified_probs


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    topk_values, _ = torch.topk(probs, k, dim=-1)
    min_topk = topk_values[..., -1:]
    mask = probs < min_topk
    probs = probs.masked_fill(mask, 0.0)
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


class TTSGenerator:
    def __init__(
        self,
        transformer: TTSTransformer,
        text_tokenizer: Tokenizer,
        audio_tokenizer: Tokenizer,
        config: GenerateConfig,
    ):
        self.transformer = transformer
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.config = config

    @torch.no_grad()
    def generate(self, text: str):
        text_tokens = torch.tensor(
            self.text_tokenizer.encode(text), device=self.config.device
        ).unsqueeze(0)  # [1, text_t]

        assert self.config.batch_size == 1, (
            "For now only consider simple case: batch size = 1"
        )
        audio_tokens = torch.full(
            (self.config.batch_size, self.transformer.num_quantizers, 1),
            self.audio_tokenizer.bos_id,
            dtype=torch.long,
            device=self.config.device,
        )  # [1, Q, 1 (timestep = 1 initially with bos)]

        assert audio_tokens.shape == (
            self.config.batch_size,
            self.transformer.num_quantizers,
            1,
        )

        for _ in tqdm(range(self.config.max_tokens), desc="Generating"):
            logits = self.transformer(text_tokens, audio_tokens)
            next_logits = logits[:, :, -1]  # Get last timestep

            next_token = self._sample_next(next_logits)

            if (next_token == self.audio_tokenizer.eos_id).any():
                break

            audio_tokens = torch.cat([audio_tokens, next_token], dim=-1)

        audio_tokens = self.post_process_audio_tokens(audio_tokens)
        waveform = self.audio_tokenizer.decode(audio_tokens)  # [B, 1, audio_t]
        waveform = waveform.squeeze(1).cpu()

        assert isinstance(waveform, torch.Tensor), "Decoder should return tensor"
        assert waveform.ndim == 2, f"Waveform should be [B, T], got {waveform.shape}"

        torchaudio.save(
            self.config.output_path, waveform, self.audio_tokenizer.model.sample_rate
        )

    def _sample_next(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, Q, vocab_size]
        if self.config.temperature == 0:
            return logits.argmax(-1).unsqueeze(-1)  # [B, Q, 1]

        probs = torch.softmax(logits / self.config.temperature, dim=-1)

        if self.config.top_p:
            probs = sample_top_p(probs, self.config.top_p)  # [B, Q, vocab_size]
        if self.config.top_k:
            probs = sample_top_k(probs, self.config.top_k)  # [B, Q, vocab_size]

        batch_size, num_codebooks, _ = probs.shape
        return torch.multinomial(
            probs.flatten(0, 1),  # [B*Q, vocab_size]
            1,
        ).view(batch_size, num_codebooks, 1)  # [B, Q, 1]

    def post_process_audio_tokens(self, audio_tokens: Tensor) -> Tensor:
        audio_tokens = audio_tokens[:, :, 1:]  # Don't include the start token
        audio_tokens[audio_tokens == self.audio_tokenizer.bos_id] = 0
        return audio_tokens


def load_checkpoint(model: torch.nn.Module, config: GenerateConfig):
    checkpoint = torch.load(
        config.model_path, map_location=config.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    logger.info(f"Loaded model dict from: {config.model_path}")


if __name__ == "__main__":
    BEST_CHECKPOINT = Path("checkpoints/best.pt")
    # BEST_CHECKPOINT = Path("checkpoints/checkpoint_epoch_0016.pt")
    # BEST_CHECKPOINT = Path("one_example_checkpoints/checkpoint_epoch_0300.pt")
    config = GenerateConfig(
        temperature=0.5,
        top_k=50,
        top_p=None,
        max_tokens=200,
        model_path=BEST_CHECKPOINT,
    )

    misaki_tokenizer = MisakiTokenizer()
    dac_model = create_dac_tokenizer_model()
    dac_tokenizer = DacTokenizer(dac_model)

    ttsTransformerArgs = TTSTransformerArgs(
        text_vocab_size=misaki_tokenizer.vocab_size,
        audio_vocab_size=dac_tokenizer.vocab_size,
    )

    model = TTSTransformer(ttsTransformerArgs).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}"
    )

    if config.model_path:
        load_checkpoint(model, config)

    generator = TTSGenerator(model, misaki_tokenizer, dac_tokenizer, config)

    while True:
        prompt = input("Please enter your text prompt. Press enter to exit.\n")
        if prompt == "":
            break
        generator.generate(text=prompt)
