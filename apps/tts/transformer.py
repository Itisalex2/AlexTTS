import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from .lingua_transformer_modified import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    cross_entropy,
)


@dataclass
class TTSTransformerArgs(BaseTransformerArgs):
    seed: int = 42
    text_vocab_size: int = -1
    text_pad_id: int = 0
    audio_vocab_size: int = -1
    audio_pad_id: int = 0
    num_quantizers: int = 12
    quantizer_max_weight: float = 1
    quantizer_min_weight: float = 0.0
    max_seqlen: int = 4096

    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12


class TTSTransformer(BaseTransformer):
    def __init__(self, args: TTSTransformerArgs):
        super().__init__(args)

        assert args.text_vocab_size > 0, "Please specify a valid text voab size"
        assert args.audio_vocab_size > 0, "Please specify a valid audio voab size"

        self.text_vocab_size = args.text_vocab_size
        self.text_pad_id = args.text_pad_id
        self.audio_vocab_size = args.audio_vocab_size
        self.audio_pad_id = args.audio_pad_id
        self.num_quantizers = args.num_quantizers
        self.dim = args.dim
        self.quantizer_max_weight = args.quantizer_max_weight
        self.quantizer_min_weight = args.quantizer_min_weight

        self.text_embeddings = nn.Embedding(self.text_vocab_size, self.dim)
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.audio_vocab_size, self.dim)
                for _ in range(self.num_quantizers)
            ]
        )

        self.norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.output = nn.Linear(self.dim, self.audio_vocab_size * self.num_quantizers)
        self.init_weights()

    def forward(  # type: ignore[override]
        self,
        text_tokens: Tensor,  # [B, text_t]
        audio_tokens: Tensor,  # [B, Q, audio_t]
        target: Optional[torch.Tensor] = None,  # [B, Q, audio_t]
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        assert text_tokens.dim() == 2, (
            f"Text tokens should have dimensions 2, but got {text_tokens.dim()} instead!"
        )
        text_embeddings = self.text_embeddings(text_tokens)

        assert audio_tokens.dim() == 3, (
            f"Audio tokens should have dimenions 3, but got {audio_tokens.dim()} instead!"
        )
        bsz, q, audio_t = audio_tokens.shape
        assert q == self.num_quantizers, (
            f"Audio tokens quantizers don't match! Expected {self.num_quantizers} but got {q} instead!"
        )

        audio_embeddings = torch.zeros(
            bsz, audio_t, self.dim, device=audio_tokens.device
        )
        for i in range(self.num_quantizers):
            audio_embeddings += self.audio_embeddings[i](audio_tokens[:, i, :])

        assert audio_embeddings.shape == (bsz, audio_t, self.dim), (
            f"Audio embeddings shape don't match! Expected {(bsz, audio_t, self.dim)} but got {audio_embeddings.shape} instead!"
        )

        combined_embeddings = torch.cat([text_embeddings, audio_embeddings], dim=1)

        if not mask:
            mask = self._create_mask(text_tokens, audio_tokens)

        h = super().forward(combined_embeddings, mask=mask, attn_impl=attn_impl)

        audio_not_projected = h[:, text_embeddings.size(1) :, :]  # [B, audio_t, D]

        assert audio_not_projected.shape == (bsz, audio_t, self.dim), (
            f"Audio (not projected) shape doesn't match! Expected {(bsz, audio_t, self.dim)} but got {audio_not_projected.shape} instead!"
        )

        logits = self.output(
            self.norm(audio_not_projected)
        )  # [B, audio_t, Q*vocab_size]

        assert logits.shape == (
            bsz,
            audio_t,
            self.num_quantizers * self.audio_vocab_size,
        ), (
            f"Initial shape doesn't match! Expected {(bsz, audio_t, self.num_quantizers * self.audio_vocab_size)} but got {logits.shape} instead!"
        )

        logits = logits.view(
            bsz, audio_t, self.num_quantizers, self.audio_vocab_size
        ).permute(0, 2, 1, 3)  # => [B, Q, audio_t, vocab_size]

        assert logits.shape == (
            bsz,
            self.num_quantizers,
            audio_t,
            self.audio_vocab_size,
        ), (
            f"Final logits shape doesn't match! Expected {(bsz, self.num_quantizers, audio_t, self.audio_vocab_size)} but got {logits.shape} instead!"
        )

        if target is not None:
            losses = []
            for i in range(self.num_quantizers):
                losses.append(
                    cross_entropy(
                        logits[:, i]
                        .contiguous()
                        .view(-1, self.audio_vocab_size),  # [B*T_audio, vocab_size]
                        target[:, i].contiguous().view(-1),  # [B*T_audio]
                        ignore_index=self.audio_pad_id,
                    )
                )

            total_loss = torch.stack(losses).mean()
            return total_loss, logits

        return logits

    def reset_parameters(self):
        super().reset_parameters()
        init_gain = nn.init.calculate_gain("linear")

        # Text embeddings
        nn.init.xavier_uniform_(self.text_embeddings.weight, gain=init_gain)

        # Audio embeddings
        for emb in self.audio_embeddings:
            nn.init.xavier_uniform_(emb.weight, gain=init_gain)

        # Output layer
        nn.init.xavier_uniform_(
            self.output.weight,
            gain=nn.init.calculate_gain("linear") / math.sqrt(self.num_quantizers),
        )

    def _create_mask(self, text_tokens: Tensor, audio_tokens: Tensor) -> Tensor:
        assert text_tokens.device == audio_tokens.device, (
            "Text tokens and audio tokens must be on the same device!"
        )
        device = text_tokens.device

        text_mask = self._create_text_padding_mask(text_tokens).to(device)
        audio_mask = self._create_audio_padding_mask(audio_tokens).to(device)

        padding_mask = torch.cat([text_mask, audio_mask], dim=1).to(device)  # [B, T]

        batch_size = text_tokens.shape[0]
        seq_len = padding_mask.shape[1]

        pad_query = padding_mask.unsqueeze(2)  # shape (B, T, 1)
        pad_key = padding_mask.unsqueeze(1)  # shape (B, 1, T)
        padding_mask = pad_query & pad_key  # shape (B, T, T)

        attention_mask = self._create_attention_mask(text_tokens, audio_tokens).to(
            device
        )  # [B, T, T]

        mask = padding_mask & attention_mask  # [B, T, T]

        assert mask.shape == (batch_size, seq_len, seq_len), (
            f"Mask at this point should be [B, T, T], but got {mask.shape} instead"
        )
        mask = mask.unsqueeze(1)  # [B, 1, T, T]
        assert mask.shape == (batch_size, 1, seq_len, seq_len), (
            f"Final mask should be [B, 1, T, T], but got {mask.shape} instead"
        )

        return mask

    def _create_text_padding_mask(self, text_padded: Tensor) -> Tensor:
        assert text_padded.dim() == 2, (
            f"Padded text tokens should have dimensions 2, but got {text_padded.dim()} instead"
        )
        return (
            text_padded != self.text_pad_id
        )  # [B, text_length] where True means attend to

    def _create_audio_padding_mask(self, audio_padded: Tensor) -> Tensor:
        # Check first quantizer's time steps for padding
        assert audio_padded.dim() == 3, (
            f"Padded audio tokens should have dimensions 2, but got {audio_padded.dim()} instead"
        )
        return (
            audio_padded[:, 0, :] != self.audio_pad_id
        )  # [B, audio_length] where True means attend to

    def _create_attention_mask(
        self, text_tokens: Tensor, audio_tokens: Tensor
    ) -> Tensor:
        assert text_tokens.dim() == 2, (
            f"Text tokens should have dimensions 2, but got {text_tokens.dim()} instead"
        )
        assert audio_tokens.dim() == 3, (
            f"Audio tokens should have dimensions 2, but got {audio_tokens.dim()} instead"
        )

        # Text: [B, text_t], Audio: [B, Q, audio_t]
        batch_size, text_len = text_tokens.shape
        audio_len = audio_tokens.shape[2]

        total_len = text_len + audio_len
        causal_mask = torch.zeros(
            (total_len, total_len), dtype=torch.bool, device=text_tokens.device
        )
        causal_mask[:text_len, :text_len] = True  # Text can attend to all text
        causal_mask[text_len:, :text_len] = True  # Audio can attend to all text

        causal_mask[text_len:, text_len:] = torch.tril(
            torch.ones(
                audio_len, audio_len, dtype=torch.bool, device=text_tokens.device
            )
        )

        assert causal_mask.shape == (total_len, total_len), (
            f"Expected {total_len, total_len} but got {causal_mask.shape} instead"
        )  # [total_t, total_t]

        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        assert causal_mask.shape == (
            batch_size,
            total_len,
            total_len,
        ), (
            f"Expected {(batch_size, total_len, total_len)} but got {causal_mask.shape} instead"
        )  # [B, total_t, total_t

        return causal_mask
