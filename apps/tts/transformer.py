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

    n_heads: int = 8


class TTSTransformer(BaseTransformer):
    def __init__(self, args: TTSTransformerArgs):
        super().__init__(args)

        assert args.text_vocab_size > 0, "Please specify a valid text voab size"
        assert args.audio_vocab_size > 0, "Please specify a valid audio voab size"

        self.text_vocab_size = args.text_vocab_size
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

        mask = self.transform_mask(attn_impl, mask)

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
                # Flatten to [B*T_audio, vocab_size] vs [B*T_audio]
                losses.append(
                    cross_entropy(
                        logits[:, i].contiguous().view(-1, self.audio_vocab_size),
                        target[:, i].contiguous().view(-1),
                        ignore_index=self.audio_pad_id,
                    )
                )

            # Weight losses by quantizer level
            loss_weights = torch.linspace(
                self.quantizer_max_weight,
                self.quantizer_min_weight,
                self.num_quantizers,
                device=logits.device,
            )
            total_loss = sum(w * loss for w, loss in zip(loss_weights, losses))
            return total_loss, logits

        return logits

    def reset_parameters(self):
        super().reset_parameters()
        init_std = self.dim**-0.5

        # Text embeddings
        nn.init.trunc_normal_(
            self.text_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        # Audio embeddings
        for emb in self.audio_embeddings:
            nn.init.trunc_normal_(
                emb.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std
            )

        # Output layer
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=(self.dim * self.num_quantizers) ** -0.5,
            a=-3 * (self.dim * self.num_quantizers) ** -0.5,
            b=3 * (self.dim * self.num_quantizers) ** -0.5,
        )

    def transform_mask(
        self,
        attn_impl: str,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
    ):
        if isinstance(mask, torch.Tensor) and attn_impl == "sdpa":
            # B, S -> B, 1, 1, S. Necessary for broadcasting in sdpa's F.scaled_dot_product_attention
            mask = mask.unsqueeze(1).unsqueeze(1)

        return mask
