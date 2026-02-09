# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Eval masking wrapper for controlled, reproducible eval masking."""

from __future__ import annotations

import torch
from torch import Tensor

from ..masking import UniformMasker

__all__ = ["EvalMasker"]


class EvalMasker:
    """Reproducible uniform masking for evaluation.

    Always uses uniform masking (not BART noise or weighted masking) so that
    evaluation metrics are comparable across runs and configurations.

    Parameters
    ----------
    mask_rate : float
        Fraction of maskable tokens to mask.
    mask_token_id : int
        Token ID used for masking.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mask_rate: float = 0.15,
        mask_token_id: int = 31,
        seed: int = 42,
    ) -> None:
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id
        self.seed = seed
        self._masker = UniformMasker(
            mask_rate=mask_rate,
            mask_token_id=mask_token_id,
        )

    def get_generator(self, device: torch.device) -> torch.Generator:
        """Create a seeded RNG on the given device.

        Parameters
        ----------
        device
            Device for the generator.

        Returns
        -------
        torch.Generator
            Seeded generator.
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return gen

    def apply_mask(
        self,
        batch: dict[str, Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply uniform masking to a batch.

        Parameters
        ----------
        batch
            Batch dict containing at minimum ``input_ids``, ``attention_mask``,
            and optionally ``special_tokens_mask``.
        generator
            Optional pre-seeded RNG. If None, creates one from ``self.seed``.

        Returns
        -------
        masked_ids : Tensor
            Token IDs with masked positions replaced.
        mask_labels : Tensor
            Boolean mask indicating which positions were masked.
        """
        if generator is None:
            generator = self.get_generator(batch["input_ids"].device)

        return self._masker.apply_mask(
            token_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            special_tokens_mask=batch.get("special_tokens_mask"),
            generator=generator,
        )
