# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Optional

from transformers import Trainer

from ..masking import InformationWeightedMasker, UniformMasker

__all__ = ["DenoisingTrainer"]

# Keys that carry annotation masks (not model inputs)
_MASK_KEYS = frozenset({
    "special_tokens_mask",
    "cdr_mask",
    "nongermline_mask",
    "segment_mask",
})


class DenoisingTrainer(Trainer):
    """
    Trainer for Phase 1 denoising pre-training.

    When ``use_weighted_masking=True``, applies batch-level masking inside
    ``compute_loss`` before delegating to the model. Dispatches to
    ``InformationWeightedMasker`` when annotation masks are present in the
    batch, otherwise falls back to ``UniformMasker``.

    Parameters
    ----------
    masker : InformationWeightedMasker, optional
        Weighted masker for batches with annotation masks.
    uniform_masker : UniformMasker, optional
        Fallback masker for batches without annotation masks.
    use_weighted_masking : bool, default=False
        Whether batch-level masking is enabled.
    **kwargs
        All other arguments are forwarded to ``transformers.Trainer``.
    """

    def __init__(
        self,
        masker: Optional[InformationWeightedMasker] = None,
        uniform_masker: Optional[UniformMasker] = None,
        use_weighted_masking: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.masker = masker
        self.uniform_masker = uniform_masker
        self.use_weighted_masking = use_weighted_masking

    def _apply_masking(self, inputs: dict) -> dict:
        """Apply batch-level masking and strip annotation keys.

        If annotation masks (cdr_mask, nongermline_mask, segment_mask) are
        present, uses ``self.masker``; otherwise falls back to
        ``self.uniform_masker``.

        Returns a new dict with masked ``input_ids`` and annotation keys removed.
        """
        inputs = dict(inputs)  # shallow copy to avoid mutating caller's dict

        # Extract annotation masks
        special_tokens_mask = inputs.pop("special_tokens_mask", None)
        cdr_mask = inputs.pop("cdr_mask", None)
        nongermline_mask = inputs.pop("nongermline_mask", None)
        segment_mask = inputs.pop("segment_mask", None)

        has_annotations = cdr_mask is not None or nongermline_mask is not None or segment_mask is not None

        if has_annotations and self.masker is not None:
            masked_ids, _ = self.masker.apply_mask(
                token_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                cdr_mask=cdr_mask,
                nongermline_mask=nongermline_mask,
                segment_mask=segment_mask,
                special_tokens_mask=special_tokens_mask,
            )
        elif self.uniform_masker is not None:
            masked_ids, _ = self.uniform_masker.apply_mask(
                token_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                special_tokens_mask=special_tokens_mask,
            )
        else:
            return inputs

        inputs["input_ids"] = masked_ids
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.use_weighted_masking:
            inputs = self._apply_masking(inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
