# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from transformers import Trainer

from ..masking import InformationWeightedMasker, UniformMasker

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ..eval.evaluator import Evaluator

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

    When ``evaluator`` is provided, overrides the default ``evaluate()``
    method to use the custom multi-dataset evaluation harness.

    Parameters
    ----------
    masker : InformationWeightedMasker, optional
        Weighted masker for batches with annotation masks.
    uniform_masker : UniformMasker, optional
        Fallback masker for batches without annotation masks.
    use_weighted_masking : bool, default=False
        Whether batch-level masking is enabled.
    evaluator : Evaluator, optional
        Custom evaluator for multi-dataset, multi-metric evaluation.
    eval_loaders : dict[str, DataLoader], optional
        Named eval DataLoaders for the evaluator.
    **kwargs
        All other arguments are forwarded to ``transformers.Trainer``.
    """

    def __init__(
        self,
        masker: Optional[InformationWeightedMasker] = None,
        uniform_masker: Optional[UniformMasker] = None,
        use_weighted_masking: bool = False,
        evaluator: Optional[Evaluator] = None,
        eval_loaders: Optional[dict[str, DataLoader]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.masker = masker
        self.uniform_masker = uniform_masker
        self.use_weighted_masking = use_weighted_masking
        self.evaluator = evaluator
        self.eval_loaders = eval_loaders or {}

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

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Evaluate using the custom evaluator if available, else fall back to default.

        When ``self.evaluator`` is set, runs multi-dataset evaluation and
        flattens results into ``eval/{dataset_name}/{metric}`` format for
        logging.
        """
        if self.evaluator is None or not self.eval_loaders:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        all_results = self.evaluator.evaluate_all(self.eval_loaders)

        # Flatten into {eval/{name}/{metric}: value}
        flat: dict[str, float] = {}
        for eval_name, metrics in all_results.items():
            for k, v in metrics.items():
                flat[f"eval/{eval_name}/{k}"] = v

        self.log(flat)
        return flat
