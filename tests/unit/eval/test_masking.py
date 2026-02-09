# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for EvalMasker reproducibility and behavior."""

import torch
import pytest

from mutable.eval.masking import EvalMasker


def _make_batch(B=2, L=200, seed=0):
    """Create a simple batch for masking tests."""
    torch.manual_seed(seed)
    input_ids = torch.randint(4, 29, (B, L))
    input_ids[:, 0] = 0   # BOS
    input_ids[:, -1] = 2  # EOS
    sep_pos = L // 2
    input_ids[:, sep_pos] = 29  # SEP

    attention_mask = torch.ones(B, L, dtype=torch.long)

    special_tokens_mask = torch.zeros(B, L, dtype=torch.bool)
    special_tokens_mask[:, 0] = True
    special_tokens_mask[:, sep_pos] = True
    special_tokens_mask[:, -1] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "special_tokens_mask": special_tokens_mask,
    }


class TestEvalMasker:
    def test_reproducibility_same_seed(self):
        batch = _make_batch()
        masker1 = EvalMasker(seed=42)
        masker2 = EvalMasker(seed=42)

        masked1, labels1 = masker1.apply_mask(batch)
        masked2, labels2 = masker2.apply_mask(batch)

        assert torch.equal(masked1, masked2)
        assert torch.equal(labels1, labels2)

    def test_different_seeds_differ(self):
        batch = _make_batch()
        masker1 = EvalMasker(seed=42)
        masker2 = EvalMasker(seed=123)

        _, labels1 = masker1.apply_mask(batch)
        _, labels2 = masker2.apply_mask(batch)

        assert not torch.equal(labels1, labels2)

    def test_mask_rate_approximate(self):
        batch = _make_batch(B=1, L=200)
        masker = EvalMasker(mask_rate=0.15, seed=42)
        _, mask_labels = masker.apply_mask(batch)

        # Count maskable positions (non-special, attended)
        special = batch["special_tokens_mask"][0]
        attended = batch["attention_mask"][0].bool()
        maskable = (~special & attended).sum().item()

        masked_count = mask_labels[0].sum().item()
        actual_rate = masked_count / maskable
        assert abs(actual_rate - 0.15) < 0.05, f"Mask rate {actual_rate:.3f} not close to 0.15"

    def test_special_tokens_not_masked(self):
        batch = _make_batch()
        masker = EvalMasker(seed=42)
        _, mask_labels = masker.apply_mask(batch)

        special = batch["special_tokens_mask"]
        # No special token positions should be masked
        assert (mask_labels & special).sum().item() == 0

    def test_returns_masked_ids_and_labels(self):
        batch = _make_batch()
        masker = EvalMasker(seed=42)
        masked_ids, mask_labels = masker.apply_mask(batch)

        assert masked_ids.dtype == torch.long
        assert mask_labels.dtype == torch.bool
        assert masked_ids.shape == batch["input_ids"].shape
        assert mask_labels.shape == batch["input_ids"].shape

    def test_mask_token_id(self):
        batch = _make_batch()
        masker = EvalMasker(seed=42, mask_token_id=31)
        masked_ids, mask_labels = masker.apply_mask(batch)

        # Masked positions should have mask_token_id=31
        assert (masked_ids[mask_labels] == 31).all()
