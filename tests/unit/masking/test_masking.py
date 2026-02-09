# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for masking strategies."""

import pytest
import torch

from mutable.masking.masking import InformationWeightedMasker, UniformMasker


class TestInformationWeightedMasker:
    def test_mask_rate_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mask_rate must be in"):
            InformationWeightedMasker(mask_rate=0.0)
        with pytest.raises(ValueError, match="mask_rate must be in"):
            InformationWeightedMasker(mask_rate=1.0)

    def test_invalid_selection_method_raises(self):
        with pytest.raises(ValueError, match="selection_method"):
            InformationWeightedMasker(selection_method="invalid")

    def test_uniform_weights_without_annotations(self):
        masker = InformationWeightedMasker(mask_rate=0.3)
        maskable = torch.tensor([[True, True, True, False, False]])
        weights = masker.compute_weights(maskable)
        # 3 maskable positions, each should get 1/3
        expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0]])
        torch.testing.assert_close(weights, expected, atol=1e-6, rtol=1e-6)

    def test_cdr_bonus_increases_weights(self):
        masker = InformationWeightedMasker(mask_rate=0.3, cdr_weight_multiplier=1.0)
        maskable = torch.tensor([[True, True, True, True]])
        cdr_mask = torch.tensor([[0, 1, 0, 0]])  # position 1 is CDR
        weights = masker.compute_weights(maskable, cdr_mask=cdr_mask)
        # CDR position should have higher weight
        assert weights[0, 1] > weights[0, 0]

    def test_nongermline_bonus(self):
        masker = InformationWeightedMasker(mask_rate=0.3, nongermline_weight_multiplier=1.0)
        maskable = torch.tensor([[True, True, True, True]])
        ng = torch.tensor([[0, 0, 1, 0]])
        weights = masker.compute_weights(maskable, nongermline_mask=ng)
        assert weights[0, 2] > weights[0, 0]

    def test_combined_bonuses(self):
        masker = InformationWeightedMasker(
            mask_rate=0.3, cdr_weight_multiplier=1.0, nongermline_weight_multiplier=1.0,
        )
        maskable = torch.tensor([[True, True, True, True]])
        cdr = torch.tensor([[0, 1, 1, 0]])
        ng = torch.tensor([[0, 0, 1, 0]])
        weights = masker.compute_weights(maskable, cdr_mask=cdr, nongermline_mask=ng)
        # pos2 has both CDR and nongermline → highest weight
        assert weights[0, 2] > weights[0, 1]
        assert weights[0, 1] > weights[0, 0]

    def test_non_maskable_zeroed(self):
        masker = InformationWeightedMasker(mask_rate=0.3)
        maskable = torch.tensor([[True, True, False, False]])
        weights = masker.compute_weights(maskable)
        assert weights[0, 2] == 0.0
        assert weights[0, 3] == 0.0

    def test_normalized_to_sum_one(self):
        masker = InformationWeightedMasker(mask_rate=0.3)
        maskable = torch.tensor([[True, True, True, True, True]])
        weights = masker.compute_weights(maskable)
        torch.testing.assert_close(weights.sum(dim=-1), torch.tensor([1.0]), atol=1e-6, rtol=1e-6)

    def test_ranked_mode_deterministic(self):
        masker = InformationWeightedMasker(mask_rate=0.3, selection_method="ranked")
        ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
        attn = torch.ones(1, 10, dtype=torch.long)
        cdr = torch.zeros(1, 10, dtype=torch.long)
        cdr[0, 3:6] = 1
        g1 = torch.Generator(); g1.manual_seed(42)
        _, mask1 = masker.apply_mask(ids, attn, cdr_mask=cdr, generator=g1)
        g2 = torch.Generator(); g2.manual_seed(99)
        _, mask2 = masker.apply_mask(ids, attn, cdr_mask=cdr, generator=g2)
        # ranked should be nearly deterministic (noise is 1e-6)
        torch.testing.assert_close(mask1, mask2)

    def test_sampled_mode_varies(self):
        masker = InformationWeightedMasker(mask_rate=0.3, selection_method="sampled")
        ids = torch.randint(4, 29, (1, 50))
        attn = torch.ones(1, 50, dtype=torch.long)
        g1 = torch.Generator(); g1.manual_seed(42)
        _, mask1 = masker.apply_mask(ids, attn, generator=g1)
        g2 = torch.Generator(); g2.manual_seed(99)
        _, mask2 = masker.apply_mask(ids, attn, generator=g2)
        # Different seeds should (almost certainly) give different masks
        assert not torch.equal(mask1, mask2)

    def test_special_tokens_never_masked(self):
        masker = InformationWeightedMasker(mask_rate=0.5)
        ids = torch.tensor([[0, 5, 6, 7, 29, 8, 9, 2]])
        attn = torch.ones(1, 8, dtype=torch.long)
        stm = torch.tensor([[True, False, False, False, True, False, False, True]])
        masked_ids, mask_labels = masker.apply_mask(ids, attn, special_tokens_mask=stm)
        # BOS(0), SEP(4), EOS(7) should never be masked
        assert not mask_labels[0, 0]
        assert not mask_labels[0, 4]
        assert not mask_labels[0, 7]

    def test_correct_count(self):
        masker = InformationWeightedMasker(mask_rate=0.3)
        ids = torch.randint(4, 29, (1, 20))
        attn = torch.ones(1, 20, dtype=torch.long)
        stm = torch.zeros(1, 20, dtype=torch.bool)
        stm[0, 0] = True; stm[0, -1] = True
        _, mask_labels = masker.apply_mask(ids, attn, special_tokens_mask=stm)
        maskable_count = 18  # 20 - 2 special
        expected = round(maskable_count * 0.3)
        assert mask_labels.sum().item() == expected

    def test_correct_dtypes(self):
        masker = InformationWeightedMasker(mask_rate=0.3)
        ids = torch.randint(4, 29, (2, 10))
        attn = torch.ones(2, 10, dtype=torch.long)
        masked_ids, mask_labels = masker.apply_mask(ids, attn)
        assert masked_ids.dtype == torch.long
        assert mask_labels.dtype == torch.bool


class TestUniformMasker:
    def test_shape_and_dtype(self):
        masker = UniformMasker(mask_rate=0.3)
        ids = torch.randint(4, 29, (2, 10))
        attn = torch.ones(2, 10, dtype=torch.long)
        masked_ids, mask_labels = masker.apply_mask(ids, attn)
        assert masked_ids.shape == ids.shape
        assert mask_labels.dtype == torch.bool

    def test_approximately_correct_fraction(self):
        masker = UniformMasker(mask_rate=0.3)
        torch.manual_seed(42)
        ids = torch.randint(4, 29, (10, 100))
        attn = torch.ones(10, 100, dtype=torch.long)
        _, mask_labels = masker.apply_mask(ids, attn)
        fraction = mask_labels.float().mean().item()
        assert 0.2 < fraction < 0.4

    def test_special_tokens_excluded(self):
        masker = UniformMasker(mask_rate=0.5)
        ids = torch.tensor([[0, 5, 6, 7, 29, 8, 9, 2]])
        attn = torch.ones(1, 8, dtype=torch.long)
        stm = torch.tensor([[True, False, False, False, True, False, False, True]])
        _, mask_labels = masker.apply_mask(ids, attn, special_tokens_mask=stm)
        assert not mask_labels[0, 0]
        assert not mask_labels[0, 4]
        assert not mask_labels[0, 7]

    def test_reproducible_with_same_seed(self):
        masker = UniformMasker(mask_rate=0.3)
        ids = torch.randint(4, 29, (2, 20))
        attn = torch.ones(2, 20, dtype=torch.long)
        g1 = torch.Generator(); g1.manual_seed(42)
        _, mask1 = masker.apply_mask(ids, attn, generator=g1)
        g2 = torch.Generator(); g2.manual_seed(42)
        _, mask2 = masker.apply_mask(ids, attn, generator=g2)
        torch.testing.assert_close(mask1, mask2)
