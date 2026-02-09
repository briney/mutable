# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for noise/corruption functions."""

import random

import pytest

from mutable.modules.noise import (
    BartNoiseFunction,
    token_masking,
    token_deletion,
    text_infilling,
    sentence_permutation,
)


class TestTokenMasking:
    def test_approximately_correct_ratio(self):
        random.seed(42)
        tokens = list(range(4, 30))  # 26 tokens
        result = token_masking(tokens, mask_token_id=31, ratio=0.3)
        assert len(result) == len(tokens)
        masked = sum(1 for t in result if t == 31)
        expected = max(1, int(len(tokens) * 0.3))
        assert masked == expected

    def test_mask_token_id_used(self):
        random.seed(42)
        tokens = list(range(4, 14))
        result = token_masking(tokens, mask_token_id=31, ratio=0.5)
        assert 31 in result

    def test_random_token_ratio(self):
        random.seed(42)
        tokens = list(range(4, 104))  # 100 tokens for statistical reliability
        result = token_masking(
            tokens, mask_token_id=31, ratio=0.5,
            random_token_ratio=1.0, vocab_size=32,
        )
        # All corrupted tokens should be random (not mask) since ratio=1.0
        # Some should be in [4, 30] range
        changed = [r for r, o in zip(result, tokens) if r != o]
        assert len(changed) > 0

    def test_preserves_length(self):
        tokens = list(range(4, 20))
        result = token_masking(tokens, mask_token_id=31, ratio=0.3)
        assert len(result) == len(tokens)


class TestTokenDeletion:
    def test_output_shorter(self):
        random.seed(42)
        tokens = list(range(4, 30))
        result = token_deletion(tokens, ratio=0.3)
        assert len(result) < len(tokens)

    def test_at_least_one_kept(self):
        random.seed(42)
        tokens = [5, 6, 7]
        result = token_deletion(tokens, ratio=0.99)
        assert len(result) >= 1

    def test_ratio_zero_preserves_all(self):
        tokens = list(range(4, 20))
        result = token_deletion(tokens, ratio=0.0)
        assert result == tokens


class TestTextInfilling:
    def test_output_shorter(self):
        random.seed(42)
        tokens = list(range(4, 34))
        result = text_infilling(tokens, mask_token_id=31, ratio=0.3)
        assert len(result) < len(tokens)

    def test_mask_token_present(self):
        random.seed(42)
        tokens = list(range(4, 34))
        result = text_infilling(tokens, mask_token_id=31, ratio=0.3)
        assert 31 in result

    def test_coverage_approximate(self):
        random.seed(42)
        tokens = list(range(4, 104))
        result = text_infilling(tokens, mask_token_id=31, ratio=0.3)
        # result should be shorter due to span collapse
        assert len(result) < len(tokens)


class TestSentencePermutation:
    def test_single_sentence_unchanged(self):
        tokens = [5, 6, 7, 8]
        result = sentence_permutation(tokens, sep_token_id=29)
        assert result == tokens

    def test_sep_positions_preserved(self):
        # heavy [sep] light
        tokens = [5, 6, 7, 29, 8, 9, 10]
        sep_positions_orig = [i for i, t in enumerate(tokens) if t == 29]
        random.seed(42)
        result = sentence_permutation(tokens, sep_token_id=29)
        sep_positions_new = [i for i, t in enumerate(result) if t == 29]
        assert sep_positions_new == sep_positions_orig

    def test_all_tokens_preserved(self):
        tokens = [5, 6, 7, 29, 8, 9, 10]
        random.seed(42)
        result = sentence_permutation(tokens, sep_token_id=29)
        assert sorted(result) == sorted(tokens)


class TestBartNoiseFunction:
    def test_mask_only_mode(self):
        random.seed(42)
        noise = BartNoiseFunction(
            mask_token_id=31, pad_token_id=1, vocab_size=32,
            mask_ratio=0.3, delete_ratio=0.0, infill_ratio=0.0,
        )
        tokens = list(range(4, 20))
        result = noise(tokens)
        assert len(result) == len(tokens)
        assert 31 in result

    def test_delete_only_mode(self):
        random.seed(42)
        noise = BartNoiseFunction(
            mask_token_id=31, pad_token_id=1, vocab_size=32,
            mask_ratio=0.0, delete_ratio=0.3, infill_ratio=0.0,
        )
        tokens = list(range(4, 30))
        result = noise(tokens)
        assert len(result) < len(tokens)

    def test_infill_only_mode(self):
        random.seed(42)
        noise = BartNoiseFunction(
            mask_token_id=31, pad_token_id=1, vocab_size=32,
            mask_ratio=0.0, delete_ratio=0.0, infill_ratio=0.3,
        )
        tokens = list(range(4, 34))
        result = noise(tokens)
        assert 31 in result
        assert len(result) <= len(tokens)

    def test_combined_operations(self):
        random.seed(42)
        noise = BartNoiseFunction(
            mask_token_id=31, pad_token_id=1, vocab_size=32,
            mask_ratio=0.2, delete_ratio=0.1, infill_ratio=0.0,
        )
        tokens = list(range(4, 30))
        result = noise(tokens)
        assert isinstance(result, list)
        assert len(result) > 0
