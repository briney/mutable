# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for DenoisingDataset and DenoisingCollator."""

import pytest
import torch

from mutable.datasets.denoising_dataset import (
    DenoisingDataset,
    DenoisingCollator,
    _parse_digit_mask,
    _parse_segment_mask,
)
from mutable.modules.noise import BartNoiseFunction
from mutable.tokenizer import MutableTokenizer


class TestParsers:
    def test_parse_digit_mask(self):
        result = _parse_digit_mask("000111")
        assert result == [0, 0, 0, 1, 1, 1]

    def test_parse_segment_mask(self):
        result = _parse_segment_mask("VVNDJ")
        assert result == [1, 1, 4, 2, 3]

    def test_parse_segment_mask_unknown_chars(self):
        result = _parse_segment_mask("VXND")
        assert result == [1, 0, 4, 2]


class TestDenoisingDataset:
    @pytest.fixture
    def tokenizer(self):
        return MutableTokenizer()

    @pytest.fixture
    def noise_fn(self):
        return BartNoiseFunction(
            mask_token_id=31, pad_token_id=1, vocab_size=32, mask_ratio=0.3,
        )

    @pytest.fixture
    def simple_dataset(self):
        class MockDS:
            column_names = ["heavy", "light"]
            def __getitem__(self, idx):
                return {"heavy": "EVQLVESGG", "light": "DIQMTQSPS"}
            def __len__(self):
                return 4
        return MockDS()

    def test_default_mode_returns_corrupted_and_labels(self, simple_dataset, tokenizer, noise_fn):
        ds = DenoisingDataset(simple_dataset, tokenizer, noise_fn=noise_fn)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        # labels should be clean, input_ids may be corrupted
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_noise_fn_none_raises_for_default_mode(self, simple_dataset, tokenizer):
        with pytest.raises(ValueError, match="noise_fn is required"):
            DenoisingDataset(simple_dataset, tokenizer, noise_fn=None)

    def test_weighted_mode_returns_clean_ids(self, simple_dataset, tokenizer):
        ds = DenoisingDataset(
            simple_dataset, tokenizer, use_weighted_masking=True,
        )
        item = ds[0]
        # In weighted mode, input_ids are clean
        assert torch.equal(item["input_ids"], item["labels"])

    def test_special_tokens_mask_correct(self, simple_dataset, tokenizer):
        ds = DenoisingDataset(
            simple_dataset, tokenizer, use_weighted_masking=True,
        )
        item = ds[0]
        stm = item["special_tokens_mask"]
        ids = item["input_ids"]
        # BOS, SEP, EOS should be True
        assert stm[0] == True  # BOS
        assert stm[-1] == True  # EOS
        # Find sep position
        sep_positions = (ids == 29).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            assert stm[sep_positions[0]] == True

    def test_cdr_mask_built_correctly(self, tokenizer):
        class MockDS:
            column_names = ["heavy", "light", "heavy_cdr", "light_cdr"]
            def __getitem__(self, idx):
                return {
                    "heavy": "EVQLVE",
                    "light": "DIQMTQ",
                    "heavy_cdr": "001100",
                    "light_cdr": "000110",
                }
            def __len__(self):
                return 1

        ds = DenoisingDataset(
            MockDS(), tokenizer, use_weighted_masking=True,
            heavy_cdr_col="heavy_cdr", light_cdr_col="light_cdr",
        )
        item = ds[0]
        assert "cdr_mask" in item
        cdr = item["cdr_mask"]
        # BOS=0, heavy CDR positions, SEP=0, light CDR positions, EOS=0
        assert cdr[0] == 0  # BOS
        assert cdr[-1] == 0  # EOS

    def test_build_mask_pads_short_masks(self, tokenizer):
        class MockDS:
            column_names = ["heavy", "light", "heavy_cdr"]
            def __getitem__(self, idx):
                return {"heavy": "EVQLVE", "light": "DIQMTQ", "heavy_cdr": "00"}
            def __len__(self):
                return 1

        ds = DenoisingDataset(
            MockDS(), tokenizer, use_weighted_masking=True,
            heavy_cdr_col="heavy_cdr",
        )
        item = ds[0]
        # Should not error; mask padded with zeros
        assert "cdr_mask" in item

    def test_max_length_truncation(self, simple_dataset, tokenizer, noise_fn):
        ds = DenoisingDataset(
            simple_dataset, tokenizer, noise_fn=noise_fn, max_length=10,
        )
        item = ds[0]
        assert item["input_ids"].shape[0] <= 10

    def test_len_returns_dataset_length(self, simple_dataset, tokenizer, noise_fn):
        ds = DenoisingDataset(simple_dataset, tokenizer, noise_fn=noise_fn)
        assert len(ds) == 4


class TestDenoisingCollator:
    def test_pads_to_max_length(self):
        collator = DenoisingCollator(pad_token_id=1)
        features = [
            {"input_ids": torch.tensor([0, 5, 6, 2]), "labels": torch.tensor([0, 5, 6, 2])},
            {"input_ids": torch.tensor([0, 7, 8, 9, 2]), "labels": torch.tensor([0, 7, 8, 9, 2])},
        ]
        batch = collator(features)
        assert batch["input_ids"].shape == (2, 5)

    def test_attention_mask_zeros_at_padding(self):
        collator = DenoisingCollator(pad_token_id=1)
        features = [
            {"input_ids": torch.tensor([0, 5, 2]), "labels": torch.tensor([0, 5, 2])},
            {"input_ids": torch.tensor([0, 7, 8, 2]), "labels": torch.tensor([0, 7, 8, 2])},
        ]
        batch = collator(features)
        # first sample padded by 1
        assert batch["attention_mask"][0, -1] == 0
        assert batch["attention_mask"][1, -1] == 1

    def test_labels_padded_with_neg100(self):
        collator = DenoisingCollator(pad_token_id=1)
        features = [
            {"input_ids": torch.tensor([0, 5, 2]), "labels": torch.tensor([0, 5, 2])},
            {"input_ids": torch.tensor([0, 7, 8, 2]), "labels": torch.tensor([0, 7, 8, 2])},
        ]
        batch = collator(features)
        assert batch["labels"][0, -1] == -100

    def test_weighted_mode_pads_special_tokens_mask(self):
        collator = DenoisingCollator(pad_token_id=1, use_weighted_masking=True)
        features = [
            {
                "input_ids": torch.tensor([0, 5, 2]),
                "labels": torch.tensor([0, 5, 2]),
                "special_tokens_mask": torch.tensor([True, False, True]),
            },
            {
                "input_ids": torch.tensor([0, 7, 8, 2]),
                "labels": torch.tensor([0, 7, 8, 2]),
                "special_tokens_mask": torch.tensor([True, False, False, True]),
            },
        ]
        batch = collator(features)
        # Padding in special_tokens_mask should be True (not maskable)
        assert batch["special_tokens_mask"][0, -1] == True

    def test_weighted_mode_pads_integer_masks(self):
        collator = DenoisingCollator(pad_token_id=1, use_weighted_masking=True)
        features = [
            {
                "input_ids": torch.tensor([0, 5, 2]),
                "labels": torch.tensor([0, 5, 2]),
                "special_tokens_mask": torch.tensor([True, False, True]),
                "cdr_mask": torch.tensor([0, 1, 0]),
            },
            {
                "input_ids": torch.tensor([0, 7, 8, 2]),
                "labels": torch.tensor([0, 7, 8, 2]),
                "special_tokens_mask": torch.tensor([True, False, False, True]),
                "cdr_mask": torch.tensor([0, 1, 2, 0]),
            },
        ]
        batch = collator(features)
        # Padding in integer masks should be 0
        assert batch["cdr_mask"][0, -1] == 0
