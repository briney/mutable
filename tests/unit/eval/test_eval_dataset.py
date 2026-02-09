# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for EvalDenoisingDataset."""

import pytest
import torch
from datasets import Dataset as HFDataset

from mutable.eval.datasets.eval_dataset import EvalDenoisingDataset


@pytest.fixture
def hf_dataset():
    """Minimal HuggingFace dataset with heavy/light columns."""
    return HFDataset.from_dict({
        "heavy": ["EVQLVESGG", "QVQLVESGP"],
        "light": ["DIQMTQSPS", "EIVLTQSPA"],
    })


@pytest.fixture
def hf_dataset_with_annotations():
    """HuggingFace dataset with annotation columns."""
    return HFDataset.from_dict({
        "heavy": ["EVQLVESGG", "QVQLVESGP"],
        "light": ["DIQMTQSPS", "EIVLTQSPA"],
        "heavy_cdr": ["000110000", "001100000"],
        "light_cdr": ["000011000", "000001100"],
        "heavy_ng": ["000010000", "000100000"],
        "light_ng": ["000001000", "000000100"],
    })


class TestEvalDenoisingDataset:
    def test_output_keys(self, hf_dataset, tokenizer):
        ds = EvalDenoisingDataset(hf_dataset, tokenizer)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert "special_tokens_mask" in item
        assert "chain_ids" in item

    def test_labels_equal_input_ids(self, hf_dataset, tokenizer):
        """Clean eval: labels == input_ids (no masking applied)."""
        ds = EvalDenoisingDataset(hf_dataset, tokenizer)
        item = ds[0]
        assert torch.equal(item["input_ids"], item["labels"])

    def test_special_tokens_mask(self, hf_dataset, tokenizer):
        """BOS, SEP, EOS positions are True in special_tokens_mask."""
        ds = EvalDenoisingDataset(hf_dataset, tokenizer)
        item = ds[0]
        stm = item["special_tokens_mask"]
        ids = item["input_ids"]

        assert stm[0].item() is True  # BOS
        assert stm[-1].item() is True  # EOS

        # Find SEP position
        sep_positions = (ids == 29).nonzero(as_tuple=True)[0]
        assert len(sep_positions) == 1
        assert stm[sep_positions[0]].item() is True

    def test_chain_ids_from_sep(self, hf_dataset, tokenizer):
        """0 before sep, 1 after."""
        ds = EvalDenoisingDataset(hf_dataset, tokenizer)
        item = ds[0]
        chain_ids = item["chain_ids"]
        ids = item["input_ids"]

        sep_pos = (ids == 29).nonzero(as_tuple=True)[0][0].item()
        assert (chain_ids[:sep_pos + 1] == 0).all()
        assert (chain_ids[sep_pos + 1:] == 1).all()

    def test_annotation_masks_present(self, hf_dataset_with_annotations, tokenizer):
        ds = EvalDenoisingDataset(
            hf_dataset_with_annotations,
            tokenizer,
            heavy_cdr_col="heavy_cdr",
            light_cdr_col="light_cdr",
            heavy_nongermline_col="heavy_ng",
            light_nongermline_col="light_ng",
        )
        item = ds[0]
        assert "cdr_mask" in item
        assert "nongermline_mask" in item

    def test_annotation_masks_absent(self, hf_dataset, tokenizer):
        ds = EvalDenoisingDataset(hf_dataset, tokenizer)
        item = ds[0]
        assert "cdr_mask" not in item
        assert "nongermline_mask" not in item
