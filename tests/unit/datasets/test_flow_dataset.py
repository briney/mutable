# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for FlowMatchingDataset and FlowMatchingCollator."""

import logging

import pytest
import torch

from mutable.datasets.flow_dataset import FlowMatchingCollator, FlowMatchingDataset
from mutable.tokenizer import MutableTokenizer


class TestFlowMatchingDataset:
    @staticmethod
    def _make_mock_dataset(data=None, columns=None):
        class MockDS:
            def __init__(self, data, columns):
                self.data = data
                self.column_names = columns

            def __getitem__(self, idx):
                return self.data[idx]

            def __len__(self):
                return len(self.data)

        if data is None:
            data = [
                {
                    "germline_heavy": "EVQLVESGG",
                    "germline_light": "DIQMTQSPS",
                    "mutated_heavy": "EVQLVQSGG",
                    "mutated_light": "DIQMTQSPS",
                },
                {
                    "germline_heavy": "QVQLVQSGA",
                    "germline_light": "EIVLTQSPG",
                    "mutated_heavy": "QVQLAQSGA",
                    "mutated_light": "EIVLTASPG",
                },
            ]
        if columns is None:
            columns = list(data[0].keys())
        return MockDS(data, columns)

    def test_getitem_returns_expected_keys(self):
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=64)
        item = ds[0]
        assert "germline_input_ids" in item
        assert "germline_attention_mask" in item
        assert "mutated_input_ids" in item
        assert "mutated_attention_mask" in item
        assert "mu" in item

    def test_compute_mu_identical_is_zero(self):
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=64)
        mu = ds._compute_mu("ABCDE", "ABCDE")
        assert mu == 0.0

    def test_compute_mu_all_different_is_one(self):
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=64)
        mu = ds._compute_mu("ABCDE", "FGHIJ")
        assert mu == 1.0

    def test_compute_mu_partial(self):
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=64)
        mu = ds._compute_mu("ABCDE", "ABXYZ")
        # 3 out of 5 differ
        assert abs(mu - 3.0 / 5.0) < 1e-6

    def test_length_difference_counted(self):
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=64)
        mu = ds._compute_mu("ABC", "ABCDE")
        # All 3 match, but 2 extra → 2/5
        assert abs(mu - 2.0 / 5.0) < 1e-6

    def test_precomputed_mu_col(self):
        data = [
            {
                "germline_heavy": "EVQLVESGG",
                "germline_light": "DIQMTQSPS",
                "mutated_heavy": "EVQLVQSGG",
                "mutated_light": "DIQMTQSPS",
                "mu": 0.42,
            },
        ]

        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(data), tok, max_length=64, mu_col="mu")
        item = ds[0]
        assert abs(item["mu"].item() - 0.42) < 1e-6

    def test_dataset_length(self):
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=64)
        assert len(ds) == 2

    def test_no_padding_in_getitem(self):
        """Dataset items should NOT be padded to max_length (collator handles that)."""
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=512)
        item = ds[0]
        # 9 chars + 9 chars + BOS + SEP + EOS = 21 tokens — much less than 512
        assert item["germline_input_ids"].shape[0] < 512

    # --- P2: Validation tests ---

    def test_missing_required_column_raises(self):
        """ValueError when a required sequence column is missing."""
        data = [{"germline_heavy": "EVQLVESGG", "germline_light": "DIQMTQSPS"}]
        mock_ds = self._make_mock_dataset(
            data, columns=["germline_heavy", "germline_light"]
        )
        tok = MutableTokenizer()
        with pytest.raises(ValueError, match="Required column"):
            FlowMatchingDataset(mock_ds, tok, max_length=64)

    def test_missing_mu_col_warns(self, caplog):
        """Warning (not error) when mu_col is specified but not in dataset."""
        tok = MutableTokenizer()
        mock_ds = self._make_mock_dataset()
        with caplog.at_level(logging.WARNING):
            ds = FlowMatchingDataset(mock_ds, tok, max_length=64, mu_col="nonexistent_mu")
        assert "nonexistent_mu" in caplog.text
        assert ds.mu_col is None  # fell back to None
        # Dataset should still be usable
        item = ds[0]
        assert "mu" in item


class TestFlowMatchingCollator:
    def _make_features(self, germline_lens, mutated_lens, pad_token_id=1):
        """Create feature dicts with specified sequence lengths."""
        features = []
        for g_len, m_len in zip(germline_lens, mutated_lens):
            germ_ids = torch.randint(4, 29, (g_len,))
            germ_ids[0] = 0
            germ_ids[-1] = 2
            mut_ids = torch.randint(4, 29, (m_len,))
            mut_ids[0] = 0
            mut_ids[-1] = 2
            features.append({
                "germline_input_ids": germ_ids,
                "germline_attention_mask": torch.ones(g_len, dtype=torch.long),
                "mutated_input_ids": mut_ids,
                "mutated_attention_mask": torch.ones(m_len, dtype=torch.long),
                "mu": torch.tensor(0.1),
            })
        return features

    def test_collator_pads_to_max_length(self):
        """Output shapes should match the max length in the batch."""
        features = self._make_features(
            germline_lens=[10, 15], mutated_lens=[12, 8]
        )
        collator = FlowMatchingCollator(pad_token_id=1)
        batch = collator(features)
        assert batch["germline_input_ids"].shape == (2, 15)
        assert batch["mutated_input_ids"].shape == (2, 12)
        assert batch["mu"].shape == (2,)

    def test_collator_attention_mask_zeros_at_padding(self):
        """Padded positions should have attention_mask=0."""
        features = self._make_features(
            germline_lens=[10, 15], mutated_lens=[12, 8]
        )
        collator = FlowMatchingCollator(pad_token_id=1)
        batch = collator(features)
        # First sample germline padded from 10 to 15
        assert batch["germline_attention_mask"][0, 9] == 1  # last real token
        assert batch["germline_attention_mask"][0, 10] == 0  # first pad
        assert batch["germline_attention_mask"][0, 14] == 0  # last pad
        # Second sample mutated padded from 8 to 12
        assert batch["mutated_attention_mask"][1, 7] == 1
        assert batch["mutated_attention_mask"][1, 8] == 0

    def test_collator_pad_token_id(self):
        """Padded input_ids positions should use the configured pad_token_id."""
        pad_id = 1
        features = self._make_features(
            germline_lens=[10, 15], mutated_lens=[12, 8]
        )
        collator = FlowMatchingCollator(pad_token_id=pad_id)
        batch = collator(features)
        # Shorter germline sample padded with pad_token_id
        assert batch["germline_input_ids"][0, 10].item() == pad_id
        assert batch["germline_input_ids"][0, 14].item() == pad_id
        # Shorter mutated sample padded with pad_token_id
        assert batch["mutated_input_ids"][1, 8].item() == pad_id
        assert batch["mutated_input_ids"][1, 11].item() == pad_id

    def test_collator_equal_lengths_no_padding(self):
        """When all samples are the same length, no padding should be added."""
        features = self._make_features(
            germline_lens=[10, 10], mutated_lens=[10, 10]
        )
        collator = FlowMatchingCollator(pad_token_id=1)
        batch = collator(features)
        assert batch["germline_input_ids"].shape == (2, 10)
        # All attention mask values should be 1
        assert batch["germline_attention_mask"].sum().item() == 20
