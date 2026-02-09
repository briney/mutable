# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for FlowMatchingDataset."""

import torch

from mutable.datasets.flow_dataset import FlowMatchingDataset
from mutable.tokenizer import MutableTokenizer


class TestFlowMatchingDataset:
    @staticmethod
    def _make_mock_dataset(data=None):
        class MockDS:
            column_names = [
                "germline_heavy", "germline_light",
                "mutated_heavy", "mutated_light",
            ]
            def __init__(self, data):
                self.data = data
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
        return MockDS(data)

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

        class MockDS:
            column_names = ["germline_heavy", "germline_light", "mutated_heavy", "mutated_light", "mu"]
            def __init__(self, data):
                self.data = data
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self):
                return len(self.data)

        tok = MutableTokenizer()
        ds = FlowMatchingDataset(MockDS(data), tok, max_length=64, mu_col="mu")
        item = ds[0]
        assert abs(item["mu"].item() - 0.42) < 1e-6

    def test_dataset_length(self):
        tok = MutableTokenizer()
        ds = FlowMatchingDataset(self._make_mock_dataset(), tok, max_length=64)
        assert len(ds) == 2
