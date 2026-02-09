# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Integration tests for dataset → model pipelines."""

import torch
from torch.utils.data import DataLoader

from mutable.datasets.denoising_dataset import DenoisingDataset, DenoisingCollator
from mutable.datasets.flow_dataset import FlowMatchingDataset
from mutable.modules.noise import BartNoiseFunction
from mutable.tokenizer import MutableTokenizer


class _MockHFDataset:
    """Simple mock HF dataset."""
    def __init__(self, data, columns=None):
        self.data = data
        self.column_names = columns or list(data[0].keys())

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TestDenoisingDatasetToModel:
    def test_dataset_to_model_forward(self, small_model):
        tok = MutableTokenizer()
        noise = BartNoiseFunction(mask_token_id=31, pad_token_id=1, vocab_size=32, mask_ratio=0.3)
        data = [
            {"heavy": "EVQLVESGG", "light": "DIQMTQSPS"},
            {"heavy": "QVQLVQSGA", "light": "EIVLTQSPG"},
        ]
        ds = DenoisingDataset(_MockHFDataset(data), tok, noise_fn=noise, max_length=64)
        collator = DenoisingCollator(pad_token_id=1)
        batch = collator([ds[0], ds[1]])
        out = small_model(**batch, return_dict=True)
        assert out.loss is not None

    def test_weighted_masking_pipeline(self, small_model):
        tok = MutableTokenizer()
        data = [
            {"heavy": "EVQLVESGG", "light": "DIQMTQSPS"},
            {"heavy": "QVQLVQSGA", "light": "EIVLTQSPG"},
        ]
        ds = DenoisingDataset(
            _MockHFDataset(data), tok,
            use_weighted_masking=True, max_length=64,
        )
        collator = DenoisingCollator(pad_token_id=1, use_weighted_masking=True)
        batch = collator([ds[0], ds[1]])
        # In weighted mode, we need to apply masking before model forward
        from mutable.masking import UniformMasker
        masker = UniformMasker(mask_rate=0.15)
        masked_ids, _ = masker.apply_mask(
            batch["input_ids"], batch["attention_mask"],
            special_tokens_mask=batch.get("special_tokens_mask"),
        )
        batch["input_ids"] = masked_ids
        # Remove non-model keys
        for key in ["special_tokens_mask"]:
            batch.pop(key, None)
        out = small_model(**batch, return_dict=True)
        assert out.loss is not None

    def test_dataloader_batching(self, small_model):
        tok = MutableTokenizer()
        noise = BartNoiseFunction(mask_token_id=31, pad_token_id=1, vocab_size=32, mask_ratio=0.3)
        data = [{"heavy": "EVQLVESGG", "light": "DIQMTQSPS"}] * 4
        ds = DenoisingDataset(_MockHFDataset(data), tok, noise_fn=noise, max_length=64)
        collator = DenoisingCollator(pad_token_id=1)
        loader = DataLoader(ds, batch_size=2, collate_fn=collator)
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 2


class TestFlowDatasetToModel:
    def test_dataset_to_model_forward(self, small_flow_model):
        tok = MutableTokenizer()
        data = [
            {
                "germline_heavy": "EVQLVESGG",
                "germline_light": "DIQMTQSPS",
                "mutated_heavy": "EVQLVQSGG",
                "mutated_light": "DIQMTQSPS",
            },
        ]

        class MockDS:
            column_names = list(data[0].keys())
            def __getitem__(self, idx):
                return data[idx]
            def __len__(self):
                return len(data)

        ds = FlowMatchingDataset(MockDS(), tok, max_length=64)
        item = ds[0]
        # Create a batch of size 1
        batch = {k: v.unsqueeze(0) for k, v in item.items()}
        small_flow_model.train()
        out = small_flow_model(
            germline_input_ids=batch["germline_input_ids"],
            germline_attention_mask=batch["germline_attention_mask"],
            mutated_input_ids=batch["mutated_input_ids"],
            mutated_attention_mask=batch["mutated_attention_mask"],
            mu=batch["mu"],
            return_dict=True,
        )
        small_flow_model.eval()
        assert out.loss is not None

    def test_dataloader_batching(self):
        tok = MutableTokenizer()
        data = [
            {
                "germline_heavy": "EVQLVESGG",
                "germline_light": "DIQMTQSPS",
                "mutated_heavy": "EVQLVQSGG",
                "mutated_light": "DIQMTQSPS",
            },
        ] * 4

        class MockDS:
            column_names = list(data[0].keys())
            def __init__(self):
                self.data = data
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self):
                return len(self.data)

        ds = FlowMatchingDataset(MockDS(), tok, max_length=64)
        loader = DataLoader(ds, batch_size=2)
        batch = next(iter(loader))
        assert batch["germline_input_ids"].shape[0] == 2
