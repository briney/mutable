# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Integration tests for Evaluator end-to-end evaluation."""

from __future__ import annotations

import math

import torch
import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from mutable.eval.evaluator import Evaluator


def _make_eval_loader(B=4, L=32, sep_pos=16, num_samples=8):
    """Create a synthetic eval DataLoader."""
    torch.manual_seed(42)

    all_items = []
    for _ in range(num_samples):
        ids = torch.randint(4, 29, (L,))
        ids[0] = 0     # BOS
        ids[sep_pos] = 29  # SEP
        ids[-1] = 2    # EOS

        attn = torch.ones(L, dtype=torch.long)
        labels = ids.clone()

        stm = torch.zeros(L, dtype=torch.bool)
        stm[0] = True
        stm[sep_pos] = True
        stm[-1] = True

        chain_ids = torch.zeros(L, dtype=torch.long)
        chain_ids[sep_pos + 1:] = 1

        all_items.append({
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attn,
            "special_tokens_mask": stm,
            "chain_ids": chain_ids,
        })

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items, has_coords=False):
            self.items = items
            self.has_coords = has_coords

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    ds = ListDataset(all_items)

    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    return DataLoader(ds, batch_size=B, collate_fn=collate)


class TestEvaluator:
    @pytest.fixture
    def evaluator_cfg(self):
        return OmegaConf.create({
            "model": {"num_encoder_layers": 2},
            "eval": {
                "masking": {"mask_rate": 0.15, "seed": 42},
                "metrics": {
                    "loss": {"enabled": True},
                    "masked_accuracy": {"enabled": True},
                    "perplexity": {"enabled": True},
                    "p_at_l": {"enabled": True, "num_layers": 1, "contact_mode": "all"},
                },
                "regions": {"enabled": False},
            },
        })

    def test_evaluate_returns_classification_metrics(self, evaluator_cfg, small_model, tokenizer):
        evaluator = Evaluator(evaluator_cfg, small_model, tokenizer)
        loader = _make_eval_loader()
        result = evaluator.evaluate(loader, "test")

        assert "loss" in result
        assert "mask_acc" in result
        assert "ppl" in result
        assert isinstance(result["loss"], float)
        assert math.isfinite(result["loss"])

    def test_evaluate_all_multiple_datasets(self, evaluator_cfg, small_model, tokenizer):
        evaluator = Evaluator(evaluator_cfg, small_model, tokenizer)
        loader1 = _make_eval_loader(num_samples=4)
        loader2 = _make_eval_loader(num_samples=4)

        results = evaluator.evaluate_all({"val": loader1, "test": loader2})
        assert "val" in results
        assert "test" in results
        assert "loss" in results["val"]
        assert "loss" in results["test"]

    def test_evaluate_with_regions(self, small_model, tokenizer):
        cfg = OmegaConf.create({
            "model": {"num_encoder_layers": 2},
            "eval": {
                "masking": {"mask_rate": 0.15, "seed": 42},
                "metrics": {
                    "loss": {"enabled": True},
                    "masked_accuracy": {"enabled": True},
                    "perplexity": {"enabled": True},
                },
                "regions": {
                    "enabled": True,
                    "hcdr3": True,
                    "all_cdr": True,
                },
            },
        })

        # Build loader with cdr_mask
        torch.manual_seed(42)
        L, sep_pos = 32, 16
        items = []
        for _ in range(4):
            ids = torch.randint(4, 29, (L,))
            ids[0] = 0
            ids[sep_pos] = 29
            ids[-1] = 2

            cdr_mask = torch.zeros(L, dtype=torch.long)
            cdr_mask[3:6] = 1
            cdr_mask[8:11] = 2
            cdr_mask[12:15] = 3
            cdr_mask[19:22] = 1
            cdr_mask[24:27] = 2
            cdr_mask[28:31] = 3

            stm = torch.zeros(L, dtype=torch.bool)
            stm[0] = True
            stm[sep_pos] = True
            stm[-1] = True

            chain_ids = torch.zeros(L, dtype=torch.long)
            chain_ids[sep_pos + 1:] = 1

            items.append({
                "input_ids": ids,
                "labels": ids.clone(),
                "attention_mask": torch.ones(L, dtype=torch.long),
                "special_tokens_mask": stm,
                "chain_ids": chain_ids,
                "cdr_mask": cdr_mask,
            })

        class CDRDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items
                self.has_coords = False
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]

        def collate(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

        loader = DataLoader(CDRDataset(items), batch_size=2, collate_fn=collate)
        evaluator = Evaluator(cfg, small_model, tokenizer)
        result = evaluator.evaluate(loader, "test")

        # Should have region-prefixed keys
        region_keys = [k for k in result if k.startswith("region/")]
        assert len(region_keys) > 0

    def test_evaluate_no_structure_skips_p_at_l(self, evaluator_cfg, small_model, tokenizer):
        evaluator = Evaluator(evaluator_cfg, small_model, tokenizer)
        loader = _make_eval_loader()  # has_coords=False
        result = evaluator.evaluate(loader, "test")
        assert "p_at_l" not in result

    def test_model_eval_mode(self, evaluator_cfg, small_model, tokenizer):
        evaluator = Evaluator(evaluator_cfg, small_model, tokenizer)
        loader = _make_eval_loader()
        # Put model in train mode, evaluator should set eval
        small_model.train()
        evaluator.evaluate(loader, "test")
        assert not small_model.training

    def test_metric_reset_between_evals(self, evaluator_cfg, small_model, tokenizer):
        evaluator = Evaluator(evaluator_cfg, small_model, tokenizer)
        loader = _make_eval_loader(num_samples=4)

        result1 = evaluator.evaluate(loader, "test")
        # Clear cache to force metric rebuild
        evaluator.clear_cache()
        result2 = evaluator.evaluate(loader, "test")

        # Results should be identical since same data + same seed
        assert result1["loss"] == pytest.approx(result2["loss"], rel=1e-4)

    def test_evaluate_max_samples(self, small_model, tokenizer):
        cfg = OmegaConf.create({
            "model": {"num_encoder_layers": 2},
            "eval": {
                "masking": {"mask_rate": 0.15, "seed": 42},
                "metrics": {
                    "loss": {"enabled": True, "max_samples": 4},
                    "masked_accuracy": {"enabled": True},
                    "perplexity": {"enabled": True},
                },
                "regions": {"enabled": False},
            },
        })
        evaluator = Evaluator(cfg, small_model, tokenizer)
        loader = _make_eval_loader(num_samples=16, B=4)
        result = evaluator.evaluate(loader, "test")
        # Should still return metrics (max_samples limits processing)
        assert "loss" in result
        assert math.isfinite(result["loss"])
