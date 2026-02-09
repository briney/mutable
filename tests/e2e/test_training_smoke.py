# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""End-to-end smoke test: DenoisingTrainer with eval harness."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path

import torch
import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from mutable.config import MutableConfig
from mutable.eval.evaluator import Evaluator
from mutable.models.mutable_denoising import MutableForDenoising
from mutable.tokenizer import MutableTokenizer
from mutable.trainer.denoising_trainer import DenoisingTrainer


@pytest.mark.slow
class TestTrainingSmoke:
    def test_denoising_training_with_eval(self, tmp_path):
        """Run a tiny training loop with evaluation enabled."""
        torch.manual_seed(42)

        # --- Model ---
        config = MutableConfig(
            hidden_size=64,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_attention_heads=4,
            num_latents=8,
            intermediate_size=128,
            vocab_size=32,
            dropout=0.0,
        )
        model = MutableForDenoising(config)
        tokenizer = MutableTokenizer()

        # --- Synthetic datasets ---
        L, sep_pos = 32, 16

        def _make_items(n):
            items = []
            for _ in range(n):
                ids = torch.randint(4, 29, (L,))
                ids[0] = 0
                ids[sep_pos] = 29
                ids[-1] = 2
                stm = torch.zeros(L, dtype=torch.bool)
                stm[0] = True
                stm[sep_pos] = True
                stm[-1] = True
                items.append({
                    "input_ids": ids,
                    "labels": ids.clone(),
                    "attention_mask": torch.ones(L, dtype=torch.long),
                    "special_tokens_mask": stm,
                })
            return items

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, items, has_coords=False):
                self.items = items
                self.has_coords = has_coords
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]

        def collate(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

        train_ds = SimpleDataset(_make_items(8))
        eval_ds = SimpleDataset(_make_items(4))
        eval_loader = DataLoader(eval_ds, batch_size=2, collate_fn=collate)

        # --- Config for evaluator ---
        eval_cfg = OmegaConf.create({
            "model": {"num_encoder_layers": 2},
            "eval": {
                "masking": {"mask_rate": 0.15, "seed": 42},
                "metrics": {
                    "loss": {"enabled": True},
                    "masked_accuracy": {"enabled": True},
                    "perplexity": {"enabled": True},
                },
                "regions": {"enabled": False},
            },
        })

        evaluator = Evaluator(eval_cfg, model, tokenizer)

        # --- Training args ---
        output_dir = str(tmp_path / "output")
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=2,
            per_device_train_batch_size=4,
            eval_strategy="steps",
            eval_steps=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
        )

        trainer = DenoisingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collate,
            evaluator=evaluator,
            eval_loaders={"val": eval_loader},
        )

        # --- Train ---
        trainer.train()

        # --- Evaluate ---
        result = trainer.evaluate()

        # Verify results
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check for eval-prefixed keys
        eval_keys = [k for k in result if "eval" in k.lower() or "loss" in k or "acc" in k]
        assert len(eval_keys) > 0

        # All values should be finite
        for k, v in result.items():
            if isinstance(v, (int, float)):
                assert math.isfinite(v), f"{k} = {v} is not finite"
