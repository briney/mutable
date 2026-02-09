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

    def test_denoising_weighted_masking_training(self, tmp_path):
        """Run a tiny training loop with weighted masking enabled."""
        torch.manual_seed(42)

        from mutable.masking import InformationWeightedMasker, UniformMasker

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
                cdr = torch.zeros(L, dtype=torch.long)
                cdr[3:6] = 1
                cdr[8:11] = 2
                items.append({
                    "input_ids": ids,
                    "labels": ids.clone(),
                    "attention_mask": torch.ones(L, dtype=torch.long),
                    "special_tokens_mask": stm,
                    "cdr_mask": cdr,
                })
            return items

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]

        def collate(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

        train_ds = SimpleDataset(_make_items(8))

        masker = InformationWeightedMasker(mask_rate=0.15)
        uniform_masker = UniformMasker(mask_rate=0.15)

        training_args = TrainingArguments(
            output_dir=str(tmp_path / "weighted_output"),
            max_steps=2,
            per_device_train_batch_size=4,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
        )

        trainer = DenoisingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collate,
            masker=masker,
            uniform_masker=uniform_masker,
            use_weighted_masking=True,
        )

        trainer.train()

    def test_flow_matching_training_smoke(self, tmp_path):
        """Run a tiny flow matching training loop."""
        torch.manual_seed(42)

        from mutable.config import FlowMatchingConfig
        from mutable.models.flow_matching import MutableFlowMatching
        from mutable.trainer.flow_trainer import FlowMatchingTrainer

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
        flow_config = FlowMatchingConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            time_embedding_dim=32,
            mu_embedding_dim=32,
            dropout=0.0,
        )
        model = MutableFlowMatching(config, flow_config)
        model.train()

        B, L = 4, 16

        def _make_items(n):
            items = []
            for _ in range(n):
                germ = torch.randint(4, 29, (L,))
                germ[0] = 0; germ[-1] = 2
                mut = torch.randint(4, 29, (L,))
                mut[0] = 0; mut[-1] = 2
                items.append({
                    "germline_input_ids": germ,
                    "germline_attention_mask": torch.ones(L, dtype=torch.long),
                    "mutated_input_ids": mut,
                    "mutated_attention_mask": torch.ones(L, dtype=torch.long),
                    "mu": torch.tensor(0.1),
                })
            return items

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]

        def collate(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

        train_ds = SimpleDataset(_make_items(8))

        training_args = TrainingArguments(
            output_dir=str(tmp_path / "flow_output"),
            max_steps=2,
            per_device_train_batch_size=4,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
        )

        trainer = FlowMatchingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collate,
        )

        trainer.train()

    def test_checkpoint_save_and_load(self, tmp_path):
        """Save a checkpoint and reload, verify weights match."""
        torch.manual_seed(42)

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

        # Save
        ckpt_path = str(tmp_path / "checkpoint")
        model.save_pretrained(ckpt_path)

        # Load
        model2 = MutableForDenoising.from_pretrained(ckpt_path)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1, p2)
