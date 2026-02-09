# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Integration tests for trainers."""

import torch
from transformers import TrainingArguments

from mutable.config import MutableConfig, FlowMatchingConfig
from mutable.masking import InformationWeightedMasker, UniformMasker
from mutable.models.mutable_denoising import MutableForDenoising
from mutable.models.flow_matching import MutableFlowMatching
from mutable.trainer.denoising_trainer import DenoisingTrainer
from mutable.trainer.flow_trainer import FlowMatchingTrainer


class TestDenoisingTrainerIntegration:
    @staticmethod
    def _model_batch(batch):
        return {k: v for k, v in batch.items() if k not in ("special_tokens_mask", "chain_ids", "cdr_mask", "nongermline_mask")}

    def test_compute_loss_standard(self, small_config, sample_batch, tmp_path):
        model = MutableForDenoising(small_config)
        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=1,
            per_device_train_batch_size=2, report_to="none",
            use_cpu=True,
        )
        trainer = DenoisingTrainer(model=model, args=args)
        loss = trainer.compute_loss(model, self._model_batch(sample_batch))
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_compute_loss_weighted_masking(self, small_config, sample_batch_with_annotations, tmp_path):
        model = MutableForDenoising(small_config)
        masker = InformationWeightedMasker(mask_rate=0.15)
        uniform_masker = UniformMasker(mask_rate=0.15)
        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=1,
            per_device_train_batch_size=2, report_to="none",
            use_cpu=True,
        )
        trainer = DenoisingTrainer(
            model=model, args=args,
            masker=masker, uniform_masker=uniform_masker,
            use_weighted_masking=True,
        )
        # Only include keys that the trainer/model understand
        batch = {k: v for k, v in sample_batch_with_annotations.items()
                 if k in ("input_ids", "labels", "attention_mask", "special_tokens_mask",
                          "cdr_mask", "nongermline_mask")}
        loss = trainer.compute_loss(model, batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_apply_masking_dispatches_correctly(self, small_config, sample_batch, tmp_path):
        model = MutableForDenoising(small_config)
        masker = InformationWeightedMasker(mask_rate=0.15)
        uniform_masker = UniformMasker(mask_rate=0.15)
        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=1,
            per_device_train_batch_size=2, report_to="none",
            use_cpu=True,
        )
        trainer = DenoisingTrainer(
            model=model, args=args,
            masker=masker, uniform_masker=uniform_masker,
            use_weighted_masking=True,
        )
        # Without annotation masks, should fall back to uniform
        batch = dict(sample_batch)
        batch["special_tokens_mask"] = torch.zeros(4, 32, dtype=torch.bool)
        result = trainer._apply_masking(batch)
        assert "input_ids" in result
        # annotation keys should be removed
        assert "special_tokens_mask" not in result


class TestFlowMatchingTrainerIntegration:
    def test_compute_loss(self, small_config, small_flow_config, tmp_path):
        model = MutableFlowMatching(small_config, small_flow_config)
        model.train()
        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=1,
            per_device_train_batch_size=2, report_to="none",
            use_cpu=True,
        )
        trainer = FlowMatchingTrainer(model=model, args=args)

        B, L = 2, 16
        inputs = {
            "germline_input_ids": torch.randint(4, 29, (B, L)),
            "germline_attention_mask": torch.ones(B, L, dtype=torch.long),
            "mutated_input_ids": torch.randint(4, 29, (B, L)),
            "mutated_attention_mask": torch.ones(B, L, dtype=torch.long),
            "mu": torch.rand(B),
        }
        inputs["germline_input_ids"][:, 0] = 0; inputs["germline_input_ids"][:, -1] = 2
        inputs["mutated_input_ids"][:, 0] = 0; inputs["mutated_input_ids"][:, -1] = 2

        loss = trainer.compute_loss(model, inputs)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_return_outputs_true(self, small_config, small_flow_config, tmp_path):
        model = MutableFlowMatching(small_config, small_flow_config)
        model.train()
        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=1,
            per_device_train_batch_size=2, report_to="none",
            use_cpu=True,
        )
        trainer = FlowMatchingTrainer(model=model, args=args)

        B, L = 2, 16
        inputs = {
            "germline_input_ids": torch.randint(4, 29, (B, L)),
            "germline_attention_mask": torch.ones(B, L, dtype=torch.long),
            "mutated_input_ids": torch.randint(4, 29, (B, L)),
            "mutated_attention_mask": torch.ones(B, L, dtype=torch.long),
        }
        inputs["germline_input_ids"][:, 0] = 0; inputs["germline_input_ids"][:, -1] = 2
        inputs["mutated_input_ids"][:, 0] = 0; inputs["mutated_input_ids"][:, -1] = 2

        result = trainer.compute_loss(model, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        loss, outputs = result
        assert outputs.loss is not None
