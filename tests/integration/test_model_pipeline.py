# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Integration tests for model pipelines."""

import torch

from mutable.config import MutableConfig, FlowMatchingConfig
from mutable.models.mutable_denoising import MutableForDenoising
from mutable.models.flow_matching import MutableFlowMatching


class TestDenoisingPipeline:
    @staticmethod
    def _model_batch(sample_batch):
        return {k: v for k, v in sample_batch.items() if k not in ("special_tokens_mask",)}

    def test_full_encode_decode_shape(self, small_model, sample_batch):
        batch = self._model_batch(sample_batch)
        out = small_model(**batch, return_dict=True)
        B, L = sample_batch["input_ids"].shape
        assert out.logits.shape == (B, L, 32)
        assert out.latent_states.shape == (B, 8, 64)

    def test_forward_with_labels_produces_loss(self, small_model, sample_batch):
        batch = self._model_batch(sample_batch)
        out = small_model(**batch, return_dict=True)
        assert out.loss is not None
        assert out.loss.dim() == 0
        assert out.loss.item() > 0

    def test_loss_backward_succeeds(self, small_config):
        model = MutableForDenoising(small_config)
        model.train()
        B, L = 2, 16
        ids = torch.randint(4, 29, (B, L))
        ids[:, 0] = 0; ids[:, -1] = 2
        labels = ids.clone()
        mask = torch.ones(B, L, dtype=torch.long)
        out = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
        out.loss.backward()
        # Check at least some gradients are non-None
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_output_attentions_populates_all_fields(self, small_model, sample_batch):
        batch = self._model_batch(sample_batch)
        out = small_model(**batch, output_attentions=True, return_dict=True)
        assert out.encoder_attentions is not None
        assert len(out.encoder_attentions) == 2


class TestFlowMatchingPipeline:
    def test_forward_produces_loss(self, small_flow_model):
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        mut_ids = torch.randint(4, 29, (B, L))
        mut_ids[:, 0] = 0; mut_ids[:, -1] = 2

        small_flow_model.train()
        out = small_flow_model(
            germline_input_ids=germ_ids, mutated_input_ids=mut_ids,
            mu=torch.rand(B), return_dict=True,
        )
        small_flow_model.eval()
        assert out.loss is not None
        assert out.loss.item() > 0

    def test_generate_produces_integer_ids(self, small_flow_model):
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        gen = small_flow_model.generate(
            germline_input_ids=germ_ids, num_steps=2, solver="euler",
        )
        assert gen.dtype == torch.long
        assert gen.shape == (B, L)

    def test_backbone_params_no_gradient(self, small_config, small_flow_config):
        model = MutableFlowMatching(small_config, small_flow_config)
        model.train()
        B, L = 2, 16
        germ_ids = torch.randint(4, 29, (B, L))
        germ_ids[:, 0] = 0; germ_ids[:, -1] = 2
        mut_ids = torch.randint(4, 29, (B, L))
        mut_ids[:, 0] = 0; mut_ids[:, -1] = 2

        out = model(
            germline_input_ids=germ_ids, mutated_input_ids=mut_ids,
            return_dict=True,
        )
        out.loss.backward()

        for p in model.mutable.parameters():
            assert p.grad is None
