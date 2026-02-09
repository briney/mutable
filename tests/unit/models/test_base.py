# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for base model classes."""

import torch
import torch.nn as nn

from mutable.config import MutableConfig
from mutable.models.base import MutablePreTrainedModel, ParameterCountMixin
from mutable.models.mutable_denoising import MutableForDenoising


class TestInitWeights:
    def test_linear_weight_normal(self, small_config):
        model = MutableForDenoising(small_config)
        # Check a linear layer weight is not all zeros
        linear = model.lm_head
        assert linear.weight.std() > 0

    def test_linear_bias_zero(self, small_config):
        model = MutableForDenoising(small_config)
        # Find a linear with bias
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert module.bias.abs().max() == 0.0
                break

    def test_embedding_weights_normal(self, small_config):
        model = MutableForDenoising(small_config)
        emb = model.mutable.encoder.word_embeddings
        assert emb.weight.std() > 0

    def test_padding_idx_row_zeros(self, small_config):
        model = MutableForDenoising(small_config)
        emb = model.mutable.encoder.word_embeddings
        assert emb.weight.data[emb.padding_idx].abs().max() == 0.0

    def test_layernorm_weight_one_bias_zero(self, small_config):
        model = MutableForDenoising(small_config)
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) and module.weight is not None:
                torch.testing.assert_close(
                    module.weight, torch.ones_like(module.weight)
                )
                if module.bias is not None:
                    torch.testing.assert_close(
                        module.bias, torch.zeros_like(module.bias)
                    )
                break

    def test_adaptive_layernorm_no_crash(self, small_config, small_flow_config):
        """AdaptiveLayerNorm (elementwise_affine=False) should not crash in _init_weights."""
        from mutable.models.flow_matching import MutableFlowMatching

        torch.manual_seed(42)
        model = MutableFlowMatching(small_config, small_flow_config)
        # If we got here, init didn't crash
        assert model is not None


class TestParameterCountMixin:
    def test_count_parameters_returns_int(self, small_model):
        count = small_model.count_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_trainable_only_excludes_frozen(self, small_config):
        model = MutableForDenoising(small_config)
        total = model.count_parameters(only_trainable=False)
        # Freeze some params
        for p in model.lm_head.parameters():
            p.requires_grad = False
        trainable = model.count_parameters(only_trainable=True)
        assert trainable < total
