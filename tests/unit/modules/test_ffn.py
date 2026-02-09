# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for feed-forward network modules."""

import torch

from mutable.modules.ffn import DenseFFN, GluFFN


class TestDenseFFN:
    def test_forward_shape(self):
        ffn = DenseFFN(model_dim=64, ffn_dim=128, bias=True, activation="gelu")
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_relu_activation(self):
        ffn = DenseFFN(model_dim=64, ffn_dim=128, bias=True, activation="relu")
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_bias_vs_no_bias_param_count(self):
        ffn_bias = DenseFFN(model_dim=64, ffn_dim=128, bias=True, activation="gelu")
        ffn_no_bias = DenseFFN(model_dim=64, ffn_dim=128, bias=False, activation="gelu")
        params_bias = sum(p.numel() for p in ffn_bias.parameters())
        params_no_bias = sum(p.numel() for p in ffn_no_bias.parameters())
        assert params_bias > params_no_bias


class TestGluFFN:
    def test_forward_shape(self):
        ffn = GluFFN(model_dim=64, ffn_dim=128, bias=True, activation="swiglu")
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_geglu_variant(self):
        ffn = GluFFN(model_dim=64, ffn_dim=128, bias=True, activation="geglu")
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_reglu_variant(self):
        ffn = GluFFN(model_dim=64, ffn_dim=128, bias=True, activation="reglu")
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_more_params_than_dense(self):
        dense = DenseFFN(model_dim=64, ffn_dim=128, bias=True, activation="gelu")
        glu = GluFFN(model_dim=64, ffn_dim=128, bias=True, activation="swiglu")
        dense_params = sum(p.numel() for p in dense.parameters())
        glu_params = sum(p.numel() for p in glu.parameters())
        assert glu_params > dense_params
