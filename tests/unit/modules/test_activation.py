# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for activation functions."""

import math

import pytest
import torch
import torch.nn as nn

from mutable.modules.activation import get_activation_fn, GELU, GLU


class TestGetActivationFn:
    def test_gelu_returns_gelu(self):
        act = get_activation_fn("gelu")
        assert isinstance(act, GELU)

    def test_relu_returns_relu(self):
        act = get_activation_fn("relu")
        assert isinstance(act, nn.ReLU)

    def test_tanh_returns_tanh(self):
        act = get_activation_fn("tanh")
        assert isinstance(act, nn.Tanh)

    @pytest.mark.parametrize("name", ["glu", "swiglu", "geglu", "reglu"])
    def test_glu_variants_return_glu(self, name):
        act = get_activation_fn(name, input_dim=64, output_dim=128, bias=True)
        assert isinstance(act, GLU)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            get_activation_fn("badact")


class TestGELU:
    def test_output_shape_matches_input(self):
        act = GELU()
        x = torch.randn(2, 10, 64)
        out = act(x)
        assert out.shape == x.shape

    def test_values_match_formula(self):
        act = GELU()
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        expected = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        out = act(x)
        torch.testing.assert_close(out, expected)

    def test_zero_input_returns_zero(self):
        act = GELU()
        x = torch.zeros(5)
        out = act(x)
        torch.testing.assert_close(out, torch.zeros(5))


class TestGLU:
    def test_forward_shape(self):
        glu = GLU(in_dim=64, out_dim=128, activation="swiglu", bias=True)
        x = torch.randn(2, 10, 64)
        out = glu(x)
        assert out.shape == (2, 10, 128)

    @pytest.mark.parametrize("variant", ["glu", "swiglu", "geglu", "reglu"])
    def test_all_variants_work(self, variant):
        glu = GLU(in_dim=32, out_dim=64, activation=variant, bias=True)
        x = torch.randn(2, 5, 32)
        out = glu(x)
        assert out.shape == (2, 5, 64)
