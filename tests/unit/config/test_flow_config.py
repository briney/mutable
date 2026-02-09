# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for FlowMatchingConfig."""

import pytest

from mutable.config import FlowMatchingConfig


class TestFlowMatchingConfig:
    def test_defaults(self):
        config = FlowMatchingConfig()
        assert config.hidden_size == 320
        assert config.num_layers == 6
        assert config.num_attention_heads == 20
        assert config.activation == "swiglu"
        assert config.ode_solver == "euler"
        assert config.ode_steps == 100
        assert config.timestep_sampling == "uniform"

    def test_intermediate_size_defaults_to_hidden_times_4(self):
        config = FlowMatchingConfig(hidden_size=64, num_attention_heads=4)
        assert config.intermediate_size == 256

    def test_dropout_cascading(self):
        config = FlowMatchingConfig(dropout=0.2)
        assert config.attention_dropout == 0.2
        assert config.hidden_dropout == 0.2

    def test_dropout_explicit(self):
        config = FlowMatchingConfig(dropout=0.2, attention_dropout=0.05)
        assert config.attention_dropout == 0.05
        assert config.hidden_dropout == 0.2

    def test_invalid_ode_solver_raises(self):
        with pytest.raises(ValueError, match="Invalid ode_solver"):
            FlowMatchingConfig(ode_solver="badsolve")

    def test_hidden_size_divisibility_check(self):
        with pytest.raises(ValueError, match="must be divisible"):
            FlowMatchingConfig(hidden_size=65, num_attention_heads=4)

    def test_type_conversions(self):
        config = FlowMatchingConfig(hidden_size=64, num_attention_heads=4)
        assert isinstance(config.hidden_size, int)
        assert isinstance(config.num_layers, int)
        assert isinstance(config.sigma_min, float)
