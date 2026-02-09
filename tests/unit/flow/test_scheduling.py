# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for timestep sampling."""

import torch

from mutable.flow.scheduling import TimestepSampler


class TestTimestepSampler:
    def test_uniform_shape(self):
        sampler = TimestepSampler(method="uniform")
        t = sampler.sample(8, torch.device("cpu"))
        assert t.shape == (8,)

    def test_uniform_range(self):
        sampler = TimestepSampler(method="uniform")
        t = sampler.sample(1000, torch.device("cpu"))
        assert t.min() >= 0.0
        assert t.max() < 1.0

    def test_logit_normal_shape(self):
        sampler = TimestepSampler(method="logit_normal")
        t = sampler.sample(8, torch.device("cpu"))
        assert t.shape == (8,)

    def test_logit_normal_range(self):
        sampler = TimestepSampler(method="logit_normal")
        t = sampler.sample(1000, torch.device("cpu"))
        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_logit_normal_biased_towards_half(self):
        sampler = TimestepSampler(method="logit_normal", logit_normal_mean=0.0, logit_normal_std=1.0)
        torch.manual_seed(42)
        t = sampler.sample(10000, torch.device("cpu"))
        # Mean should be near 0.5 due to symmetry of logit-normal with mean=0
        assert 0.4 < t.mean().item() < 0.6

    def test_default_method_is_uniform(self):
        sampler = TimestepSampler()
        assert sampler.method == "uniform"
