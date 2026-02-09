# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for OT-CFM flow matching."""

import torch

from mutable.flow.ot_cfm import ConditionalFlowMatcher, optimal_transport_plan


class TestConditionalFlowMatcher:
    def test_interpolation_at_t0(self):
        cfm = ConditionalFlowMatcher(sigma_min=0.0)
        x_0 = torch.randn(4, 8, 64)
        x_1 = torch.randn(4, 8, 64)
        t = torch.zeros(4)
        x_t, _ = cfm.sample_location_and_conditional_flow(x_0, x_1, t)
        torch.testing.assert_close(x_t, x_0)

    def test_interpolation_at_t1(self):
        cfm = ConditionalFlowMatcher(sigma_min=0.0)
        x_0 = torch.randn(4, 8, 64)
        x_1 = torch.randn(4, 8, 64)
        t = torch.ones(4)
        x_t, _ = cfm.sample_location_and_conditional_flow(x_0, x_1, t)
        torch.testing.assert_close(x_t, x_1)

    def test_midpoint(self):
        cfm = ConditionalFlowMatcher(sigma_min=0.0)
        x_0 = torch.zeros(2, 4, 8)
        x_1 = torch.ones(2, 4, 8)
        t = torch.full((2,), 0.5)
        x_t, _ = cfm.sample_location_and_conditional_flow(x_0, x_1, t)
        expected = 0.5 * x_0 + 0.5 * x_1
        torch.testing.assert_close(x_t, expected)

    def test_velocity_formula(self):
        sigma_min = 1e-4
        cfm = ConditionalFlowMatcher(sigma_min=sigma_min)
        x_0 = torch.randn(4, 8, 64)
        x_1 = torch.randn(4, 8, 64)
        t = torch.rand(4)
        _, u_t = cfm.sample_location_and_conditional_flow(x_0, x_1, t)
        expected = x_1 - (1.0 - sigma_min) * x_0
        torch.testing.assert_close(u_t, expected)

    def test_compute_loss_zero_for_equal(self):
        cfm = ConditionalFlowMatcher()
        v = torch.randn(4, 8, 64)
        loss = cfm.compute_loss(v, v)
        assert loss.item() == 0.0

    def test_compute_loss_positive_for_different(self):
        cfm = ConditionalFlowMatcher()
        v1 = torch.randn(4, 8, 64)
        v2 = torch.randn(4, 8, 64)
        loss = cfm.compute_loss(v1, v2)
        assert loss.item() > 0.0

    def test_t_dimension_expansion(self):
        cfm = ConditionalFlowMatcher()
        x_0 = torch.randn(4, 8, 64)
        x_1 = torch.randn(4, 8, 64)
        t = torch.rand(4)  # (4,) → should expand to (4, 1, 1)
        x_t, u_t = cfm.sample_location_and_conditional_flow(x_0, x_1, t)
        assert x_t.shape == x_0.shape
        assert u_t.shape == x_0.shape


class TestOptimalTransportPlan:
    def test_outputs_are_permutations(self):
        x_0 = torch.randn(4, 8)
        x_1 = torch.randn(4, 8)
        x_0_p, x_1_p = optimal_transport_plan(x_0, x_1)
        # Each output should contain the same set of rows (possibly reordered)
        x_0_rows = set(tuple(r.tolist()) for r in x_0)
        x_0_p_rows = set(tuple(r.tolist()) for r in x_0_p)
        assert x_0_rows == x_0_p_rows

    def test_ot_coupling_reduces_distance(self):
        torch.manual_seed(42)
        x_0 = torch.randn(8, 16)
        x_1 = torch.randn(8, 16)
        x_0_ot, x_1_ot = optimal_transport_plan(x_0, x_1)
        # OT distance should be <= identity distance
        ot_dist = (x_0_ot - x_1_ot).pow(2).sum()
        id_dist = (x_0 - x_1).pow(2).sum()
        assert ot_dist <= id_dist + 1e-6
