# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for ODE integrators."""

import pytest
import torch

from mutable.flow.ode import euler_step, rk4_step, ODEIntegrator


class TestEulerStep:
    def test_constant_velocity(self):
        c = torch.tensor([1.0, 2.0])
        velocity_fn = lambda x, t: c
        x = torch.zeros(2)
        dt = 0.1
        x_next = euler_step(velocity_fn, x, 0.0, dt)
        expected = x + dt * c
        torch.testing.assert_close(x_next, expected)

    def test_zero_velocity_unchanged(self):
        velocity_fn = lambda x, t: torch.zeros_like(x)
        x = torch.tensor([3.0, 4.0])
        x_next = euler_step(velocity_fn, x, 0.0, 0.1)
        torch.testing.assert_close(x_next, x)


class TestRK4Step:
    def test_constant_velocity_matches_euler(self):
        c = torch.tensor([1.0, 2.0])
        velocity_fn = lambda x, t: c
        x = torch.zeros(2)
        dt = 0.1
        euler_result = euler_step(velocity_fn, x, 0.0, dt)
        rk4_result = rk4_step(velocity_fn, x, 0.0, dt)
        torch.testing.assert_close(rk4_result, euler_result)

    def test_linear_velocity(self):
        # v(x, t) = t → exact integral from 0 to 1 is 0.5
        # dt=1.0 single step: RK4 should be much closer than Euler
        velocity_fn = lambda x, t: torch.full_like(x, t)
        x = torch.zeros(2)
        rk4_result = rk4_step(velocity_fn, x, 0.0, 1.0)
        euler_result = euler_step(velocity_fn, x, 0.0, 1.0)
        analytical = torch.full((2,), 0.5)
        rk4_err = (rk4_result - analytical).abs().max()
        euler_err = (euler_result - analytical).abs().max()
        assert rk4_err < euler_err


class TestODEIntegrator:
    def test_euler_constant_velocity(self):
        c = torch.tensor([1.0, 2.0])
        velocity_fn = lambda x, t: c
        integrator = ODEIntegrator(solver="euler", num_steps=100)
        x_0 = torch.zeros(2)
        x_1 = integrator.integrate(velocity_fn, x_0)
        # integral of constant c from 0 to 1 = c
        torch.testing.assert_close(x_1, c, atol=1e-4, rtol=1e-4)

    def test_rk4_constant_velocity(self):
        c = torch.tensor([1.0, 2.0])
        velocity_fn = lambda x, t: c
        integrator = ODEIntegrator(solver="rk4", num_steps=10)
        x_0 = torch.zeros(2)
        x_1 = integrator.integrate(velocity_fn, x_0)
        torch.testing.assert_close(x_1, c, atol=1e-4, rtol=1e-4)

    def test_rk4_more_accurate_for_nonlinear(self):
        # v(x, t) = t^2 → exact integral from 0 to 1 is 1/3
        velocity_fn = lambda x, t: torch.full_like(x, t ** 2)
        x_0 = torch.zeros(4)
        analytical = torch.full((4,), 1.0 / 3.0)

        euler_int = ODEIntegrator(solver="euler", num_steps=50)
        rk4_int = ODEIntegrator(solver="rk4", num_steps=50)

        euler_result = euler_int.integrate(velocity_fn, x_0)
        rk4_result = rk4_int.integrate(velocity_fn, x_0)

        euler_err = (euler_result - analytical).abs().max()
        rk4_err = (rk4_result - analytical).abs().max()
        assert rk4_err < euler_err

    def test_num_steps_affects_accuracy(self):
        velocity_fn = lambda x, t: torch.full_like(x, t)
        x_0 = torch.zeros(2)
        analytical = torch.full((2,), 0.5)

        int_10 = ODEIntegrator(solver="euler", num_steps=10)
        int_100 = ODEIntegrator(solver="euler", num_steps=100)

        err_10 = (int_10.integrate(velocity_fn, x_0) - analytical).abs().max()
        err_100 = (int_100.integrate(velocity_fn, x_0) - analytical).abs().max()
        assert err_100 < err_10

    def test_adaptive_requires_torchdiffeq(self):
        velocity_fn = lambda x, t: torch.zeros(2)
        integrator = ODEIntegrator(solver="adaptive", num_steps=10)
        try:
            integrator.integrate(velocity_fn, torch.zeros(2))
        except ImportError:
            pytest.skip("torchdiffeq not installed")

    def test_custom_t_range(self):
        c = torch.tensor([2.0])
        velocity_fn = lambda x, t: c
        integrator = ODEIntegrator(solver="euler", num_steps=100)
        x_0 = torch.zeros(1)
        x_1 = integrator.integrate(velocity_fn, x_0, t_start=0.0, t_end=0.5)
        # integral of 2 from 0 to 0.5 = 1
        torch.testing.assert_close(x_1, torch.tensor([1.0]), atol=1e-3, rtol=1e-3)
