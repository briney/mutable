# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Callable, Optional

import torch

__all__ = ["ODEIntegrator", "euler_step", "rk4_step"]


def euler_step(
    velocity_fn: Callable,
    x: torch.Tensor,
    t: float,
    dt: float,
) -> torch.Tensor:
    """Single Euler integration step: x_{t+dt} = x_t + dt * v(x_t, t)."""
    return x + dt * velocity_fn(x, t)


def rk4_step(
    velocity_fn: Callable,
    x: torch.Tensor,
    t: float,
    dt: float,
) -> torch.Tensor:
    """Single RK4 integration step."""
    k1 = velocity_fn(x, t)
    k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = velocity_fn(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class ODEIntegrator:
    """
    ODE integrator for flow matching generation.

    Integrates dx/dt = v(x, t) from t=0 to t=1 using the specified solver.

    Parameters
    ----------
    solver : str, default="euler"
        ODE solver type ("euler", "rk4", "adaptive").
    num_steps : int, default=100
        Number of integration steps (for euler/rk4).
    atol : float, default=1e-5
        Absolute tolerance (for adaptive solver).
    rtol : float, default=1e-5
        Relative tolerance (for adaptive solver).
    """

    def __init__(
        self,
        solver: str = "euler",
        num_steps: int = 100,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        self.solver = solver
        self.num_steps = num_steps
        self.atol = atol
        self.rtol = rtol

    @torch.no_grad()
    def integrate(
        self,
        velocity_fn: Callable,
        x_0: torch.Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> torch.Tensor:
        """
        Integrate the ODE from t_start to t_end.

        Parameters
        ----------
        velocity_fn : Callable
            Function (x, t) -> velocity, where t is a float.
        x_0 : torch.Tensor
            Initial state.
        t_start : float
            Start time.
        t_end : float
            End time.

        Returns
        -------
        torch.Tensor
            Final state at t_end.
        """
        if self.solver == "adaptive":
            return self._adaptive_integrate(velocity_fn, x_0, t_start, t_end)
        elif self.solver == "rk4":
            step_fn = rk4_step
        else:
            step_fn = euler_step

        dt = (t_end - t_start) / self.num_steps
        x = x_0
        t = t_start
        for _ in range(self.num_steps):
            x = step_fn(velocity_fn, x, t, dt)
            t += dt
        return x

    def _adaptive_integrate(
        self,
        velocity_fn: Callable,
        x_0: torch.Tensor,
        t_start: float,
        t_end: float,
    ) -> torch.Tensor:
        """Integrate using torchdiffeq's adaptive solver."""
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError(
                "Adaptive ODE solver requires torchdiffeq. "
                "Install with: pip install torchdiffeq"
            )

        # wrap velocity_fn for torchdiffeq interface: (t, x) -> dx/dt
        class VelocityWrapper(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, t, x):
                return self.fn(x, t.item())

        t_span = torch.tensor([t_start, t_end], device=x_0.device, dtype=x_0.dtype)
        wrapper = VelocityWrapper(velocity_fn)
        result = odeint(
            wrapper,
            x_0,
            t_span,
            method="dopri5",
            atol=self.atol,
            rtol=self.rtol,
        )
        return result[-1]
