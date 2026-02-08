# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Literal

from transformers import PretrainedConfig


class FlowMatchingConfig(PretrainedConfig):
    """
    Configuration for the flow matching network (Phase 2).

    Parameters
    ----------
    hidden_size : int, default=320
        Hidden dimension (should match MutableConfig.latent_dim).
    num_layers : int, default=6
        Number of flow transformer layers.
    num_attention_heads : int, default=20
        Number of attention heads.
    intermediate_size : int, default=None
        FFN intermediate dimension. Defaults to hidden_size * 4.
    activation : str, default="swiglu"
        Activation function for FFN layers.
    time_embedding_dim : int, default=128
        Dimension of the sinusoidal time embedding.
    mu_embedding_dim : int, default=128
        Dimension of the sinusoidal mutation intensity embedding.
    conditioning_method : str, default="adaln"
        How to condition the flow network. Currently only "adaln" supported.
    ode_solver : str, default="euler"
        ODE solver for generation ("euler", "rk4", "adaptive").
    ode_steps : int, default=100
        Number of integration steps for euler/rk4 solvers.
    sigma_min : float, default=1e-4
        Minimum noise level for flow matching.
    timestep_sampling : str, default="uniform"
        How to sample timesteps during training ("uniform" or "logit_normal").
    logit_normal_mean : float, default=0.0
        Mean for logit-normal timestep sampling.
    logit_normal_std : float, default=1.0
        Std for logit-normal timestep sampling.
    dropout : float, default=0.1
        Default dropout rate.
    attention_dropout : float, default=None
        Attention dropout. Defaults to dropout.
    hidden_dropout : float, default=None
        Hidden dropout. Defaults to dropout.
    ffn_bias : bool, default=True
        Whether to use bias in FFN layers.
    initializer_range : float, default=0.02
        Standard deviation for weight initialization.
    layer_norm_eps : float, default=1e-5
        Epsilon for layer normalization.
    """

    model_type = "mutable_flow"

    def __init__(
        self,
        hidden_size: int = 320,
        num_layers: int = 6,
        num_attention_heads: int = 20,
        intermediate_size: Optional[int] = None,
        activation: Literal[
            "gelu", "relu", "glu", "swiglu", "geglu", "reglu"
        ] = "swiglu",
        # conditioning
        time_embedding_dim: int = 128,
        mu_embedding_dim: int = 128,
        conditioning_method: Literal["adaln"] = "adaln",
        # ODE
        ode_solver: Literal["euler", "rk4", "adaptive"] = "euler",
        ode_steps: int = 100,
        sigma_min: float = 1e-4,
        # timestep sampling
        timestep_sampling: Literal["uniform", "logit_normal"] = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        # dropout
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        hidden_dropout: Optional[float] = None,
        # ffn
        ffn_bias: bool = True,
        # init
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size or hidden_size * 4)
        self.activation = activation.lower()

        # conditioning
        self.time_embedding_dim = int(time_embedding_dim)
        self.mu_embedding_dim = int(mu_embedding_dim)
        self.conditioning_method = conditioning_method.lower()

        # ODE
        self.ode_solver = ode_solver.lower()
        self.ode_steps = int(ode_steps)
        self.sigma_min = float(sigma_min)

        # timestep sampling
        self.timestep_sampling = timestep_sampling.lower()
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)

        # dropout
        self.dropout = float(dropout)
        self.attention_dropout = float(
            attention_dropout if attention_dropout is not None else dropout
        )
        self.hidden_dropout = float(
            hidden_dropout if hidden_dropout is not None else dropout
        )

        # ffn
        self.ffn_bias = bool(ffn_bias)

        # init
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)

        # validation
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.ode_solver not in ["euler", "rk4", "adaptive"]:
            raise ValueError(
                f"Invalid ode_solver: {self.ode_solver}. "
                "Options are 'euler', 'rk4', or 'adaptive'."
            )
