# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import MutableConfig, FlowMatchingConfig
from ..modules import SinusoidalTimestepEmbedding, FlowTransformerLayer
from ..outputs import FlowMatchingOutput
from ..flow import ConditionalFlowMatcher, ODEIntegrator, TimestepSampler
from .base import MutablePreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin
from .mutable import MutableModel

__all__ = ["FlowNetwork", "MutableFlowMatching"]


class FlowNetwork(nn.Module):
    """
    Transformer-based velocity field network for flow matching.

    Predicts velocity given noisy latents, germline latents, timestep, and mutation intensity.
    Uses AdaLN conditioning (DiT-style).

    Parameters
    ----------
    config : FlowMatchingConfig
        Flow matching configuration.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # timestep and mutation intensity embeddings
        self.time_embed = SinusoidalTimestepEmbedding(config.time_embedding_dim)
        self.mu_embed = SinusoidalTimestepEmbedding(config.mu_embedding_dim)

        # project concatenated conditioning to shared dim
        conditioning_dim = config.time_embedding_dim + config.mu_embedding_dim
        self.conditioning_proj = nn.Linear(conditioning_dim, conditioning_dim)

        # transformer layers
        self.layers = nn.ModuleList(
            [
                FlowTransformerLayer(config, conditioning_dim=conditioning_dim)
                for _ in range(config.num_layers)
            ]
        )

        # final norm and output projection
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        germline_latents: torch.Tensor,
        t: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_latents : torch.Tensor
            Interpolated latent states, shape (batch, num_latents, hidden_size).
        germline_latents : torch.Tensor
            Germline-encoded latent states, shape (batch, num_latents, hidden_size).
        t : torch.Tensor
            Timesteps, shape (batch,).
        mu : torch.Tensor
            Mutation intensity, shape (batch,).

        Returns
        -------
        torch.Tensor
            Predicted velocity, shape (batch, num_latents, hidden_size).
        """
        # embed timestep and mutation intensity
        t_emb = self.time_embed(t)
        mu_emb = self.mu_embed(mu)

        # concatenate and project conditioning
        conditioning = torch.cat([t_emb, mu_emb], dim=-1)
        conditioning = self.conditioning_proj(conditioning)

        # pass through transformer layers
        x = noisy_latents
        for layer in self.layers:
            x = layer(x, germline_latents=germline_latents, conditioning=conditioning)

        # final norm and output projection
        x = self.final_norm(x)
        x = self.output_proj(x)

        return x


class MutableFlowMatching(
    MutablePreTrainedModel, FreezeBaseModelMixin, ParameterCountMixin
):
    """
    Phase 2: Flow matching in latent space for SHM generation.

    Uses a frozen MutableModel (encoder + bottleneck + decoder) as the backbone,
    and trains a FlowNetwork to predict velocity fields that transform germline
    latents into mutated latents.

    Parameters
    ----------
    config : MutableConfig
        Base model configuration.
    flow_config : FlowMatchingConfig, optional
        Flow network configuration. If None, uses default FlowMatchingConfig
        with hidden_size matching config.latent_dim.
    """

    config_class = MutableConfig
    base_model_prefix = "mutable"

    def __init__(
        self,
        config: MutableConfig,
        flow_config: Optional[FlowMatchingConfig] = None,
    ):
        super().__init__(config)
        self.config = config

        # flow config defaults
        if flow_config is None:
            flow_config = FlowMatchingConfig(hidden_size=config.latent_dim)
        self.flow_config = flow_config

        # frozen backbone
        self.mutable = MutableModel(config)

        # flow network
        self.flow_network = FlowNetwork(flow_config)

        # decoder LM head (for generation)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # flow matching utilities
        self.cfm = ConditionalFlowMatcher(sigma_min=flow_config.sigma_min)
        self.timestep_sampler = TimestepSampler(
            method=flow_config.timestep_sampling,
            logit_normal_mean=flow_config.logit_normal_mean,
            logit_normal_std=flow_config.logit_normal_std,
        )

        self.init_weights()

    def forward(
        self,
        germline_input_ids: Optional[torch.LongTensor] = None,
        germline_attention_mask: Optional[torch.LongTensor] = None,
        mutated_input_ids: Optional[torch.LongTensor] = None,
        mutated_attention_mask: Optional[torch.LongTensor] = None,
        mu: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[FlowMatchingOutput, tuple]:
        """
        Training forward pass.

        Parameters
        ----------
        germline_input_ids : torch.LongTensor
            Germline sequence token IDs.
        germline_attention_mask : torch.LongTensor
            Attention mask for germline input.
        mutated_input_ids : torch.LongTensor
            Mutated sequence token IDs.
        mutated_attention_mask : torch.LongTensor
            Attention mask for mutated input.
        mu : torch.FloatTensor
            Mutation intensity, shape (batch,). Values in [0, 1].
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        device = germline_input_ids.device
        batch_size = germline_input_ids.shape[0]

        # encode both germline and mutated sequences through frozen backbone
        with torch.no_grad():
            germline_bottleneck = self.mutable.encode(
                input_ids=germline_input_ids,
                attention_mask=germline_attention_mask,
                return_dict=True,
            )
            mutated_bottleneck = self.mutable.encode(
                input_ids=mutated_input_ids,
                attention_mask=mutated_attention_mask,
                return_dict=True,
            )

        x_0 = germline_bottleneck.latent_states  # source
        x_1 = mutated_bottleneck.latent_states  # target
        germline_latents = germline_bottleneck.latent_states

        # sample timesteps
        t = self.timestep_sampler.sample(batch_size, device)

        # interpolate and get target velocity
        x_t, u_t = self.cfm.sample_location_and_conditional_flow(x_0, x_1, t)

        # default mu if not provided
        if mu is None:
            mu = torch.zeros(batch_size, device=device)

        # predict velocity
        predicted_velocity = self.flow_network(
            noisy_latents=x_t,
            germline_latents=germline_latents,
            t=t,
            mu=mu,
        )

        # compute loss
        loss = self.cfm.compute_loss(predicted_velocity, u_t)

        if not return_dict:
            return tuple(
                v
                for v in [loss, predicted_velocity, u_t, x_t]
                if v is not None
            )
        return FlowMatchingOutput(
            loss=loss,
            predicted_velocity=predicted_velocity,
            target_velocity=u_t,
            latent_t=x_t,
        )

    @torch.no_grad()
    def generate(
        self,
        germline_input_ids: torch.LongTensor,
        germline_attention_mask: Optional[torch.LongTensor] = None,
        mu: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate mutated sequences by flowing germline latents through the learned velocity field.

        Parameters
        ----------
        germline_input_ids : torch.LongTensor
            Germline sequence token IDs.
        germline_attention_mask : torch.LongTensor, optional
            Attention mask for germline input.
        mu : torch.FloatTensor, optional
            Mutation intensity, shape (batch,). Defaults to zeros.
        decoder_input_ids : torch.LongTensor, optional
            Decoder input for autoregressive decoding. If None, uses BOS token.
        num_steps : int, optional
            Override for ODE integration steps.
        solver : str, optional
            Override for ODE solver type.

        Returns
        -------
        torch.Tensor
            Generated token IDs, shape (batch, seq_len).
        """
        device = germline_input_ids.device
        batch_size = germline_input_ids.shape[0]

        # encode germline
        germline_bottleneck = self.mutable.encode(
            input_ids=germline_input_ids,
            attention_mask=germline_attention_mask,
            return_dict=True,
        )
        germline_latents = germline_bottleneck.latent_states

        # sample noise as starting point
        x_0 = torch.randn_like(germline_latents)

        # default mu
        if mu is None:
            mu = torch.zeros(batch_size, device=device)

        # create velocity function for ODE integrator
        def velocity_fn(x, t_scalar):
            t_tensor = torch.full((batch_size,), t_scalar, device=device)
            return self.flow_network(
                noisy_latents=x,
                germline_latents=germline_latents,
                t=t_tensor,
                mu=mu,
            )

        # integrate ODE
        integrator = ODEIntegrator(
            solver=solver or self.flow_config.ode_solver,
            num_steps=num_steps or self.flow_config.ode_steps,
        )
        x_1 = integrator.integrate(velocity_fn, x_0)

        # decode latents to token logits
        if decoder_input_ids is None:
            # create simple BOS-seeded input for decoding
            seq_len = germline_input_ids.shape[1]
            decoder_input_ids = torch.full(
                (batch_size, seq_len),
                self.config.pad_token_id,
                device=device,
                dtype=torch.long,
            )
            decoder_input_ids[:, 0] = self.config.bos_token_id

        decoder_output = self.mutable.decode(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=x_1,
            return_dict=True,
        )

        logits = self.lm_head(decoder_output.last_hidden_state)
        return logits.argmax(dim=-1)
