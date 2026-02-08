# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import MutableConfig
from ..modules import LearnedLatentEmbedding, PerceiverBottleneckLayer
from ..outputs import BottleneckOutput

__all__ = ["PerceiverBottleneck"]


class PerceiverBottleneck(nn.Module):
    """
    Perceiver bottleneck that compresses variable-length encoder output
    into fixed-size latent vectors via cross-attention.

    Parameters
    ----------
    config : MutableConfig
        Model configuration.
    """

    def __init__(self, config: MutableConfig):
        super().__init__()
        self.config = config

        # learned latent queries
        self.latent_embedding = LearnedLatentEmbedding(
            num_latents=config.num_latents,
            latent_dim=config.latent_dim,
        )

        # bottleneck cross-attention layer
        self.bottleneck_layer = PerceiverBottleneckLayer(config)

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BottleneckOutput, tuple]:
        """
        Parameters
        ----------
        encoder_output : torch.Tensor
            Shape (batch, seq_len, hidden_size).
        encoder_mask : Optional[torch.Tensor]
            Shape (batch, seq_len). True = valid token.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        batch_size = encoder_output.shape[0]
        latent_queries = self.latent_embedding(batch_size)

        result = self.bottleneck_layer(
            latent_queries=latent_queries,
            encoder_output=encoder_output,
            encoder_mask=encoder_mask,
            need_weights=need_weights,
        )

        if need_weights:
            latent_states, cross_weights = result
        else:
            latent_states = result
            cross_weights = None

        if not return_dict:
            return tuple(
                v for v in [latent_states, cross_weights] if v is not None
            )
        return BottleneckOutput(
            latent_states=latent_states,
            attentions=cross_weights,
        )
