# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .base_outputs import MutableModelOutput

__all__ = [
    "EncoderOutput",
    "BottleneckOutput",
    "DecoderOutput",
    "MutableModelFullOutput",
    "DenoisingOutput",
    "FlowMatchingOutput",
]


@dataclass
class EncoderOutput(MutableModelOutput):
    """
    Output from the MutableEncoder.

    Parameters
    ----------
    last_hidden_state : torch.FloatTensor
        Shape (batch_size, seq_len, hidden_size).
    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer hidden states.
    attentions : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer attention weights.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BottleneckOutput(MutableModelOutput):
    """
    Output from the PerceiverBottleneck.

    Parameters
    ----------
    latent_states : torch.FloatTensor
        Shape (batch_size, num_latents, latent_dim).
    attentions : Optional[torch.FloatTensor]
        Cross-attention weights from the bottleneck layer.
    """

    latent_states: torch.FloatTensor = None
    attentions: Optional[torch.FloatTensor] = None


@dataclass
class DecoderOutput(MutableModelOutput):
    """
    Output from the MutableDecoder.

    Parameters
    ----------
    last_hidden_state : torch.FloatTensor
        Shape (batch_size, seq_len, hidden_size).
    hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer hidden states.
    self_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer self-attention weights.
    cross_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer cross-attention weights.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    self_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MutableModelFullOutput(MutableModelOutput):
    """
    Full output from MutableModel (encoder + bottleneck + decoder).

    Parameters
    ----------
    last_hidden_state : torch.FloatTensor
        Decoder output, shape (batch_size, seq_len, hidden_size).
    latent_states : torch.FloatTensor
        Bottleneck latents, shape (batch_size, num_latents, latent_dim).
    encoder_last_hidden_state : torch.FloatTensor
        Encoder output, shape (batch_size, seq_len, hidden_size).
    encoder_hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer encoder hidden states.
    encoder_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer encoder attention weights.
    decoder_hidden_states : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer decoder hidden states.
    decoder_self_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer decoder self-attention weights.
    decoder_cross_attentions : Optional[Tuple[torch.FloatTensor, ...]]
        Per-layer decoder cross-attention weights.
    """

    last_hidden_state: torch.FloatTensor = None
    latent_states: torch.FloatTensor = None
    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_self_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DenoisingOutput(MutableModelOutput):
    """
    Output from MutableForDenoising (Phase 1).

    Parameters
    ----------
    loss : Optional[torch.FloatTensor]
        Cross-entropy loss.
    logits : torch.FloatTensor
        LM head logits, shape (batch_size, seq_len, vocab_size).
    latent_states : Optional[torch.FloatTensor]
        Bottleneck latents, shape (batch_size, num_latents, latent_dim).
    encoder_last_hidden_state : Optional[torch.FloatTensor]
        Encoder output.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    latent_states: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class FlowMatchingOutput(MutableModelOutput):
    """
    Output from MutableFlowMatching (Phase 2).

    Parameters
    ----------
    loss : Optional[torch.FloatTensor]
        OT-CFM MSE loss.
    predicted_velocity : Optional[torch.FloatTensor]
        Predicted velocity field, shape (batch_size, num_latents, latent_dim).
    target_velocity : Optional[torch.FloatTensor]
        Target velocity field, shape (batch_size, num_latents, latent_dim).
    latent_t : Optional[torch.FloatTensor]
        Interpolated latent at time t.
    """

    loss: Optional[torch.FloatTensor] = None
    predicted_velocity: Optional[torch.FloatTensor] = None
    target_velocity: Optional[torch.FloatTensor] = None
    latent_t: Optional[torch.FloatTensor] = None
