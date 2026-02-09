# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Masking strategies for denoising pre-training.

Provides information-weighted and uniform masking at the batch level,
enabling efficient vectorized position selection via Gumbel-top-k.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["InformationWeightedMasker", "UniformMasker"]


class InformationWeightedMasker:
    """
    Applies masking with preference for high-information positions.

    CDR, nongermline, and gene segment positions receive additive weight bonuses
    controlled by separate multipliers. The base weight for every position is 1.0:

        weight = 1.0 + cdr_bonus + nongermline_bonus + segment_bonus

    With default CDR and nongermline multipliers (1.0 each) and segment multipliers
    at 0.0:
    - Nongermline CDR positions: 3x base weight (1 + 1 + 1)
    - Germline CDR or nongermline framework: 2x base weight
    - Germline framework: 1x base weight

    Two selection methods are available:
    - "ranked": Deterministically masks the top-K highest-weighted positions
    - "sampled": Probabilistically samples positions using Gumbel-top-k,
                 where higher weights increase selection probability but
                 don't guarantee selection

    Parameters
    ----------
    mask_rate : float
        Fraction of maskable tokens to mask.
    mask_token_id : int
        Token ID used for masking.
    cdr_weight_multiplier : float
        Additive bonus for CDR positions (CDR1/2/3).
    nongermline_weight_multiplier : float
        Additive bonus for nongermline (somatically mutated) positions.
    v_weight_multiplier : float
        Additive bonus for V gene segment positions.
    d_weight_multiplier : float
        Additive bonus for D gene segment positions.
    j_weight_multiplier : float
        Additive bonus for J gene segment positions.
    n_weight_multiplier : float
        Additive bonus for N-addition positions.
    selection_method : str
        Either "ranked" (deterministic) or "sampled" (Gumbel-top-k).
    """

    def __init__(
        self,
        mask_rate: float = 0.3,
        mask_token_id: int = 31,
        cdr_weight_multiplier: float = 1.0,
        nongermline_weight_multiplier: float = 1.0,
        v_weight_multiplier: float = 0.0,
        d_weight_multiplier: float = 0.0,
        j_weight_multiplier: float = 0.0,
        n_weight_multiplier: float = 0.0,
        selection_method: str = "sampled",
    ) -> None:
        if not 0.0 < mask_rate < 1.0:
            raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")
        if selection_method not in ("ranked", "sampled"):
            raise ValueError(
                f"selection_method must be 'ranked' or 'sampled', got {selection_method}"
            )
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id
        self.cdr_weight_multiplier = cdr_weight_multiplier
        self.nongermline_weight_multiplier = nongermline_weight_multiplier
        self.v_weight_multiplier = v_weight_multiplier
        self.d_weight_multiplier = d_weight_multiplier
        self.j_weight_multiplier = j_weight_multiplier
        self.n_weight_multiplier = n_weight_multiplier
        self.selection_method = selection_method

    def compute_weights(
        self,
        maskable_positions: Tensor,
        cdr_mask: Tensor | None = None,
        nongermline_mask: Tensor | None = None,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute per-position masking weights.

        Parameters
        ----------
        maskable_positions : Tensor
            Boolean mask of shape (batch, seq_len) indicating maskable positions.
        cdr_mask : Tensor, optional
            Integer tensor (batch, seq_len). 0=FW, 1=CDR1, 2=CDR2, 3=CDR3.
        nongermline_mask : Tensor, optional
            Integer tensor (batch, seq_len). 0=germline, 1=nongermline.
        segment_mask : Tensor, optional
            Integer tensor (batch, seq_len). 0=other, 1=V, 2=D, 3=J, 4=N.

        Returns
        -------
        Tensor
            Normalized weights of shape (batch, seq_len).
        """
        batch_size, seq_len = maskable_positions.shape
        device = maskable_positions.device

        weights = torch.ones(batch_size, seq_len, device=device)

        # CDR bonus: any CDR position (1, 2, or 3) gets the bonus
        if cdr_mask is not None:
            cdr_binary = (cdr_mask > 0).float()
            weights = weights + cdr_binary * self.cdr_weight_multiplier

        # Nongermline bonus
        if nongermline_mask is not None:
            weights = weights + nongermline_mask.float() * self.nongermline_weight_multiplier

        # Gene segment bonuses (per-segment multipliers)
        if segment_mask is not None:
            if self.v_weight_multiplier != 0.0:
                weights = weights + (segment_mask == 1).float() * self.v_weight_multiplier
            if self.d_weight_multiplier != 0.0:
                weights = weights + (segment_mask == 2).float() * self.d_weight_multiplier
            if self.j_weight_multiplier != 0.0:
                weights = weights + (segment_mask == 3).float() * self.j_weight_multiplier
            if self.n_weight_multiplier != 0.0:
                weights = weights + (segment_mask == 4).float() * self.n_weight_multiplier

        # Zero out non-maskable positions and normalize
        weights = weights * maskable_positions.float()
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return weights / weights_sum

    def apply_mask(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        cdr_mask: Tensor | None = None,
        nongermline_mask: Tensor | None = None,
        segment_mask: Tensor | None = None,
        special_tokens_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply weighted masking to a batch of token IDs.

        Parameters
        ----------
        token_ids : Tensor
            Token IDs of shape (batch, seq_len).
        attention_mask : Tensor
            Attention mask of shape (batch, seq_len).
        cdr_mask : Tensor, optional
            CDR region mask (batch, seq_len).
        nongermline_mask : Tensor, optional
            Nongermline mask (batch, seq_len).
        segment_mask : Tensor, optional
            Gene segment mask (batch, seq_len).
        special_tokens_mask : Tensor, optional
            Boolean mask for special tokens (BOS, EOS, SEP, PAD) that should
            never be masked.
        generator : torch.Generator, optional
            RNG for reproducibility.

        Returns
        -------
        masked_ids : Tensor
            Token IDs with masked positions replaced by mask_token_id.
        mask_labels : Tensor
            Boolean mask indicating which positions were masked.
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Determine maskable positions: attended and not special tokens
        maskable_positions = attention_mask.bool().clone()
        if special_tokens_mask is not None:
            maskable_positions = maskable_positions & ~special_tokens_mask.bool()

        # Count maskable positions per sample and compute number to mask
        valid_counts = maskable_positions.sum(dim=-1)
        num_to_mask = (valid_counts.float() * self.mask_rate).round().long().clamp(min=0)

        # Compute position weights
        weights = self.compute_weights(
            maskable_positions, cdr_mask, nongermline_mask, segment_mask
        )

        # Compute scores for position selection
        if self.selection_method == "ranked":
            # Deterministic top-K (tiny noise for tie-breaking only)
            noise = torch.rand(weights.shape, device=device, generator=generator) * 1e-6
            scores = weights + noise
        else:
            # Gumbel-top-k: weighted probabilistic sampling without replacement
            # See: https://arxiv.org/abs/1903.06059
            eps = 1e-10
            uniform = torch.rand(weights.shape, device=device, generator=generator)
            uniform = uniform.clamp(min=eps, max=1 - eps)
            gumbel_noise = -torch.log(-torch.log(uniform))
            scores = torch.log(weights + eps) + gumbel_noise

        # Exclude non-maskable positions
        scores = scores.masked_fill(~maskable_positions, float("-inf"))

        # Sort by score descending and find each position's rank
        _, indices = scores.sort(dim=-1, descending=True)
        position_ranks = torch.zeros_like(indices)
        position_ranks.scatter_(
            dim=-1,
            index=indices,
            src=torch.arange(seq_len, device=device).expand(batch_size, -1),
        )

        # Mask positions whose rank < num_to_mask
        mask_labels = position_ranks < num_to_mask.unsqueeze(-1)
        mask_labels = mask_labels & maskable_positions

        # Apply masking
        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = self.mask_token_id

        return masked_ids, mask_labels


class UniformMasker:
    """Simple uniform random masking without information weighting.

    Parameters
    ----------
    mask_rate : float
        Fraction of maskable tokens to mask.
    mask_token_id : int
        Token ID used for masking.
    """

    def __init__(
        self,
        mask_rate: float = 0.3,
        mask_token_id: int = 31,
    ) -> None:
        if not 0.0 < mask_rate < 1.0:
            raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id

    def apply_mask(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        special_tokens_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply uniform random masking to a batch of token IDs.

        Parameters
        ----------
        token_ids : Tensor
            Token IDs of shape (batch, seq_len).
        attention_mask : Tensor
            Attention mask of shape (batch, seq_len).
        special_tokens_mask : Tensor, optional
            Boolean mask for special tokens that should never be masked.
        generator : torch.Generator, optional
            RNG for reproducibility.

        Returns
        -------
        masked_ids : Tensor
            Token IDs with masked positions replaced by mask_token_id.
        mask_labels : Tensor
            Boolean mask indicating which positions were masked.
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        rand = torch.rand(batch_size, seq_len, device=device, generator=generator)

        maskable = attention_mask.bool()
        if special_tokens_mask is not None:
            maskable = maskable & ~special_tokens_mask.bool()

        mask_labels = (rand < self.mask_rate) & maskable

        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = self.mask_token_id

        return masked_ids, mask_labels
