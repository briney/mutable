# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Precision@L contact prediction metric.

Adapted from libreplm's contact metric for paired antibody evaluation
with multiple contact modes (all, cross_chain, intra_heavy, intra_light,
cdr_contact).
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from ..base import MetricBase
from ..registry import register_metric
from ...outputs import DenoisingOutput

__all__ = ["PrecisionAtLMetric"]


def _compute_contact_map(coords: Tensor, threshold: float = 8.0) -> Tensor:
    """Compute binary contact map from CA atom coordinates.

    Parameters
    ----------
    coords
        Coordinates tensor (B, L, 3, 3) with N, CA, C atoms.
    threshold
        Distance threshold in angstroms.

    Returns
    -------
    Tensor
        Binary contact map (B, L, L).
    """
    ca = coords[:, :, 1, :]  # (B, L, 3) -- CA atoms
    diff = ca.unsqueeze(2) - ca.unsqueeze(1)  # (B, L, L, 3)
    dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)  # (B, L, L)
    contact_map = (dist < threshold) & ~torch.isnan(dist)
    return contact_map


def _apply_apc(matrix: Tensor) -> Tensor:
    """Apply Average Product Correction (APC).

    APC_ij = A_ij - (A_i_mean * A_j_mean) / A_global_mean
    """
    row_mean = matrix.mean(dim=-1, keepdim=True)
    col_mean = matrix.mean(dim=-2, keepdim=True)
    global_mean = matrix.mean(dim=(-1, -2), keepdim=True)
    correction = (row_mean * col_mean) / (global_mean + 1e-8)
    return matrix - correction


def _extract_attention_contacts(
    attentions: tuple[Tensor, ...],
    layer: int | str = "last",
    head_aggregation: str = "mean",
    num_layers: int = 1,
) -> Tensor | None:
    """Extract contact predictions from encoder attention weights.

    Parameters
    ----------
    attentions
        Tuple of per-layer attention tensors (B, H, L, L).
    layer
        Which layer: "last", "mean", or int index.
    head_aggregation
        How to aggregate heads: "mean" or "max".
    num_layers
        Number of final layers to average when layer="last".

    Returns
    -------
    Tensor
        Contact probability matrix (B, L, L), or None.
    """
    if attentions is None or len(attentions) == 0:
        return None

    if layer == "mean":
        attn = torch.stack(list(attentions), dim=0).mean(dim=0)
    elif isinstance(layer, int):
        attn = attentions[layer]
    elif layer == "last":
        n = min(num_layers, len(attentions))
        if n <= 1:
            attn = attentions[-1]
        else:
            attn = torch.stack(list(attentions[-n:]), dim=0).mean(dim=0)
    else:
        attn = attentions[-1]

    # Aggregate heads
    if head_aggregation == "max":
        contact_probs = attn.max(dim=1).values
    else:
        contact_probs = attn.mean(dim=1)

    # Symmetrize and APC
    contact_probs = (contact_probs + contact_probs.transpose(-1, -2)) / 2
    contact_probs = _apply_apc(contact_probs)

    return contact_probs


def _build_pair_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    special_tokens_mask: Tensor | None,
    min_seq_sep: int,
) -> Tensor:
    """Build valid residue pair mask for P@L.

    Parameters
    ----------
    input_ids
        Token IDs (B, L).
    attention_mask
        Attention mask (B, L).
    special_tokens_mask
        Special tokens mask (B, L), optional.
    min_seq_sep
        Minimum sequence separation.

    Returns
    -------
    Tensor
        Boolean pair mask (B, L, L).
    """
    B, L = input_ids.shape
    device = input_ids.device

    valid = attention_mask.bool()
    if special_tokens_mask is not None:
        valid = valid & ~special_tokens_mask.bool()

    pair_mask = valid.unsqueeze(-1) & valid.unsqueeze(-2)  # (B, L, L)

    # Sequence separation
    idx = torch.arange(L, device=device)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    pair_mask = pair_mask & (sep >= min_seq_sep).unsqueeze(0)

    # Exclude diagonal
    pair_mask = pair_mask & ~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)

    return pair_mask


def _filter_by_mode(
    pair_mask: Tensor,
    mode: str,
    chain_boundary: Tensor,
    cdr_mask: Tensor | None = None,
) -> Tensor:
    """Filter pair_mask by contact mode.

    Parameters
    ----------
    pair_mask
        Base pair mask (B, L, L).
    mode
        Contact mode: "all", "cross_chain", "intra_heavy", "intra_light", "cdr_contact".
    chain_boundary
        Position of <sep> token per sample (B,).
    cdr_mask
        CDR mask (B, L) if available.

    Returns
    -------
    Tensor
        Filtered pair mask (B, L, L).
    """
    if mode == "all":
        return pair_mask

    B, L, _ = pair_mask.shape
    device = pair_mask.device

    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
    boundary = chain_boundary.unsqueeze(-1)  # (B, 1)

    is_heavy = idx < boundary  # (B, L)
    is_light = idx > boundary  # (B, L)

    if mode == "cross_chain":
        cross = is_heavy.unsqueeze(-1) & is_light.unsqueeze(-2)
        cross = cross | (is_light.unsqueeze(-1) & is_heavy.unsqueeze(-2))
        return pair_mask & cross

    elif mode == "intra_heavy":
        intra_h = is_heavy.unsqueeze(-1) & is_heavy.unsqueeze(-2)
        return pair_mask & intra_h

    elif mode == "intra_light":
        intra_l = is_light.unsqueeze(-1) & is_light.unsqueeze(-2)
        return pair_mask & intra_l

    elif mode == "cdr_contact":
        if cdr_mask is None:
            return pair_mask
        is_cdr = cdr_mask > 0  # (B, L)
        cdr_pair = is_cdr.unsqueeze(-1) | is_cdr.unsqueeze(-2)
        return pair_mask & cdr_pair

    return pair_mask


@register_metric("p_at_l")
class PrecisionAtLMetric(MetricBase):
    """Precision@L metric for contact prediction from encoder attention.

    Computes the precision of the top-L predicted contacts using encoder
    attention weights, with support for multiple antibody-specific contact
    modes.

    Parameters
    ----------
    contact_threshold : float
        CA-CA distance threshold in angstroms for defining contacts.
    min_seq_sep : int
        Minimum sequence separation between residues.
    num_layers : int
        Number of final encoder layers to average attention from.
    head_aggregation : str
        How to aggregate attention heads ("mean" or "max").
    contact_mode : str or list[str]
        Contact filtering modes to compute. Options: "all", "cross_chain",
        "intra_heavy", "intra_light", "cdr_contact".
    """

    name: ClassVar[str] = "p_at_l"
    requires_coords: ClassVar[bool] = True
    needs_attentions: ClassVar[bool] = True

    def __init__(
        self,
        contact_threshold: float = 8.0,
        min_seq_sep: int = 6,
        num_layers: int | None = None,
        head_aggregation: str = "mean",
        contact_mode: str | list[str] = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contact_threshold = contact_threshold
        self.min_seq_sep = min_seq_sep
        self.num_layers = num_layers if num_layers is not None else 1
        self.head_aggregation = head_aggregation

        if isinstance(contact_mode, str):
            self.contact_modes = [contact_mode]
        else:
            self.contact_modes = list(contact_mode)

        # Per-mode accumulators: {mode: [correct_sum, total_sum]}
        self._accumulators: dict[str, list[float]] = {}
        self.reset()

    def update(
        self,
        outputs: DenoisingOutput,
        batch: dict[str, Tensor],
        mask_labels: Tensor,
    ) -> None:
        coords = batch.get("coords")
        if coords is None:
            return

        encoder_attentions = outputs.get("encoder_attentions")
        if encoder_attentions is None:
            return

        with torch.no_grad():
            # Compute true contact map
            true_contacts = _compute_contact_map(coords, self.contact_threshold)

            # Extract predicted contacts from attention
            pred_contacts = _extract_attention_contacts(
                encoder_attentions,
                layer="last",
                head_aggregation=self.head_aggregation,
                num_layers=self.num_layers,
            )
            if pred_contacts is None:
                return

            # Build base pair mask
            pair_mask = _build_pair_mask(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("special_tokens_mask"),
                self.min_seq_sep,
            )

            # Get chain boundary
            if "chain_boundary" in batch:
                chain_boundary = batch["chain_boundary"]
            else:
                # Derive from input_ids
                B, L = batch["input_ids"].shape
                device = batch["input_ids"].device
                chain_boundary = torch.zeros(B, dtype=torch.long, device=device)
                for b in range(B):
                    sep_pos = (batch["input_ids"][b] == 29).nonzero(as_tuple=True)[0]
                    if len(sep_pos) > 0:
                        chain_boundary[b] = sep_pos[0].item()

            cdr_mask = batch.get("cdr_mask")
            B = batch["input_ids"].shape[0]

            # Compute P@L for each mode
            for mode in self.contact_modes:
                mode_pair_mask = _filter_by_mode(
                    pair_mask, mode, chain_boundary, cdr_mask
                )

                for b in range(B):
                    upper_mask = torch.triu(mode_pair_mask[b], diagonal=1)
                    n_valid = upper_mask.sum().item()
                    if n_valid == 0:
                        continue

                    # Get valid position count (seq_len for L)
                    valid_positions = batch["attention_mask"][b].bool()
                    if batch.get("special_tokens_mask") is not None:
                        valid_positions = valid_positions & ~batch["special_tokens_mask"][b].bool()
                    seq_len = valid_positions.sum().item()

                    # Get predictions and labels for valid pairs
                    pred_flat = pred_contacts[b][upper_mask]
                    true_flat = true_contacts[b][upper_mask]

                    # Top-L (or fewer if not enough pairs)
                    k = min(seq_len, len(pred_flat))
                    if k <= 0:
                        continue

                    top_k_idx = pred_flat.argsort(descending=True)[:k]
                    correct = true_flat[top_k_idx].sum().item()

                    self._accumulators[mode][0] += correct
                    self._accumulators[mode][1] += k

    def compute(self) -> dict[str, float]:
        results = {}
        for mode in self.contact_modes:
            correct, total = self._accumulators.get(mode, [0.0, 0.0])
            precision = correct / total if total > 0 else 0.0

            if mode == "all":
                results["p_at_l"] = precision
            else:
                results[f"p_at_l/{mode}"] = precision

        return results

    def reset(self) -> None:
        self._accumulators = {mode: [0.0, 0.0] for mode in self.contact_modes}

    def state_tensors(self) -> list[Tensor]:
        tensors = []
        for mode in self.contact_modes:
            acc = self._accumulators.get(mode, [0.0, 0.0])
            tensors.append(torch.tensor(acc, dtype=torch.float64))
        return tensors

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        for i, mode in enumerate(self.contact_modes):
            if i < len(tensors):
                t = tensors[i]
                self._accumulators[mode] = [t[0].item(), t[1].item()]
