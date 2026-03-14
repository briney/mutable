# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

__all__ = ["FlowMatchingDataset", "FlowMatchingCollator"]

logger = logging.getLogger(__name__)


class FlowMatchingDataset(Dataset):
    """
    Dataset for Phase 2 flow matching training.

    Provides (germline, mutated) sequence pairs with computed mutation intensity.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset with germline and mutated sequence columns.
    tokenizer : PreTrainedTokenizerFast
        Mutable tokenizer.
    max_length : int, default=512
        Maximum sequence length.
    germline_heavy_col : str, default="germline_heavy"
        Column name for germline heavy chain.
    germline_light_col : str, default="germline_light"
        Column name for germline light chain.
    mutated_heavy_col : str, default="mutated_heavy"
        Column name for mutated heavy chain.
    mutated_light_col : str, default="mutated_light"
        Column name for mutated light chain.
    mu_col : str, optional
        Column name for precomputed mutation intensity.
        If None, computed from sequence differences.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 512,
        germline_heavy_col: str = "germline_heavy",
        germline_light_col: str = "germline_light",
        mutated_heavy_col: str = "mutated_heavy",
        mutated_light_col: str = "mutated_light",
        mu_col: Optional[str] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.germline_heavy_col = germline_heavy_col
        self.germline_light_col = germline_light_col
        self.mutated_heavy_col = mutated_heavy_col
        self.mutated_light_col = mutated_light_col

        # Validate required columns
        ds_columns = set(dataset.column_names)
        required = {
            germline_heavy_col: "germline_heavy_col",
            germline_light_col: "germline_light_col",
            mutated_heavy_col: "mutated_heavy_col",
            mutated_light_col: "mutated_light_col",
        }
        missing = [
            name for col, name in required.items() if col not in ds_columns
        ]
        if missing:
            raise ValueError(
                f"Required column(s) not found: {missing}. "
                f"Available: {sorted(ds_columns)}"
            )

        # Validate optional mu column
        if mu_col and mu_col not in ds_columns:
            logger.warning(
                "mu column '%s' not found in dataset; will compute from sequences.",
                mu_col,
            )
            mu_col = None
        self.mu_col = mu_col

    def __len__(self):
        return len(self.dataset)

    def _compute_mu(self, germline: str, mutated: str) -> float:
        """Compute mutation intensity as fraction of positions that differ."""
        if len(germline) == 0:
            return 0.0
        min_len = min(len(germline), len(mutated))
        diffs = sum(1 for a, b in zip(germline[:min_len], mutated[:min_len]) if a != b)
        # also count length difference as mutations
        diffs += abs(len(germline) - len(mutated))
        return diffs / max(len(germline), len(mutated))

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.dataset[idx]

        # build sequences
        germline_heavy = row[self.germline_heavy_col]
        germline_light = row[self.germline_light_col]
        mutated_heavy = row[self.mutated_heavy_col]
        mutated_light = row[self.mutated_light_col]

        germline_seq = f"{germline_heavy}<sep>{germline_light}"
        mutated_seq = f"{mutated_heavy}<sep>{mutated_light}"

        # tokenize (no padding — collator handles dynamic batching)
        germline_enc = self.tokenizer(
            germline_seq,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        mutated_enc = self.tokenizer(
            mutated_seq,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # mutation intensity
        if self.mu_col is not None and self.mu_col in row:
            mu = float(row[self.mu_col])
        else:
            mu = self._compute_mu(
                germline_heavy + germline_light,
                mutated_heavy + mutated_light,
            )

        return {
            "germline_input_ids": germline_enc["input_ids"].squeeze(0),
            "germline_attention_mask": germline_enc["attention_mask"].squeeze(0),
            "mutated_input_ids": mutated_enc["input_ids"].squeeze(0),
            "mutated_attention_mask": mutated_enc["attention_mask"].squeeze(0),
            "mu": torch.tensor(mu, dtype=torch.float32),
        }


class FlowMatchingCollator:
    """Data collator for flow matching training.

    Dynamically pads germline and mutated sequences to the maximum length
    in each batch, and stacks scalar ``mu`` values.

    Parameters
    ----------
    pad_token_id : int
        Token ID used for padding.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        max_germline_len = max(
            f["germline_input_ids"].shape[0] for f in features
        )
        max_mutated_len = max(
            f["mutated_input_ids"].shape[0] for f in features
        )

        germline_ids = []
        germline_mask = []
        mutated_ids = []
        mutated_mask = []
        mus = []

        for f in features:
            # pad germline
            g_len = f["germline_input_ids"].shape[0]
            g_pad = max_germline_len - g_len
            germline_ids.append(
                torch.cat([
                    f["germline_input_ids"],
                    torch.full((g_pad,), self.pad_token_id, dtype=torch.long),
                ])
            )
            germline_mask.append(
                torch.cat([
                    f["germline_attention_mask"],
                    torch.zeros(g_pad, dtype=torch.long),
                ])
            )

            # pad mutated
            m_len = f["mutated_input_ids"].shape[0]
            m_pad = max_mutated_len - m_len
            mutated_ids.append(
                torch.cat([
                    f["mutated_input_ids"],
                    torch.full((m_pad,), self.pad_token_id, dtype=torch.long),
                ])
            )
            mutated_mask.append(
                torch.cat([
                    f["mutated_attention_mask"],
                    torch.zeros(m_pad, dtype=torch.long),
                ])
            )

            mus.append(f["mu"])

        return {
            "germline_input_ids": torch.stack(germline_ids),
            "germline_attention_mask": torch.stack(germline_mask),
            "mutated_input_ids": torch.stack(mutated_ids),
            "mutated_attention_mask": torch.stack(mutated_mask),
            "mu": torch.stack(mus),
        }
