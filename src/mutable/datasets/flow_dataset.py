# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

__all__ = ["FlowMatchingDataset"]


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

        # tokenize
        germline_enc = self.tokenizer(
            germline_seq,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        mutated_enc = self.tokenizer(
            mutated_seq,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
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
