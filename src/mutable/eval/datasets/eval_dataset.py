# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Eval dataset that extends DenoisingDataset with chain_ids for region eval."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from ...modules.noise import BartNoiseFunction

__all__ = ["EvalDenoisingDataset"]


class EvalDenoisingDataset(Dataset):
    """Dataset for evaluation that provides clean tokens plus chain_ids.

    Unlike the training ``DenoisingDataset``, this always returns clean tokens
    (masking is applied by the ``EvalMasker`` during evaluation). Adds
    ``chain_ids`` to the output for region-based evaluation.

    Parameters
    ----------
    dataset
        HuggingFace dataset with heavy/light columns.
    tokenizer
        Mutable tokenizer.
    max_length : int
        Maximum sequence length.
    heavy_col : str
        Column name for heavy chain sequences.
    light_col : str
        Column name for light chain sequences.
    heavy_cdr_col : str, optional
        Column for heavy CDR mask.
    light_cdr_col : str, optional
        Column for light CDR mask.
    heavy_nongermline_col : str, optional
        Column for heavy nongermline mask.
    light_nongermline_col : str, optional
        Column for light nongermline mask.
    heavy_segment_col : str, optional
        Column for heavy gene segment mask.
    light_segment_col : str, optional
        Column for light gene segment mask.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 512,
        heavy_col: str = "heavy",
        light_col: str = "light",
        heavy_cdr_col: Optional[str] = None,
        light_cdr_col: Optional[str] = None,
        heavy_nongermline_col: Optional[str] = None,
        light_nongermline_col: Optional[str] = None,
        heavy_segment_col: Optional[str] = None,
        light_segment_col: Optional[str] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.heavy_col = heavy_col
        self.light_col = light_col
        self.has_coords = False

        # Check which columns exist
        ds_columns = set(dataset.column_names)
        self.heavy_cdr_col = heavy_cdr_col if heavy_cdr_col and heavy_cdr_col in ds_columns else None
        self.light_cdr_col = light_cdr_col if light_cdr_col and light_cdr_col in ds_columns else None
        self.heavy_nongermline_col = heavy_nongermline_col if heavy_nongermline_col and heavy_nongermline_col in ds_columns else None
        self.light_nongermline_col = light_nongermline_col if light_nongermline_col and light_nongermline_col in ds_columns else None
        self.heavy_segment_col = heavy_segment_col if heavy_segment_col and heavy_segment_col in ds_columns else None
        self.light_segment_col = light_segment_col if light_segment_col and light_segment_col in ds_columns else None

        self.has_cdr = self.heavy_cdr_col is not None or self.light_cdr_col is not None
        self.has_nongermline = self.heavy_nongermline_col is not None or self.light_nongermline_col is not None
        self.has_segment = self.heavy_segment_col is not None or self.light_segment_col is not None

    def __len__(self):
        return len(self.dataset)

    def _parse_digit_mask(self, mask_str: str) -> list[int]:
        return [int(c) for c in mask_str]

    def _parse_segment_mask(self, mask_str: str) -> list[int]:
        _SEGMENT_MAP = {"V": 1, "D": 2, "J": 3, "N": 4}
        return [_SEGMENT_MAP.get(c, 0) for c in mask_str]

    def _build_mask(
        self,
        row: dict,
        heavy_col: Optional[str],
        light_col: Optional[str],
        heavy_len: int,
        light_len: int,
        parser,
    ) -> torch.Tensor:
        """Build concatenated mask: [0_bos] + heavy + [0_sep] + light + [0_eos]."""
        if heavy_col is not None and row.get(heavy_col) is not None:
            heavy_mask = parser(str(row[heavy_col]))[:heavy_len]
            if len(heavy_mask) < heavy_len:
                heavy_mask = heavy_mask + [0] * (heavy_len - len(heavy_mask))
        else:
            heavy_mask = [0] * heavy_len

        if light_col is not None and row.get(light_col) is not None:
            light_mask = parser(str(row[light_col]))[:light_len]
            if len(light_mask) < light_len:
                light_mask = light_mask + [0] * (light_len - len(light_mask))
        else:
            light_mask = [0] * light_len

        full_mask = [0] + heavy_mask + [0] + light_mask + [0]
        return torch.tensor(full_mask, dtype=torch.long)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.dataset[idx]

        heavy = row[self.heavy_col]
        light = row[self.light_col]
        sequence = f"{heavy}<sep>{light}"

        clean_encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        clean_ids = clean_encoding["input_ids"].squeeze(0)

        heavy_len = len(heavy)
        light_len = len(light)

        result = {
            "input_ids": clean_ids,
            "labels": clean_ids.clone(),
        }

        # Special tokens mask: True for BOS, SEP, EOS
        special_tokens_mask = torch.zeros(len(clean_ids), dtype=torch.bool)
        special_tokens_mask[0] = True  # BOS
        special_tokens_mask[-1] = True  # EOS
        sep_pos = 1 + heavy_len
        if sep_pos < len(clean_ids):
            special_tokens_mask[sep_pos] = True
        result["special_tokens_mask"] = special_tokens_mask

        # Chain IDs: 0 for heavy chain + specials before sep, 1 for light chain + specials after
        chain_ids = torch.zeros(len(clean_ids), dtype=torch.long)
        chain_ids[sep_pos + 1:] = 1
        result["chain_ids"] = chain_ids

        # Annotation masks
        if self.has_cdr:
            result["cdr_mask"] = self._build_mask(
                row, self.heavy_cdr_col, self.light_cdr_col,
                heavy_len, light_len, self._parse_digit_mask,
            )

        if self.has_nongermline:
            result["nongermline_mask"] = self._build_mask(
                row, self.heavy_nongermline_col, self.light_nongermline_col,
                heavy_len, light_len, self._parse_digit_mask,
            )

        if self.has_segment:
            result["segment_mask"] = self._build_mask(
                row, self.heavy_segment_col, self.light_segment_col,
                heavy_len, light_len, self._parse_segment_mask,
            )

        return result
