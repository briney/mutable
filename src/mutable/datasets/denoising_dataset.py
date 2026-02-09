# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..modules.noise import BartNoiseFunction

__all__ = ["DenoisingDataset", "DenoisingCollator"]

# Gene segment character → integer encoding
_SEGMENT_MAP = {"V": 1, "D": 2, "J": 3, "N": 4}


def _parse_digit_mask(mask_str: str) -> list[int]:
    """Parse a digit string mask like ``"000111220003333"`` to a list of ints."""
    return [int(c) for c in mask_str]


def _parse_segment_mask(mask_str: str) -> list[int]:
    """Parse a gene segment string like ``"VVNDJ"`` to encoded ints.

    Mapping: V=1, D=2, J=3, N=4, anything else=0.
    """
    return [_SEGMENT_MAP.get(c, 0) for c in mask_str]


class DenoisingDataset(Dataset):
    """
    Dataset for Phase 1 denoising pre-training.

    Wraps an HF dataset, concatenates heavy + <sep> + light chains, tokenizes,
    and applies BART-style noise corruption.

    Supports two modes:
    - **Default mode** (``use_weighted_masking=False``): applies ``BartNoiseFunction``
      at the sample level (current behavior).
    - **Weighted masking mode** (``use_weighted_masking=True``): returns clean tokens
      plus parsed annotation masks; masking is deferred to the trainer at the batch
      level.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset with 'heavy' and 'light' columns (amino acid sequences).
    tokenizer : PreTrainedTokenizerFast
        Mutable tokenizer.
    noise_fn : BartNoiseFunction, optional
        Noise function for corruption. Required when ``use_weighted_masking=False``.
    max_length : int, default=512
        Maximum sequence length (including special tokens).
    heavy_col : str, default="heavy"
        Column name for heavy chain sequences.
    light_col : str, default="light"
        Column name for light chain sequences.
    use_weighted_masking : bool, default=False
        When True, return clean tokens + annotation masks instead of corrupted tokens.
    heavy_cdr_col : str, optional
        Column name for heavy chain CDR mask (digit string).
    light_cdr_col : str, optional
        Column name for light chain CDR mask (digit string).
    heavy_nongermline_col : str, optional
        Column name for heavy chain nongermline mask (digit string).
    light_nongermline_col : str, optional
        Column name for light chain nongermline mask (digit string).
    heavy_segment_col : str, optional
        Column name for heavy chain gene segment mask (char string).
    light_segment_col : str, optional
        Column name for light chain gene segment mask (char string).
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        noise_fn: Optional[BartNoiseFunction] = None,
        max_length: int = 512,
        heavy_col: str = "heavy",
        light_col: str = "light",
        use_weighted_masking: bool = False,
        heavy_cdr_col: Optional[str] = None,
        light_cdr_col: Optional[str] = None,
        heavy_nongermline_col: Optional[str] = None,
        light_nongermline_col: Optional[str] = None,
        heavy_segment_col: Optional[str] = None,
        light_segment_col: Optional[str] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.noise_fn = noise_fn
        self.max_length = max_length
        self.heavy_col = heavy_col
        self.light_col = light_col
        self.use_weighted_masking = use_weighted_masking

        if not use_weighted_masking and noise_fn is None:
            raise ValueError(
                "noise_fn is required when use_weighted_masking=False"
            )

        # Determine which mask columns are available in the dataset
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

    def _build_mask(
        self,
        row: dict,
        heavy_col: Optional[str],
        light_col: Optional[str],
        heavy_len: int,
        light_len: int,
        parser,
    ) -> torch.Tensor:
        """Build a concatenated mask: [0_bos] + heavy_mask + [0_sep] + light_mask + [0_eos].

        If a column is missing for a chain, zeros are used for that chain.
        """
        # Heavy chain mask
        if heavy_col is not None and row.get(heavy_col) is not None:
            heavy_mask = parser(str(row[heavy_col]))[:heavy_len]
            # Pad if mask is shorter than sequence
            if len(heavy_mask) < heavy_len:
                heavy_mask = heavy_mask + [0] * (heavy_len - len(heavy_mask))
        else:
            heavy_mask = [0] * heavy_len

        # Light chain mask
        if light_col is not None and row.get(light_col) is not None:
            light_mask = parser(str(row[light_col]))[:light_len]
            if len(light_mask) < light_len:
                light_mask = light_mask + [0] * (light_len - len(light_mask))
        else:
            light_mask = [0] * light_len

        # Wrap with zeros for BOS, SEP, EOS
        full_mask = [0] + heavy_mask + [0] + light_mask + [0]
        return torch.tensor(full_mask, dtype=torch.long)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.dataset[idx]

        # concatenate heavy + <sep> + light
        heavy = row[self.heavy_col]
        light = row[self.light_col]
        sequence = f"{heavy}<sep>{light}"

        # tokenize clean sequence (these are the labels)
        clean_encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        clean_ids = clean_encoding["input_ids"].squeeze(0)

        if not self.use_weighted_masking:
            # --- Default mode: apply BartNoiseFunction at sample level ---
            inner_ids = clean_ids[1:-1].tolist()
            corrupted_ids = self.noise_fn(inner_ids)

            bos = self.tokenizer.bos_token_id
            eos = self.tokenizer.eos_token_id
            corrupted_ids = [bos] + corrupted_ids + [eos]

            if len(corrupted_ids) > self.max_length:
                corrupted_ids = corrupted_ids[: self.max_length - 1] + [eos]

            return {
                "input_ids": torch.tensor(corrupted_ids, dtype=torch.long),
                "labels": clean_ids,
            }

        # --- Weighted masking mode: return clean tokens + masks ---
        heavy_len = len(heavy)
        light_len = len(light)

        # In weighted mode, input_ids are the CLEAN tokens (masking applied later)
        result = {
            "input_ids": clean_ids,
            "labels": clean_ids.clone(),
        }

        # Special tokens mask: True for BOS, SEP, EOS positions
        special_tokens_mask = torch.zeros(len(clean_ids), dtype=torch.bool)
        special_tokens_mask[0] = True  # BOS
        special_tokens_mask[-1] = True  # EOS
        # Find SEP position: BOS + heavy_len
        sep_pos = 1 + heavy_len
        if sep_pos < len(clean_ids):
            special_tokens_mask[sep_pos] = True
        result["special_tokens_mask"] = special_tokens_mask

        # Build annotation masks
        if self.has_cdr:
            result["cdr_mask"] = self._build_mask(
                row, self.heavy_cdr_col, self.light_cdr_col,
                heavy_len, light_len, _parse_digit_mask,
            )

        if self.has_nongermline:
            result["nongermline_mask"] = self._build_mask(
                row, self.heavy_nongermline_col, self.light_nongermline_col,
                heavy_len, light_len, _parse_digit_mask,
            )

        if self.has_segment:
            result["segment_mask"] = self._build_mask(
                row, self.heavy_segment_col, self.light_segment_col,
                heavy_len, light_len, _parse_segment_mask,
            )

        return result


class DenoisingCollator:
    """
    Data collator for denoising pre-training. Pads input_ids and labels to
    the maximum length in the batch.

    When ``use_weighted_masking=True``, also pads annotation mask tensors
    (``special_tokens_mask``, ``cdr_mask``, ``nongermline_mask``, ``segment_mask``).

    Parameters
    ----------
    pad_token_id : int
        Token ID for padding.
    label_pad_token_id : int, default=-100
        Label padding value (ignored in loss).
    use_weighted_masking : bool, default=False
        Whether to expect and pad annotation mask tensors.
    """

    def __init__(
        self,
        pad_token_id: int,
        label_pad_token_id: int = -100,
        use_weighted_masking: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.use_weighted_masking = use_weighted_masking

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # find max lengths
        max_input_len = max(f["input_ids"].shape[0] for f in features)
        max_label_len = max(f["labels"].shape[0] for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            # pad input
            input_len = f["input_ids"].shape[0]
            pad_len = max_input_len - input_len
            input_ids.append(
                torch.cat([
                    f["input_ids"],
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ])
            )
            attention_mask.append(
                torch.cat([
                    torch.ones(input_len, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )

            # pad labels
            label_len = f["labels"].shape[0]
            label_pad_len = max_label_len - label_len
            labels.append(
                torch.cat([
                    f["labels"],
                    torch.full(
                        (label_pad_len,), self.label_pad_token_id, dtype=torch.long
                    ),
                ])
            )

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

        if self.use_weighted_masking:
            # Pad annotation masks
            # special_tokens_mask: pad positions → True (not maskable)
            if "special_tokens_mask" in features[0]:
                stm = []
                for f in features:
                    seq_len = f["special_tokens_mask"].shape[0]
                    pad_len = max_input_len - seq_len
                    stm.append(
                        torch.cat([
                            f["special_tokens_mask"],
                            torch.ones(pad_len, dtype=torch.bool),
                        ])
                    )
                batch["special_tokens_mask"] = torch.stack(stm)

            # Integer annotation masks: pad positions → 0
            for key in ("cdr_mask", "nongermline_mask", "segment_mask"):
                if key in features[0]:
                    padded = []
                    for f in features:
                        seq_len = f[key].shape[0]
                        pad_len = max_input_len - seq_len
                        padded.append(
                            torch.cat([
                                f[key],
                                torch.zeros(pad_len, dtype=torch.long),
                            ])
                        )
                    batch[key] = torch.stack(padded)

        return batch
