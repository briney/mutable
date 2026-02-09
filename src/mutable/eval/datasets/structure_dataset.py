# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Structure dataset for antibody PDB evaluation (P@L metrics)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..structure_parser import parse_paired_structure

__all__ = ["AntibodyStructureDataset", "StructureCollator"]


class AntibodyStructureDataset(Dataset):
    """Dataset for paired antibody structure evaluation.

    Parses PDB/CIF files, tokenizes paired heavy+light sequences, and
    provides coordinate data for contact prediction metrics.

    Parameters
    ----------
    structure_dir
        Path to directory containing PDB/CIF structure files.
    tokenizer
        Mutable tokenizer.
    max_length : int
        Maximum sequence length.
    heavy_chain_id : str
        Chain ID for heavy chain in PDB files.
    light_chain_id : str
        Chain ID for light chain in PDB files.
    """

    def __init__(
        self,
        structure_dir: str | Path,
        tokenizer,
        max_length: int = 512,
        heavy_chain_id: str = "H",
        light_chain_id: str = "L",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.heavy_chain_id = heavy_chain_id
        self.light_chain_id = light_chain_id
        self.has_coords = True

        structure_dir = Path(structure_dir)
        if not structure_dir.exists():
            raise FileNotFoundError(f"Structure directory not found: {structure_dir}")

        # Find all PDB/CIF files
        extensions = {".pdb", ".ent", ".cif", ".mmcif"}
        self.files = sorted([
            f for f in structure_dir.iterdir()
            if f.suffix.lower() in extensions
        ])

        if len(self.files) == 0:
            raise ValueError(f"No structure files found in {structure_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        path = self.files[idx]

        data = parse_paired_structure(
            path,
            heavy_chain_id=self.heavy_chain_id,
            light_chain_id=self.light_chain_id,
        )

        # Tokenize: "{heavy}<sep>{light}"
        sequence = f"{data.heavy_sequence}<sep>{data.light_sequence}"
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)

        heavy_len = len(data.heavy_sequence)
        light_len = len(data.light_sequence)

        # Special tokens mask
        special_tokens_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        special_tokens_mask[0] = True  # BOS
        special_tokens_mask[-1] = True  # EOS
        sep_pos = 1 + heavy_len
        if sep_pos < len(input_ids):
            special_tokens_mask[sep_pos] = True

        # Build coords: [nan] + heavy_coords + [nan] + light_coords + [nan]
        nan_atom = np.full((1, 3, 3), np.nan, dtype=np.float32)
        coords = np.concatenate([
            nan_atom,                        # BOS
            data.heavy_coords[:heavy_len],   # heavy chain
            nan_atom,                        # SEP
            data.light_coords[:light_len],   # light chain
            nan_atom,                        # EOS
        ], axis=0)

        # Truncate coords to match token length
        coords = coords[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "special_tokens_mask": special_tokens_mask,
            "coords": torch.from_numpy(coords),
            "chain_boundary": torch.tensor(sep_pos, dtype=torch.long),
        }


class StructureCollator:
    """Collator for AntibodyStructureDataset.

    Pads input_ids, labels, special_tokens_mask, and coords to batch max length.
    Coordinates are padded with NaN.

    Parameters
    ----------
    pad_token_id : int
        Token ID for padding.
    label_pad_token_id : int
        Label padding value.
    """

    def __init__(
        self,
        pad_token_id: int = 1,
        label_pad_token_id: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids = []
        attention_mask = []
        labels = []
        special_tokens_mask = []
        coords = []
        chain_boundaries = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids.append(torch.cat([
                f["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ]))

            attention_mask.append(torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]))

            labels.append(torch.cat([
                f["labels"],
                torch.full((pad_len,), self.label_pad_token_id, dtype=torch.long),
            ]))

            special_tokens_mask.append(torch.cat([
                f["special_tokens_mask"],
                torch.ones(pad_len, dtype=torch.bool),
            ]))

            # Pad coords with NaN
            if pad_len > 0:
                nan_pad = torch.full((pad_len, 3, 3), float("nan"))
                coords.append(torch.cat([f["coords"], nan_pad], dim=0))
            else:
                coords.append(f["coords"])

            chain_boundaries.append(f["chain_boundary"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "special_tokens_mask": torch.stack(special_tokens_mask),
            "coords": torch.stack(coords),
            "chain_boundary": torch.stack(chain_boundaries),
        }
