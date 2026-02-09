# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for AntibodyStructureDataset and StructureCollator."""

from pathlib import Path

import numpy as np
import torch
import pytest

Bio = pytest.importorskip("Bio")

from mutable.eval.datasets.structure_dataset import (
    AntibodyStructureDataset,
    StructureCollator,
)


def _write_pdb(tmp_path: Path, name: str, heavy_len: int, light_len: int) -> Path:
    """Write a minimal PDB file with given chain lengths."""
    lines = []
    serial = 1

    residues = ["ALA", "GLY", "VAL", "SER", "THR", "LEU", "ILE", "PRO"]

    for i in range(heavy_len):
        res = residues[i % len(residues)]
        x = float(i * 4)
        for j, atom in enumerate(["N", "CA", "C"]):
            lines.append(
                f"ATOM  {serial:5d} {atom:<4s} {res:3s} H"
                f"{i+1:4d}    {x + j:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("TER")

    for i in range(light_len):
        res = residues[i % len(residues)]
        x = float(i * 4) + 100.0
        for j, atom in enumerate(["N", "CA", "C"]):
            lines.append(
                f"ATOM  {serial:5d} {atom:<4s} {res:3s} L"
                f"{i+1:4d}    {x + j:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")

    pdb_path = tmp_path / name
    pdb_path.write_text("\n".join(lines) + "\n")
    return pdb_path


@pytest.fixture
def structure_dir(tmp_path):
    """Directory with 3 PDB files of varying lengths."""
    _write_pdb(tmp_path, "ab1.pdb", heavy_len=5, light_len=4)
    _write_pdb(tmp_path, "ab2.pdb", heavy_len=7, light_len=6)
    _write_pdb(tmp_path, "ab3.pdb", heavy_len=4, light_len=3)
    return tmp_path


class TestAntibodyStructureDataset:
    def test_dataset_output_keys(self, structure_dir, tokenizer):
        ds = AntibodyStructureDataset(structure_dir, tokenizer)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert "special_tokens_mask" in item
        assert "coords" in item
        assert "chain_boundary" in item

    def test_coords_shape(self, structure_dir, tokenizer):
        ds = AntibodyStructureDataset(structure_dir, tokenizer)
        item = ds[0]
        seq_len = item["input_ids"].shape[0]
        assert item["coords"].shape == (seq_len, 3, 3)

    def test_coords_nan_at_special_tokens(self, structure_dir, tokenizer):
        ds = AntibodyStructureDataset(structure_dir, tokenizer)
        item = ds[0]
        coords = item["coords"]
        # BOS
        assert torch.isnan(coords[0]).all()
        # EOS
        assert torch.isnan(coords[-1]).all()
        # SEP
        sep_pos = item["chain_boundary"].item()
        assert torch.isnan(coords[sep_pos]).all()

    def test_chain_boundary_at_sep(self, structure_dir, tokenizer):
        ds = AntibodyStructureDataset(structure_dir, tokenizer)
        item = ds[0]
        sep_pos = item["chain_boundary"].item()
        assert item["input_ids"][sep_pos].item() == 29  # SEP token


class TestStructureCollator:
    def test_collator_padding(self, structure_dir, tokenizer):
        ds = AntibodyStructureDataset(structure_dir, tokenizer)
        collator = StructureCollator(pad_token_id=1, label_pad_token_id=-100)

        batch = collator([ds[0], ds[1]])
        assert batch["input_ids"].shape[0] == 2
        # Both should be padded to same length
        assert batch["input_ids"].shape[1] == batch["coords"].shape[1]

    def test_collator_coords_nan_padding(self, structure_dir, tokenizer):
        ds = AntibodyStructureDataset(structure_dir, tokenizer)
        collator = StructureCollator()

        items = [ds[0], ds[1]]
        batch = collator(items)

        # Find shorter sequence
        len0 = items[0]["input_ids"].shape[0]
        len1 = items[1]["input_ids"].shape[0]
        if len0 < len1:
            shorter_idx, shorter_len = 0, len0
        else:
            shorter_idx, shorter_len = 1, len1

        max_len = batch["input_ids"].shape[1]
        if shorter_len < max_len:
            # Padded positions should have NaN coords
            assert torch.isnan(batch["coords"][shorter_idx, shorter_len:]).all()
