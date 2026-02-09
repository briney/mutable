# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for structure_parser.parse_paired_structure."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

Bio = pytest.importorskip("Bio")

from mutable.eval.structure_parser import parse_paired_structure


@pytest.fixture
def synthetic_pdb(tmp_path) -> Path:
    """Create a minimal synthetic PDB file with 2 chains (H and L)."""
    lines = []

    def _atom_line(serial, name, resName, chainID, resSeq, x, y, z):
        return (
            f"ATOM  {serial:5d} {name:<4s} {resName:3s} {chainID:1s}"
            f"{resSeq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
        )

    serial = 1
    # Heavy chain (H): 3 residues (ALA, GLY, VAL)
    residues_h = [("ALA", "A"), ("GLY", "G"), ("VAL", "V")]
    for i, (res3, _) in enumerate(residues_h):
        resSeq = i + 1
        x_base = float(i * 4)
        lines.append(_atom_line(serial, "N", res3, "H", resSeq, x_base, 0.0, 0.0))
        serial += 1
        lines.append(_atom_line(serial, "CA", res3, "H", resSeq, x_base + 1.0, 0.0, 0.0))
        serial += 1
        lines.append(_atom_line(serial, "C", res3, "H", resSeq, x_base + 2.0, 0.0, 0.0))
        serial += 1

    lines.append("TER")

    # Light chain (L): 2 residues (SER, THR)
    residues_l = [("SER", "S"), ("THR", "T")]
    for i, (res3, _) in enumerate(residues_l):
        resSeq = i + 1
        x_base = float(i * 4) + 20.0
        lines.append(_atom_line(serial, "N", res3, "L", resSeq, x_base, 0.0, 0.0))
        serial += 1
        lines.append(_atom_line(serial, "CA", res3, "L", resSeq, x_base + 1.0, 0.0, 0.0))
        serial += 1
        lines.append(_atom_line(serial, "C", res3, "L", resSeq, x_base + 2.0, 0.0, 0.0))
        serial += 1

    lines.append("TER")
    lines.append("END")

    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text("\n".join(lines) + "\n")
    return pdb_path


class TestParsePairedStructure:
    def test_parse_synthetic_pdb(self, synthetic_pdb):
        data = parse_paired_structure(synthetic_pdb)
        assert data.heavy_sequence == "AGV"
        assert data.light_sequence == "ST"
        assert data.heavy_coords.shape == (3, 3, 3)
        assert data.light_coords.shape == (2, 3, 3)

    def test_coord_atom_order(self, synthetic_pdb):
        """Verify coords[:, 0] is N, coords[:, 1] is CA, coords[:, 2] is C."""
        data = parse_paired_structure(synthetic_pdb)
        # First heavy residue (ALA): N at (0,0,0), CA at (1,0,0), C at (2,0,0)
        np.testing.assert_allclose(data.heavy_coords[0, 0, :], [0.0, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(data.heavy_coords[0, 1, :], [1.0, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(data.heavy_coords[0, 2, :], [2.0, 0.0, 0.0], atol=1e-3)


@pytest.fixture
def pdb_fallback_chains(tmp_path) -> Path:
    """PDB with chains A and B instead of H and L."""
    lines = []

    def _atom_line(serial, name, resName, chainID, resSeq, x, y, z):
        return (
            f"ATOM  {serial:5d} {name:<4s} {resName:3s} {chainID:1s}"
            f"{resSeq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
        )

    serial = 1
    for i, res3 in enumerate(["ALA", "GLY"]):
        resSeq = i + 1
        x = float(i * 4)
        for j, atom in enumerate(["N", "CA", "C"]):
            lines.append(_atom_line(serial, atom, res3, "A", resSeq, x + j, 0.0, 0.0))
            serial += 1
    lines.append("TER")

    for i, res3 in enumerate(["SER"]):
        resSeq = i + 1
        x = float(i * 4) + 20.0
        for j, atom in enumerate(["N", "CA", "C"]):
            lines.append(_atom_line(serial, atom, res3, "B", resSeq, x + j, 0.0, 0.0))
            serial += 1
    lines.append("TER")
    lines.append("END")

    pdb_path = tmp_path / "fallback.pdb"
    pdb_path.write_text("\n".join(lines) + "\n")
    return pdb_path


class TestFallbackChains:
    def test_fallback_chain_ids(self, pdb_fallback_chains):
        """PDB with chains A/B instead of H/L should fallback to first two protein chains."""
        data = parse_paired_structure(pdb_fallback_chains)
        assert data.heavy_sequence == "AG"
        assert data.light_sequence == "S"
        assert data.heavy_chain_id == "A"
        assert data.light_chain_id == "B"


@pytest.fixture
def pdb_missing_atoms(tmp_path) -> Path:
    """PDB where one residue is missing CA."""
    lines = []

    def _atom_line(serial, name, resName, chainID, resSeq, x, y, z):
        return (
            f"ATOM  {serial:5d} {name:<4s} {resName:3s} {chainID:1s}"
            f"{resSeq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
        )

    serial = 1
    # First residue: complete
    for j, atom in enumerate(["N", "CA", "C"]):
        lines.append(_atom_line(serial, atom, "ALA", "H", 1, float(j), 0.0, 0.0))
        serial += 1
    # Second residue: missing CA
    lines.append(_atom_line(serial, "N", "GLY", "H", 2, 4.0, 0.0, 0.0))
    serial += 1
    lines.append(_atom_line(serial, "C", "GLY", "H", 2, 6.0, 0.0, 0.0))
    serial += 1
    lines.append("TER")

    # Light chain: one complete residue
    for j, atom in enumerate(["N", "CA", "C"]):
        lines.append(_atom_line(serial, atom, "SER", "L", 1, 20.0 + j, 0.0, 0.0))
        serial += 1
    lines.append("TER")
    lines.append("END")

    pdb_path = tmp_path / "missing.pdb"
    pdb_path.write_text("\n".join(lines) + "\n")
    return pdb_path


class TestMissingAtoms:
    def test_missing_atoms_nan(self, pdb_missing_atoms):
        """Residue missing CA should have NaN coords for that atom."""
        data = parse_paired_structure(pdb_missing_atoms)
        assert data.heavy_sequence == "AG"
        # Second residue (GLY) is missing CA, so all coords should be NaN
        assert np.isnan(data.heavy_coords[1]).all()
