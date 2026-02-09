# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Paired-chain antibody PDB/CIF parser.

Adapted from libreplm's ``structure_parser.py`` for paired heavy/light
chain antibody structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = ["PairedStructureData", "parse_paired_structure", "AA3TO1"]


class PairedStructureData(NamedTuple):
    """Parsed paired antibody structure data.

    Attributes
    ----------
    pid : str
        Structure identifier (filename stem).
    heavy_sequence : str
        One-letter amino acid sequence for the heavy chain.
    light_sequence : str
        One-letter amino acid sequence for the light chain.
    heavy_coords : np.ndarray
        Backbone coordinates [Lh, 3, 3] for N, CA, C atoms.
    light_coords : np.ndarray
        Backbone coordinates [Ll, 3, 3] for N, CA, C atoms.
    heavy_chain_id : str
        Chain ID used for heavy chain extraction.
    light_chain_id : str
        Chain ID used for light chain extraction.
    """

    pid: str
    heavy_sequence: str
    light_sequence: str
    heavy_coords: np.ndarray
    light_coords: np.ndarray
    heavy_chain_id: str
    light_chain_id: str


# Standard 3-letter to 1-letter amino acid mapping
AA3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    # Non-standard / modified residues
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
    "HYP": "P",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    "CSO": "C",
    "CME": "C",
    "MLY": "K",
    "UNK": "X",
}


def _get_one_letter_code(res_name: str) -> str:
    """Convert 3-letter amino acid code to 1-letter code."""
    if res_name in AA3TO1:
        return AA3TO1[res_name]
    try:
        from Bio.PDB.Polypeptide import three_to_one

        return three_to_one(res_name)
    except (KeyError, ImportError):
        pass
    return "X"


def _extract_chain(chain) -> tuple[str, np.ndarray]:
    """Extract sequence and backbone coordinates from a Biopython chain.

    Returns
    -------
    tuple[str, np.ndarray]
        One-letter sequence and coordinates [L, 3, 3].
    """
    from Bio.PDB.Polypeptide import is_aa

    seq_chars: list[str] = []
    coords_list: list[list[list[float]]] = []

    for residue in chain:
        if not is_aa(residue, standard=False):
            continue

        res_name = residue.resname.upper().strip()
        aa = _get_one_letter_code(res_name)
        seq_chars.append(aa)

        try:
            n_coord = residue["N"].coord.tolist()
            ca_coord = residue["CA"].coord.tolist()
            c_coord = residue["C"].coord.tolist()
            coords_list.append([n_coord, ca_coord, c_coord])
        except KeyError:
            coords_list.append([[np.nan] * 3, [np.nan] * 3, [np.nan] * 3])

    return "".join(seq_chars), np.array(coords_list, dtype=np.float32)


def parse_paired_structure(
    path: str | Path,
    heavy_chain_id: str = "H",
    light_chain_id: str = "L",
) -> PairedStructureData:
    """Parse a PDB/mmCIF file and extract paired heavy/light chain data.

    Parameters
    ----------
    path
        Path to .pdb, .ent, .cif, or .mmcif file.
    heavy_chain_id
        Chain ID for the heavy chain (default: "H").
    light_chain_id
        Chain ID for the light chain (default: "L").

    Returns
    -------
    PairedStructureData
        Parsed paired structure data.

    Raises
    ------
    ValueError
        If structure cannot be parsed or chains are missing.
    FileNotFoundError
        If the file does not exist.
    """
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(path.stem, str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse structure file {path}: {e}") from e

    models = list(structure.get_models())
    if len(models) == 0:
        raise ValueError(f"No models found in {path}")

    model = models[0]

    # Find chains
    chains = {ch.id: ch for ch in model}
    heavy_chain = chains.get(heavy_chain_id)
    light_chain = chains.get(light_chain_id)

    # Fallback: if specified chains not found, try first two protein chains
    if heavy_chain is None or light_chain is None:
        protein_chains = [
            ch for ch in model
            if any(is_aa(r, standard=False) for r in ch)
        ]
        if len(protein_chains) < 2:
            raise ValueError(
                f"Could not find two protein chains in {path}. "
                f"Tried chain IDs '{heavy_chain_id}' and '{light_chain_id}', "
                f"found {len(protein_chains)} protein chain(s)."
            )
        heavy_chain = protein_chains[0]
        light_chain = protein_chains[1]
        heavy_chain_id = heavy_chain.id
        light_chain_id = light_chain.id

    heavy_seq, heavy_coords = _extract_chain(heavy_chain)
    light_seq, light_coords = _extract_chain(light_chain)

    if len(heavy_seq) == 0:
        raise ValueError(f"No amino acids extracted from heavy chain in {path}")
    if len(light_seq) == 0:
        raise ValueError(f"No amino acids extracted from light chain in {path}")

    return PairedStructureData(
        pid=path.stem,
        heavy_sequence=heavy_seq,
        light_sequence=light_seq,
        heavy_coords=heavy_coords,
        light_coords=light_coords,
        heavy_chain_id=heavy_chain_id,
        light_chain_id=light_chain_id,
    )
