# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import List, Tuple

__all__ = ["compute_mutation_positions"]


def compute_mutation_positions(
    germline: str,
    mutated: str,
) -> List[int]:
    """
    Compute positions where the mutated sequence differs from germline.

    Parameters
    ----------
    germline : str
        Germline amino acid sequence.
    mutated : str
        Mutated amino acid sequence.

    Returns
    -------
    List[int]
        List of 0-indexed positions where mutations occurred.
    """
    positions = []
    for i, (g, m) in enumerate(zip(germline, mutated)):
        if g != m:
            positions.append(i)
    return positions
