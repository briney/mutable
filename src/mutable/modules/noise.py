# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import math
import random
from typing import List, Optional

import torch
import numpy as np

__all__ = [
    "BartNoiseFunction",
    "token_masking",
    "token_deletion",
    "text_infilling",
    "sentence_permutation",
]


class BartNoiseFunction:
    """
    BART-style noise function combining multiple corruption strategies.

    Parameters
    ----------
    mask_token_id : int
        Token ID to use for masking.
    pad_token_id : int
        Token ID for padding.
    vocab_size : int
        Size of the vocabulary (for random token replacement).
    mask_ratio : float, default=0.3
        Fraction of tokens to mask.
    delete_ratio : float, default=0.0
        Fraction of tokens to delete.
    infill_ratio : float, default=0.0
        Fraction of tokens to replace with infilling spans.
    poisson_lambda : float, default=3.0
        Lambda for Poisson span length distribution (text infilling).
    permute_sentences : bool, default=False
        Whether to permute sentences (separated by <sep>).
    random_token_ratio : float, default=0.0
        Fraction of masked tokens to replace with random tokens instead.
    sep_token_id : Optional[int], default=None
        Separator token ID for sentence permutation.
    """

    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        mask_ratio: float = 0.3,
        delete_ratio: float = 0.0,
        infill_ratio: float = 0.0,
        poisson_lambda: float = 3.0,
        permute_sentences: bool = False,
        random_token_ratio: float = 0.0,
        sep_token_id: Optional[int] = None,
    ):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.delete_ratio = delete_ratio
        self.infill_ratio = infill_ratio
        self.poisson_lambda = poisson_lambda
        self.permute_sentences = permute_sentences
        self.random_token_ratio = random_token_ratio
        self.sep_token_id = sep_token_id

    def __call__(self, token_ids: List[int]) -> List[int]:
        """
        Apply noise to a list of token IDs.

        Parameters
        ----------
        token_ids : List[int]
            Input token IDs (without BOS/EOS).

        Returns
        -------
        List[int]
            Corrupted token IDs.
        """
        result = list(token_ids)

        if self.permute_sentences and self.sep_token_id is not None:
            result = sentence_permutation(result, self.sep_token_id)

        if self.infill_ratio > 0:
            result = text_infilling(
                result,
                mask_token_id=self.mask_token_id,
                ratio=self.infill_ratio,
                poisson_lambda=self.poisson_lambda,
            )
        elif self.mask_ratio > 0:
            result = token_masking(
                result,
                mask_token_id=self.mask_token_id,
                ratio=self.mask_ratio,
                random_token_ratio=self.random_token_ratio,
                vocab_size=self.vocab_size,
            )

        if self.delete_ratio > 0:
            result = token_deletion(result, ratio=self.delete_ratio)

        return result


def token_masking(
    token_ids: List[int],
    mask_token_id: int,
    ratio: float = 0.3,
    random_token_ratio: float = 0.0,
    vocab_size: int = 32,
) -> List[int]:
    """
    Randomly replace tokens with mask token (or random tokens).

    Parameters
    ----------
    token_ids : List[int]
        Input token IDs.
    mask_token_id : int
        Token ID for masking.
    ratio : float
        Fraction of tokens to corrupt.
    random_token_ratio : float
        Of the corrupted tokens, fraction to replace with random tokens.
    vocab_size : int
        Vocabulary size for random token selection.
    """
    result = list(token_ids)
    n = len(result)
    num_to_mask = max(1, int(n * ratio))
    indices = random.sample(range(n), min(num_to_mask, n))

    for idx in indices:
        if random.random() < random_token_ratio:
            result[idx] = random.randint(4, vocab_size - 2)  # skip special tokens
        else:
            result[idx] = mask_token_id

    return result


def token_deletion(token_ids: List[int], ratio: float = 0.1) -> List[int]:
    """
    Randomly delete tokens.

    Parameters
    ----------
    token_ids : List[int]
        Input token IDs.
    ratio : float
        Fraction of tokens to delete.
    """
    result = [t for t in token_ids if random.random() >= ratio]
    if len(result) == 0:
        result = [token_ids[0]]  # keep at least one token
    return result


def text_infilling(
    token_ids: List[int],
    mask_token_id: int,
    ratio: float = 0.3,
    poisson_lambda: float = 3.0,
) -> List[int]:
    """
    Replace spans of tokens with a single mask token. Span lengths drawn from Poisson.

    Parameters
    ----------
    token_ids : List[int]
        Input token IDs.
    mask_token_id : int
        Token ID for masking.
    ratio : float
        Fraction of tokens to corrupt (by total span coverage).
    poisson_lambda : float
        Lambda for Poisson span length distribution.
    """
    n = len(token_ids)
    num_to_mask = max(1, int(n * ratio))

    # generate span lengths
    lengths = []
    total = 0
    while total < num_to_mask:
        length = np.random.poisson(poisson_lambda)
        length = max(1, length)
        lengths.append(length)
        total += length

    # randomly choose span start positions
    result = list(token_ids)
    masked = [False] * n

    for span_len in lengths:
        if all(masked):
            break
        tries = 0
        while tries < 10:
            start = random.randint(0, max(0, n - span_len))
            end = min(start + span_len, n)
            if not any(masked[start:end]):
                break
            tries += 1
        # replace span with single mask token
        for i in range(start, end):
            masked[i] = True

    # build result: replace masked spans with single mask tokens
    output = []
    i = 0
    while i < n:
        if masked[i]:
            output.append(mask_token_id)
            while i < n and masked[i]:
                i += 1
        else:
            output.append(result[i])
            i += 1

    return output


def sentence_permutation(token_ids: List[int], sep_token_id: int) -> List[int]:
    """
    Permute sentences separated by the separator token.

    Parameters
    ----------
    token_ids : List[int]
        Input token IDs.
    sep_token_id : int
        Separator token ID.
    """
    # split into sentences at separator tokens
    sentences = []
    current = []
    for t in token_ids:
        if t == sep_token_id:
            if current:
                sentences.append(current)
            sentences.append([sep_token_id])
            current = []
        else:
            current.append(t)
    if current:
        sentences.append(current)

    if len(sentences) <= 1:
        return token_ids

    # permute non-separator sentences
    non_sep = [s for s in sentences if s != [sep_token_id]]
    sep_positions = [i for i, s in enumerate(sentences) if s == [sep_token_id]]

    random.shuffle(non_sep)

    # reconstruct with separators in original positions
    result_sentences = []
    non_sep_iter = iter(non_sep)
    for i in range(len(sentences)):
        if i in sep_positions:
            result_sentences.append([sep_token_id])
        else:
            result_sentences.append(next(non_sep_iter))

    return [t for s in result_sentences for t in s]
