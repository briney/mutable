# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..modules.noise import BartNoiseFunction

__all__ = ["DenoisingDataset", "DenoisingCollator"]


class DenoisingDataset(Dataset):
    """
    Dataset for Phase 1 denoising pre-training.

    Wraps an HF dataset, concatenates heavy + <sep> + light chains, tokenizes,
    and applies BART-style noise corruption.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset with 'heavy' and 'light' columns (amino acid sequences).
    tokenizer : PreTrainedTokenizerFast
        Mutable tokenizer.
    noise_fn : BartNoiseFunction
        Noise function for corruption.
    max_length : int, default=512
        Maximum sequence length (including special tokens).
    heavy_col : str, default="heavy"
        Column name for heavy chain sequences.
    light_col : str, default="light"
        Column name for light chain sequences.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        noise_fn: BartNoiseFunction,
        max_length: int = 512,
        heavy_col: str = "heavy",
        light_col: str = "light",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.noise_fn = noise_fn
        self.max_length = max_length
        self.heavy_col = heavy_col
        self.light_col = light_col

    def __len__(self):
        return len(self.dataset)

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

        # get token IDs without BOS/EOS for noise application
        inner_ids = clean_ids[1:-1].tolist()

        # apply noise
        corrupted_ids = self.noise_fn(inner_ids)

        # re-add BOS/EOS
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        corrupted_ids = [bos] + corrupted_ids + [eos]

        # truncate if needed
        if len(corrupted_ids) > self.max_length:
            corrupted_ids = corrupted_ids[: self.max_length - 1] + [eos]

        return {
            "input_ids": torch.tensor(corrupted_ids, dtype=torch.long),
            "labels": clean_ids,
        }


class DenoisingCollator:
    """
    Data collator for denoising pre-training. Pads input_ids and labels to
    the maximum length in the batch.

    Parameters
    ----------
    pad_token_id : int
        Token ID for padding.
    label_pad_token_id : int, default=-100
        Label padding value (ignored in loss).
    """

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

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

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
