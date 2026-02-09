# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Shared test fixtures for the mutable eval harness test suite."""

from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from mutable.config import MutableConfig
from mutable.models.mutable_denoising import MutableForDenoising
from mutable.outputs import DenoisingOutput
from mutable.tokenizer import MutableTokenizer


@pytest.fixture(scope="session")
def small_config():
    """Tiny MutableConfig for fast tests."""
    return MutableConfig(
        hidden_size=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_attention_heads=4,
        num_latents=8,
        intermediate_size=128,
        vocab_size=32,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


@pytest.fixture(scope="session")
def small_model(small_config):
    """Tiny MutableForDenoising in eval mode."""
    torch.manual_seed(42)
    model = MutableForDenoising(small_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tokenizer():
    """MutableTokenizer instance."""
    return MutableTokenizer()


@pytest.fixture(scope="session")
def device():
    """CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_batch():
    """Synthetic batch with paired H/L sequences.

    Format: [BOS] heavy(15) [SEP] light(14) [EOS] = 32 tokens
    Batch size: 4
    """
    torch.manual_seed(42)
    B, L = 4, 32
    sep_pos = 16  # BOS(1) + heavy(15) = position 16

    input_ids = torch.randint(4, 29, (B, L))  # amino acid range
    input_ids[:, 0] = 0   # BOS
    input_ids[:, sep_pos] = 29  # SEP
    input_ids[:, -1] = 2  # EOS

    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels = input_ids.clone()

    special_tokens_mask = torch.zeros(B, L, dtype=torch.bool)
    special_tokens_mask[:, 0] = True
    special_tokens_mask[:, sep_pos] = True
    special_tokens_mask[:, -1] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "special_tokens_mask": special_tokens_mask,
    }


@pytest.fixture
def sample_batch_with_annotations(sample_batch):
    """Extends sample_batch with CDR, nongermline, and chain_ids masks."""
    batch = dict(sample_batch)
    B, L = batch["input_ids"].shape
    sep_pos = 16

    # chain_ids: 0 before sep, 1 after
    chain_ids = torch.zeros(B, L, dtype=torch.long)
    chain_ids[:, sep_pos + 1:] = 1
    batch["chain_ids"] = chain_ids

    # cdr_mask: 0=FW, 1=CDR1, 2=CDR2, 3=CDR3
    # Heavy: CDR1 at 3-5, CDR2 at 8-10, CDR3 at 12-14
    # Light: CDR1 at 19-21, CDR2 at 24-26, CDR3 at 28-30
    cdr_mask = torch.zeros(B, L, dtype=torch.long)
    cdr_mask[:, 3:6] = 1    # HCDR1
    cdr_mask[:, 8:11] = 2   # HCDR2
    cdr_mask[:, 12:15] = 3  # HCDR3
    cdr_mask[:, 19:22] = 1  # LCDR1
    cdr_mask[:, 24:27] = 2  # LCDR2
    cdr_mask[:, 28:31] = 3  # LCDR3
    # Zero out special tokens
    cdr_mask[:, 0] = 0
    cdr_mask[:, sep_pos] = 0
    cdr_mask[:, -1] = 0
    batch["cdr_mask"] = cdr_mask

    # nongermline_mask: random binary
    nongermline_mask = torch.zeros(B, L, dtype=torch.long)
    nongermline_mask[:, 5] = 1
    nongermline_mask[:, 13] = 1
    nongermline_mask[:, 14] = 1
    nongermline_mask[:, 29] = 1
    batch["nongermline_mask"] = nongermline_mask

    return batch


@pytest.fixture
def sample_batch_with_coords(sample_batch):
    """Extends sample_batch with coords and chain_boundary."""
    batch = dict(sample_batch)
    B, L = batch["input_ids"].shape
    sep_pos = 16

    torch.manual_seed(42)
    coords = torch.randn(B, L, 3, 3)
    # NaN at special token positions
    coords[:, 0] = float("nan")
    coords[:, sep_pos] = float("nan")
    coords[:, -1] = float("nan")
    batch["coords"] = coords

    batch["chain_boundary"] = torch.full((B,), sep_pos, dtype=torch.long)
    return batch


@pytest.fixture
def mock_denoising_output(sample_batch):
    """DenoisingOutput with random logits matching batch dims."""
    B, L = sample_batch["input_ids"].shape
    V = 32
    torch.manual_seed(42)
    logits = torch.randn(B, L, V)

    # Encoder attentions: 2 layers, 4 heads
    num_layers = 2
    num_heads = 4
    attentions = tuple(
        torch.softmax(torch.randn(B, num_heads, L, L), dim=-1)
        for _ in range(num_layers)
    )

    return DenoisingOutput(
        loss=torch.tensor(2.5),
        logits=logits,
        encoder_attentions=attentions,
    )


@pytest.fixture(scope="session")
def eval_cfg():
    """OmegaConf config from configs/eval/denoising.yaml merged with model config."""
    eval_yaml = OmegaConf.create({
        "masking": {"mask_rate": 0.15, "seed": 42},
        "metrics": {
            "loss": {"enabled": True},
            "masked_accuracy": {"enabled": True},
            "perplexity": {"enabled": True},
            "p_at_l": {
                "enabled": True,
                "contact_threshold": 8.0,
                "min_seq_sep": 6,
                "num_layers": None,
                "head_aggregation": "mean",
                "contact_mode": ["all", "cross_chain"],
            },
        },
        "regions": {"enabled": False},
    })
    model_cfg = OmegaConf.create({
        "model": {
            "num_encoder_layers": 2,
        },
    })
    return OmegaConf.merge(model_cfg, {"eval": eval_yaml})
