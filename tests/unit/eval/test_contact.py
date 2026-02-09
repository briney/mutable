# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for PrecisionAtLMetric and helpers."""

import torch
import pytest

from mutable.outputs import DenoisingOutput
from mutable.eval.metrics.contact import (
    PrecisionAtLMetric,
    _apply_apc,
    _build_pair_mask,
    _compute_contact_map,
    _extract_attention_contacts,
    _filter_by_mode,
)


class TestComputeContactMap:
    def test_simple(self):
        """Known coords produce expected contact map."""
        B, L = 1, 4
        coords = torch.zeros(B, L, 3, 3)
        # Place CA atoms (index 1) at known positions
        coords[0, 0, 1, :] = torch.tensor([0.0, 0.0, 0.0])
        coords[0, 1, 1, :] = torch.tensor([5.0, 0.0, 0.0])  # 5A from 0
        coords[0, 2, 1, :] = torch.tensor([0.0, 10.0, 0.0])  # 10A from 0
        coords[0, 3, 1, :] = torch.tensor([3.0, 0.0, 0.0])  # 3A from 0

        contacts = _compute_contact_map(coords, threshold=8.0)
        assert contacts.shape == (1, 4, 4)
        # 0-1: 5A < 8 => contact
        assert contacts[0, 0, 1].item() is True
        # 0-2: 10A > 8 => no contact
        assert contacts[0, 0, 2].item() is False
        # 0-3: 3A < 8 => contact
        assert contacts[0, 0, 3].item() is True

    def test_nan_handling(self):
        """NaN coords produce no contacts at those positions."""
        B, L = 1, 3
        coords = torch.zeros(B, L, 3, 3)
        coords[0, 0, 1, :] = torch.tensor([0.0, 0.0, 0.0])
        coords[0, 1, :, :] = float("nan")
        coords[0, 2, 1, :] = torch.tensor([3.0, 0.0, 0.0])

        contacts = _compute_contact_map(coords, threshold=8.0)
        # NaN position has no contacts
        assert contacts[0, 0, 1].item() is False
        assert contacts[0, 1, 2].item() is False


class TestApplyAPC:
    def test_known_matrix(self):
        """Verified APC output for a known input."""
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _apply_apc(matrix)
        # row_mean: [[1.5], [3.5]]
        # col_mean: [[2.0, 3.0]]
        # global_mean: 2.5
        # correction: row_mean * col_mean / global_mean
        expected_correction = torch.tensor([
            [1.5 * 2.0 / 2.5, 1.5 * 3.0 / 2.5],
            [3.5 * 2.0 / 2.5, 3.5 * 3.0 / 2.5],
        ])
        expected = matrix - expected_correction
        assert torch.allclose(result, expected, atol=1e-5)


class TestExtractAttentionContacts:
    def test_last_layer(self):
        B, H, L = 1, 2, 4
        attns = (
            torch.ones(B, H, L, L) * 0.1,
            torch.ones(B, H, L, L) * 0.5,
        )
        result = _extract_attention_contacts(attns, layer="last", num_layers=1)
        assert result is not None
        assert result.shape == (B, L, L)

    def test_mean_aggregation(self):
        B, H, L = 1, 2, 4
        head1 = torch.ones(B, 1, L, L) * 0.2
        head2 = torch.ones(B, 1, L, L) * 0.8
        attns = (torch.cat([head1, head2], dim=1),)
        result = _extract_attention_contacts(
            attns, layer="last", head_aggregation="mean", num_layers=1
        )
        assert result is not None
        # Before APC, mean of heads = 0.5
        # After symmetrize + APC, values should be defined
        assert result.shape == (B, L, L)

    def test_none_attentions(self):
        result = _extract_attention_contacts(None)
        assert result is None


class TestBuildPairMask:
    def test_min_seq_sep(self):
        B, L = 1, 10
        input_ids = torch.randint(4, 29, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        special_tokens_mask = torch.zeros(B, L, dtype=torch.bool)

        pair_mask = _build_pair_mask(input_ids, attention_mask, special_tokens_mask, min_seq_sep=6)
        # Pairs closer than 6 should be excluded
        assert pair_mask[0, 0, 1].item() is False  # |0-1| = 1 < 6
        assert pair_mask[0, 0, 5].item() is False  # |0-5| = 5 < 6
        assert pair_mask[0, 0, 6].item() is True   # |0-6| = 6 >= 6

    def test_special_tokens_excluded(self):
        B, L = 1, 10
        input_ids = torch.randint(4, 29, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        special_tokens_mask = torch.zeros(B, L, dtype=torch.bool)
        special_tokens_mask[0, 0] = True  # BOS
        special_tokens_mask[0, -1] = True  # EOS

        pair_mask = _build_pair_mask(input_ids, attention_mask, special_tokens_mask, min_seq_sep=1)
        # BOS and EOS should be excluded from all pairs
        assert pair_mask[0, 0, :].sum().item() == 0
        assert pair_mask[0, :, 0].sum().item() == 0
        assert pair_mask[0, -1, :].sum().item() == 0


class TestFilterByMode:
    @pytest.fixture
    def setup(self):
        B, L = 1, 20
        pair_mask = torch.ones(B, L, L, dtype=torch.bool)
        chain_boundary = torch.tensor([10])  # heavy: 0-9, light: 11-19
        cdr_mask = torch.zeros(B, L, dtype=torch.long)
        cdr_mask[0, 3:6] = 1  # CDR in heavy
        return pair_mask, chain_boundary, cdr_mask

    def test_all_mode(self, setup):
        pair_mask, boundary, cdr_mask = setup
        result = _filter_by_mode(pair_mask, "all", boundary)
        assert torch.equal(result, pair_mask)

    def test_cross_chain(self, setup):
        pair_mask, boundary, cdr_mask = setup
        result = _filter_by_mode(pair_mask, "cross_chain", boundary)
        # Heavy-heavy pairs should be False
        assert result[0, 0, 1].item() is False
        # Light-light pairs should be False
        assert result[0, 11, 12].item() is False
        # Heavy-light pairs should be True
        assert result[0, 0, 11].item() is True

    def test_intra_heavy(self, setup):
        pair_mask, boundary, cdr_mask = setup
        result = _filter_by_mode(pair_mask, "intra_heavy", boundary)
        # Heavy-heavy pair
        assert result[0, 0, 5].item() is True
        # Heavy-light pair excluded
        assert result[0, 0, 15].item() is False
        # Light-light pair excluded
        assert result[0, 11, 15].item() is False

    def test_intra_light(self, setup):
        pair_mask, boundary, cdr_mask = setup
        result = _filter_by_mode(pair_mask, "intra_light", boundary)
        # Light-light pair
        assert result[0, 11, 15].item() is True
        # Heavy pairs excluded
        assert result[0, 0, 5].item() is False

    def test_cdr_contact(self, setup):
        pair_mask, boundary, cdr_mask = setup
        result = _filter_by_mode(pair_mask, "cdr_contact", boundary, cdr_mask)
        # At least one CDR position required
        # Position 3 is CDR, 15 is not CDR => should be True (one CDR is enough)
        assert result[0, 3, 15].item() is True
        # Positions 0 and 15 are both non-CDR => should be False
        assert result[0, 0, 15].item() is False


class TestPrecisionAtLMetric:
    def test_p_at_l_perfect_attention(self):
        """Attention perfectly matching contacts gives precision ~ 1.0."""
        B, L = 1, 20
        V = 32
        num_heads = 4
        sep_pos = 10

        # Create coords where first few positions are in contact
        coords = torch.zeros(B, L, 3, 3)
        for i in range(L):
            coords[0, i, 1, :] = torch.tensor([float(i) * 3.0, 0.0, 0.0])
        # Make close contacts: 0-1, 1-2, etc (within 8A)
        # Positions 3 apart: 9A > 8A, so not contacts
        coords[:, 0] = float("nan")   # BOS
        coords[:, sep_pos] = float("nan")  # SEP
        coords[:, -1] = float("nan")  # EOS

        # Build attention that focuses on close contacts
        true_contacts = _compute_contact_map(coords, threshold=8.0)
        attn = true_contacts.float().unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn = attn + 1e-6
        attentions = (attn,)

        output = DenoisingOutput(
            logits=torch.randn(B, L, V),
            encoder_attentions=attentions,
        )
        batch = {
            "input_ids": torch.randint(4, 29, (B, L)),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
            "special_tokens_mask": torch.zeros(B, L, dtype=torch.bool),
            "labels": torch.randint(0, V, (B, L)),
            "coords": coords,
            "chain_boundary": torch.tensor([sep_pos]),
        }
        batch["input_ids"][:, 0] = 0
        batch["input_ids"][:, sep_pos] = 29
        batch["input_ids"][:, -1] = 2
        batch["special_tokens_mask"][:, 0] = True
        batch["special_tokens_mask"][:, sep_pos] = True
        batch["special_tokens_mask"][:, -1] = True

        mask_labels = torch.ones(B, L, dtype=torch.bool)

        m = PrecisionAtLMetric(contact_mode="all", min_seq_sep=1, num_layers=1)
        m.update(output, batch, mask_labels)
        result = m.compute()
        assert "p_at_l" in result
        # Precision should be high (attention matches contacts)
        assert result["p_at_l"] > 0.5

    def test_multi_mode(self):
        """Multiple contact_mode returns keys for each mode."""
        m = PrecisionAtLMetric(contact_mode=["all", "cross_chain"])
        assert len(m.contact_modes) == 2
        result = m.compute()  # no updates, all zeros
        assert "p_at_l" in result
        assert "p_at_l/cross_chain" in result

    def test_state_tensors_per_mode(self):
        m = PrecisionAtLMetric(contact_mode=["all", "cross_chain", "intra_heavy"])
        tensors = m.state_tensors()
        assert len(tensors) == 3  # one per mode
        for t in tensors:
            assert t.shape == (2,)  # [correct, total]
