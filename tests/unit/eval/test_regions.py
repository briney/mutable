# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for region mask extraction and aggregation."""

import torch
import pytest

from mutable.eval.regions import (
    AntibodyRegion,
    CDR_REGIONS,
    FWR_REGIONS,
    aggregate_region_masks,
    derive_chain_ids,
    extract_region_masks,
)


class TestDeriveChainIds:
    def test_basic(self):
        """Input with <sep> at pos 10 gives 0s before, 1s after."""
        B, L = 1, 20
        input_ids = torch.randint(4, 29, (B, L))
        input_ids[0, 10] = 29  # SEP

        chain_ids = derive_chain_ids(input_ids, sep_token_id=29)
        assert chain_ids[0, :11].sum().item() == 0  # 0-10 are heavy
        assert (chain_ids[0, 11:] == 1).all()

    def test_no_sep(self):
        """No <sep> token results in all zeros."""
        B, L = 1, 20
        input_ids = torch.randint(4, 28, (B, L))  # exclude 29

        chain_ids = derive_chain_ids(input_ids, sep_token_id=29)
        assert chain_ids.sum().item() == 0

    def test_batch(self):
        """Batch of 4 with varying <sep> positions."""
        B, L = 4, 20
        input_ids = torch.randint(4, 28, (B, L))
        sep_positions = [5, 8, 12, 15]
        for b, sp in enumerate(sep_positions):
            input_ids[b, sp] = 29

        chain_ids = derive_chain_ids(input_ids, sep_token_id=29)
        for b, sp in enumerate(sep_positions):
            assert chain_ids[b, :sp + 1].sum().item() == 0
            assert (chain_ids[b, sp + 1:] == 1).all()


class TestExtractRegionMasks:
    @pytest.fixture
    def region_batch(self):
        """Batch with known CDR positions for region extraction."""
        B, L = 2, 32
        sep_pos = 16

        input_ids = torch.randint(4, 29, (B, L))
        input_ids[:, 0] = 0    # BOS
        input_ids[:, sep_pos] = 29  # SEP
        input_ids[:, -1] = 2   # EOS

        attention_mask = torch.ones(B, L, dtype=torch.long)

        special_tokens_mask = torch.zeros(B, L, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, sep_pos] = True
        special_tokens_mask[:, -1] = True

        chain_ids = torch.zeros(B, L, dtype=torch.long)
        chain_ids[:, sep_pos + 1:] = 1

        # CDR mask: heavy CDR1(3-5), CDR2(8-10), CDR3(12-14)
        #           light CDR1(19-21), CDR2(24-26), CDR3(28-30)
        cdr_mask = torch.zeros(B, L, dtype=torch.long)
        cdr_mask[:, 3:6] = 1
        cdr_mask[:, 8:11] = 2
        cdr_mask[:, 12:15] = 3
        cdr_mask[:, 19:22] = 1
        cdr_mask[:, 24:27] = 2
        cdr_mask[:, 28:31] = 3
        cdr_mask[:, 0] = 0
        cdr_mask[:, sep_pos] = 0
        cdr_mask[:, -1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
        }

    def test_cdr_masks(self, region_batch):
        masks = extract_region_masks(region_batch)
        # HCDR1 should be True at positions 3-5
        assert masks[AntibodyRegion.HCDR1][0, 3:6].all()
        # LCDR3 should be True at 28-30
        assert masks[AntibodyRegion.LCDR3][0, 28:31].all()

    def test_framework_masks(self, region_batch):
        masks = extract_region_masks(region_batch)
        # HFWR1 should be True between BOS+1 and CDR1_start (positions 1-2)
        assert masks[AntibodyRegion.HFWR1][0, 1:3].all()
        # HFWR2 should be True between CDR1_end and CDR2_start (positions 6-7)
        assert masks[AntibodyRegion.HFWR2][0, 6:8].all()

    def test_returns_all_14(self, region_batch):
        masks = extract_region_masks(region_batch)
        assert len(masks) == 14
        for region in AntibodyRegion:
            assert region in masks

    def test_special_tokens_excluded(self, region_batch):
        masks = extract_region_masks(region_batch)
        for region, mask in masks.items():
            assert mask[:, 0].sum().item() == 0, f"BOS in {region}"
            assert mask[:, 16].sum().item() == 0, f"SEP in {region}"
            assert mask[:, -1].sum().item() == 0, f"EOS in {region}"


class TestAggregateRegionMasks:
    @pytest.fixture
    def simple_masks(self):
        B, L = 1, 10
        masks = {}
        for region in AntibodyRegion:
            masks[region] = torch.zeros(B, L, dtype=torch.bool)
        # Set some positions
        masks[AntibodyRegion.HCDR1][0, 0] = True
        masks[AntibodyRegion.HCDR2][0, 1] = True
        masks[AntibodyRegion.LCDR1][0, 5] = True
        masks[AntibodyRegion.HFWR1][0, 3] = True
        masks[AntibodyRegion.LFWR1][0, 7] = True
        return masks

    def test_cdr_group(self, simple_masks):
        result = aggregate_region_masks(simple_masks, "cdr")
        assert "cdr" in result
        # Should union all CDR masks
        assert result["cdr"][0, 0].item() is True   # HCDR1
        assert result["cdr"][0, 1].item() is True   # HCDR2
        assert result["cdr"][0, 5].item() is True   # LCDR1
        assert result["cdr"][0, 3].item() is False   # HFWR1 (not CDR)

    def test_chain_aggregation(self, simple_masks):
        result = aggregate_region_masks(simple_masks, "chain")
        assert "heavy" in result
        assert "light" in result
        assert result["heavy"][0, 0].item() is True   # HCDR1
        assert result["heavy"][0, 3].item() is True   # HFWR1
        assert result["light"][0, 5].item() is True   # LCDR1
        assert result["light"][0, 7].item() is True   # LFWR1
