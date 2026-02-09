# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for attention modules."""

import pytest
import torch

from mutable.modules.attention import SelfAttention, CrossAttention


class TestSelfAttention:
    def test_output_shape(self):
        attn = SelfAttention(model_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out, _ = attn(x)
        assert out.shape == (2, 10, 64)

    def test_without_rotary(self):
        attn = SelfAttention(
            model_dim=64, num_heads=4, dropout=0.0, position_embedding_type="none"
        )
        x = torch.randn(2, 10, 64)
        out, _ = attn(x)
        assert out.shape == (2, 10, 64)

    def test_with_rotary(self):
        attn = SelfAttention(
            model_dim=64, num_heads=4, dropout=0.0, position_embedding_type="rotary"
        )
        x = torch.randn(2, 10, 64)
        out, _ = attn(x)
        assert out.shape == (2, 10, 64)

    def test_need_weights_returns_weights(self):
        attn = SelfAttention(model_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out, weights = attn(x, need_weights=True)
        assert weights is not None
        assert weights.shape == (2, 4, 10, 10)

    def test_padding_mask_applied(self):
        attn = SelfAttention(
            model_dim=64, num_heads=4, dropout=0.0, position_embedding_type="none"
        )
        attn.eval()
        x = torch.randn(1, 5, 64)
        # mask: first 3 valid, last 2 padded
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        _, weights = attn(x, attention_mask=mask, need_weights=True)
        # attention to padded positions should be near zero
        assert weights[0, :, :, 3:].max() < 1e-5

    def test_causal_masking_prevents_future_leakage(self):
        attn = SelfAttention(
            model_dim=64, num_heads=4, dropout=0.0, position_embedding_type="none"
        )
        attn.eval()
        torch.manual_seed(42)
        # identical prefix, different suffix
        x1 = torch.randn(1, 6, 64)
        x2 = x1.clone()
        x2[:, 4:] = torch.randn(1, 2, 64)
        out1, _ = attn(x1, is_causal=True)
        out2, _ = attn(x2, is_causal=True)
        # outputs before the divergence point (positions 0-3) should match
        torch.testing.assert_close(out1[:, :4], out2[:, :4], atol=1e-5, rtol=1e-5)

    def test_model_dim_not_divisible_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            SelfAttention(model_dim=65, num_heads=4)

    def test_deterministic_in_eval(self):
        attn = SelfAttention(model_dim=64, num_heads=4, dropout=0.1)
        attn.eval()
        x = torch.randn(2, 10, 64)
        out1, _ = attn(x)
        out2, _ = attn(x)
        torch.testing.assert_close(out1, out2)


class TestCrossAttention:
    def test_output_shape(self):
        attn = CrossAttention(model_dim=64, num_heads=4, dropout=0.0)
        q = torch.randn(2, 8, 64)
        kv = torch.randn(2, 12, 64)
        out, _ = attn(q, kv)
        assert out.shape == (2, 8, 64)

    def test_different_kv_dim(self):
        attn = CrossAttention(model_dim=64, kv_dim=32, num_heads=4, dropout=0.0)
        q = torch.randn(2, 8, 64)
        kv = torch.randn(2, 12, 32)
        out, _ = attn(q, kv)
        assert out.shape == (2, 8, 64)

    def test_need_weights_returns_weights(self):
        attn = CrossAttention(model_dim=64, num_heads=4, dropout=0.0)
        q = torch.randn(2, 8, 64)
        kv = torch.randn(2, 12, 64)
        out, weights = attn(q, kv, need_weights=True)
        assert weights is not None
        assert weights.shape == (2, 4, 8, 12)

    def test_key_value_mask_applied(self):
        attn = CrossAttention(model_dim=64, num_heads=4, dropout=0.0)
        attn.eval()
        q = torch.randn(1, 4, 64)
        kv = torch.randn(1, 6, 64)
        # only first 3 kv positions valid
        kv_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        _, weights = attn(q, kv, key_value_mask=kv_mask, need_weights=True)
        # attention to masked kv positions should be near zero
        assert weights[0, :, :, 3:].max() < 1e-5

    def test_no_rotary_embeddings(self):
        attn = CrossAttention(model_dim=64, num_heads=4, dropout=0.0)
        # CrossAttention should not have a rotary_embed attribute
        assert not hasattr(attn, "rotary_embed")
