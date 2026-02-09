# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for embedding modules."""

import torch

from mutable.modules.embedding import (
    rotate_half,
    apply_rotary_pos_emb,
    RotaryPositionalEmbedding,
    SinusoidalTimestepEmbedding,
    LearnedLatentEmbedding,
)


class TestRotateHalf:
    def test_structure(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x)
        # should be [-x2, x1] = [-3, -4, 1, 2]
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        torch.testing.assert_close(out, expected)

    def test_double_rotation_is_negation(self):
        x = torch.randn(2, 4, 8, 16)
        out = rotate_half(rotate_half(x))
        torch.testing.assert_close(out, -x)


class TestApplyRotaryPosEmb:
    def test_shape_preserved(self):
        x = torch.randn(1, 1, 8, 16)
        cos = torch.ones(1, 1, 8, 16)
        sin = torch.zeros(1, 1, 8, 16)
        out = apply_rotary_pos_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_identity_when_cos1_sin0(self):
        x = torch.randn(1, 1, 8, 16)
        cos = torch.ones(1, 1, 8, 16)
        sin = torch.zeros(1, 1, 8, 16)
        out = apply_rotary_pos_emb(x, cos, sin)
        torch.testing.assert_close(out, x)


class TestRotaryPositionalEmbedding:
    def test_shapes_preserved(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q = torch.randn(2, 4, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_caching_reuses_tables(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q = torch.randn(2, 4, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        rope(q, k)
        cos_first = rope._cos_cached
        rope(q, k)
        # Should be same object (cached)
        assert cos_first is rope._cos_cached

    def test_different_lengths_trigger_recomputation(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q1 = torch.randn(2, 4, 10, 16)
        k1 = torch.randn(2, 4, 10, 16)
        rope(q1, k1)
        cached_len_1 = rope._seq_len_cached

        q2 = torch.randn(2, 4, 20, 16)
        k2 = torch.randn(2, 4, 20, 16)
        rope(q2, k2)
        assert rope._seq_len_cached != cached_len_1

    def test_position_dependent_output(self):
        rope = RotaryPositionalEmbedding(dim=16)
        q = torch.ones(1, 1, 4, 16)
        k = torch.ones(1, 1, 4, 16)
        q_out, _ = rope(q, k)
        # different positions should produce different outputs
        assert not torch.allclose(q_out[0, 0, 0], q_out[0, 0, 1])


class TestSinusoidalTimestepEmbedding:
    def test_shape_even_dim(self):
        emb = SinusoidalTimestepEmbedding(embedding_dim=32)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert out.shape == (3, 32)

    def test_shape_odd_dim(self):
        emb = SinusoidalTimestepEmbedding(embedding_dim=33)
        t = torch.tensor([0.0, 0.5])
        out = emb(t)
        assert out.shape == (2, 33)

    def test_different_timesteps_give_different_outputs(self):
        emb = SinusoidalTimestepEmbedding(embedding_dim=32)
        t = torch.tensor([0.1, 0.9])
        out = emb(t)
        assert not torch.allclose(out[0], out[1])


class TestLearnedLatentEmbedding:
    def test_shape(self):
        emb = LearnedLatentEmbedding(num_latents=8, latent_dim=64)
        out = emb(batch_size=4)
        assert out.shape == (4, 8, 64)

    def test_all_batch_copies_identical(self):
        emb = LearnedLatentEmbedding(num_latents=8, latent_dim=64)
        out = emb(batch_size=3)
        torch.testing.assert_close(out[0], out[1])
        torch.testing.assert_close(out[1], out[2])
