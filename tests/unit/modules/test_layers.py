# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for transformer layer modules."""

import torch

from mutable.config import MutableConfig, FlowMatchingConfig
from mutable.modules.layers import (
    EncoderLayer,
    DecoderLayer,
    PerceiverBottleneckLayer,
    AdaptiveLayerNorm,
    FlowTransformerLayer,
)


def _make_config(**overrides):
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        layer_norm_eps=1e-5,
        activation="swiglu",
        ffn_bias=True,
        position_embedding_type="rotary",
        latent_dim=64,
    )
    defaults.update(overrides)
    return MutableConfig(**defaults)


class TestEncoderLayer:
    def test_output_shape(self):
        config = _make_config()
        layer = EncoderLayer(config)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_with_attention_mask(self):
        config = _make_config()
        layer = EncoderLayer(config)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10)
        mask[:, 8:] = 0
        out = layer(x, attention_mask=mask)
        assert out.shape == (2, 10, 64)

    def test_need_weights_returns_tuple(self):
        config = _make_config()
        layer = EncoderLayer(config)
        x = torch.randn(2, 10, 64)
        result = layer(x, need_weights=True)
        assert isinstance(result, tuple)
        out, attn_weights = result
        assert out.shape == (2, 10, 64)
        assert attn_weights.shape == (2, 4, 10, 10)

    def test_with_gelu_activation(self):
        config = _make_config(activation="gelu")
        layer = EncoderLayer(config)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 64)


class TestDecoderLayer:
    def test_output_shape(self):
        config = _make_config()
        layer = DecoderLayer(config)
        x = torch.randn(2, 10, 64)
        enc = torch.randn(2, 8, 64)
        out = layer(x, encoder_hidden_states=enc)
        assert out.shape == (2, 10, 64)

    def test_cross_attention_to_different_length(self):
        config = _make_config()
        layer = DecoderLayer(config)
        x = torch.randn(2, 10, 64)
        enc = torch.randn(2, 20, 64)
        out = layer(x, encoder_hidden_states=enc)
        assert out.shape == (2, 10, 64)

    def test_need_weights_returns_3_tuple(self):
        config = _make_config()
        layer = DecoderLayer(config)
        x = torch.randn(2, 10, 64)
        enc = torch.randn(2, 8, 64)
        result = layer(x, encoder_hidden_states=enc, need_weights=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        out, self_attn, cross_attn = result
        assert out.shape == (2, 10, 64)

    def test_causal_self_attention(self):
        config = _make_config()
        layer = DecoderLayer(config)
        layer.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 6, 64)
        enc = torch.randn(1, 4, 64)
        # Decoder layer uses is_causal=True internally
        result = layer(x, encoder_hidden_states=enc, need_weights=True)
        _, self_attn_w, _ = result
        # Upper triangular should be near zero (causal mask)
        upper = torch.triu(self_attn_w[0, 0], diagonal=1)
        assert upper.abs().max() < 1e-5


class TestPerceiverBottleneckLayer:
    def test_output_shape(self):
        config = _make_config()
        layer = PerceiverBottleneckLayer(config)
        latents = torch.randn(2, 8, 64)
        enc = torch.randn(2, 20, 64)
        out = layer(latents, enc)
        assert out.shape == (2, 8, 64)

    def test_encoder_mask_applied(self):
        config = _make_config()
        layer = PerceiverBottleneckLayer(config)
        layer.eval()
        latents = torch.randn(2, 4, 64)
        enc = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0
        out = layer(latents, enc, encoder_mask=mask)
        assert out.shape == (2, 4, 64)

    def test_need_weights_returns_tuple(self):
        config = _make_config()
        layer = PerceiverBottleneckLayer(config)
        latents = torch.randn(2, 4, 64)
        enc = torch.randn(2, 10, 64)
        result = layer(latents, enc, need_weights=True)
        assert isinstance(result, tuple)
        out, weights = result
        assert out.shape == (2, 4, 64)


class TestAdaptiveLayerNorm:
    def test_output_shape(self):
        aln = AdaptiveLayerNorm(hidden_size=64, conditioning_dim=32)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 32)
        out = aln(x, cond)
        assert out.shape == (2, 10, 64)

    def test_zero_init_means_identity(self):
        aln = AdaptiveLayerNorm(hidden_size=64, conditioning_dim=32)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 32)
        out = aln(x, cond)
        # zero-init proj: scale=0, shift=0 → output == LayerNorm(x)
        expected = aln.norm(x)
        torch.testing.assert_close(out, expected)

    def test_elementwise_affine_false(self):
        aln = AdaptiveLayerNorm(hidden_size=64, conditioning_dim=32)
        assert aln.norm.weight is None

    def test_different_conditioning_different_output(self):
        aln = AdaptiveLayerNorm(hidden_size=64, conditioning_dim=32)
        # Set non-zero weights to make conditioning matter
        torch.nn.init.normal_(aln.proj.weight)
        x = torch.randn(2, 10, 64)
        cond1 = torch.randn(2, 32)
        cond2 = torch.randn(2, 32)
        out1 = aln(x, cond1)
        out2 = aln(x, cond2)
        assert not torch.allclose(out1, out2)


class TestFlowTransformerLayer:
    def _make_flow_config(self):
        return FlowMatchingConfig(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
            time_embedding_dim=32,
            mu_embedding_dim=32,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            activation="swiglu",
            ffn_bias=True,
        )

    def test_output_shape(self):
        config = self._make_flow_config()
        conditioning_dim = 64  # time + mu
        layer = FlowTransformerLayer(config, conditioning_dim=conditioning_dim)
        x = torch.randn(2, 8, 64)
        germline = torch.randn(2, 8, 64)
        cond = torch.randn(2, 64)
        out = layer(x, germline_latents=germline, conditioning=cond)
        assert out.shape == (2, 8, 64)

    def test_no_rotary_in_self_attention(self):
        config = self._make_flow_config()
        conditioning_dim = 64
        layer = FlowTransformerLayer(config, conditioning_dim=conditioning_dim)
        # The self-attention should have position_embedding_type="none" (no rotary)
        assert layer.self_attention.rotary_embed is None

    def test_conditioning_affects_output(self):
        config = self._make_flow_config()
        conditioning_dim = 64
        layer = FlowTransformerLayer(config, conditioning_dim=conditioning_dim)
        # Set non-zero AdaLN weights
        for name, module in layer.named_modules():
            if isinstance(module, AdaptiveLayerNorm):
                torch.nn.init.normal_(module.proj.weight)
        x = torch.randn(2, 8, 64)
        germline = torch.randn(2, 8, 64)
        cond1 = torch.randn(2, 64)
        cond2 = torch.randn(2, 64)
        out1 = layer(x, germline_latents=germline, conditioning=cond1)
        out2 = layer(x, germline_latents=germline, conditioning=cond2)
        assert not torch.allclose(out1, out2)
