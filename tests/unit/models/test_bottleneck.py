# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for PerceiverBottleneck."""

import torch

from mutable.config import MutableConfig
from mutable.models.bottleneck import PerceiverBottleneck
from mutable.outputs import BottleneckOutput


class TestPerceiverBottleneck:
    @staticmethod
    def _make_bottleneck(**overrides):
        defaults = dict(
            hidden_size=64, num_encoder_layers=2, num_decoder_layers=2,
            num_attention_heads=4, intermediate_size=128, vocab_size=32,
            num_latents=8, latent_dim=64,
            dropout=0.0, attention_dropout=0.0, hidden_dropout=0.0,
        )
        defaults.update(overrides)
        config = MutableConfig(**defaults)
        return PerceiverBottleneck(config)

    def test_output_shape(self):
        bn = self._make_bottleneck()
        enc = torch.randn(2, 20, 64)
        out = bn(enc, return_dict=True)
        assert out.latent_states.shape == (2, 8, 64)

    def test_compression_from_seq_len(self):
        bn = self._make_bottleneck()
        enc = torch.randn(2, 100, 64)
        out = bn(enc, return_dict=True)
        # 100 -> 8 latents
        assert out.latent_states.shape[1] == 8

    def test_encoder_mask_propagated(self):
        bn = self._make_bottleneck()
        enc = torch.randn(2, 20, 64)
        mask = torch.ones(2, 20)
        mask[:, 15:] = 0
        out = bn(enc, encoder_mask=mask, return_dict=True)
        assert out.latent_states.shape == (2, 8, 64)

    def test_need_weights_returns_attentions(self):
        bn = self._make_bottleneck()
        enc = torch.randn(2, 20, 64)
        out = bn(enc, need_weights=True, return_dict=True)
        assert out.attentions is not None

    def test_return_dict_false(self):
        bn = self._make_bottleneck()
        enc = torch.randn(2, 20, 64)
        out = bn(enc, return_dict=False)
        assert isinstance(out, tuple)

    def test_latent_queries_is_parameter(self):
        bn = self._make_bottleneck()
        assert isinstance(bn.latent_embedding.latent_queries, torch.nn.Parameter)
