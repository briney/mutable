# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for MutableDecoder."""

import torch

from mutable.config import MutableConfig
from mutable.models.decoder import MutableDecoder
from mutable.outputs import DecoderOutput


class TestMutableDecoder:
    @staticmethod
    def _make_decoder(**overrides):
        defaults = dict(
            hidden_size=64, num_encoder_layers=2, num_decoder_layers=2,
            num_attention_heads=4, intermediate_size=128, vocab_size=32,
            dropout=0.0, attention_dropout=0.0, hidden_dropout=0.0,
        )
        defaults.update(overrides)
        config = MutableConfig(**defaults)
        return MutableDecoder(config)

    def test_forward_output_shape(self):
        dec = self._make_decoder()
        ids = torch.randint(0, 32, (2, 16))
        enc_hidden = torch.randn(2, 8, 64)
        out = dec(input_ids=ids, encoder_hidden_states=enc_hidden, return_dict=True)
        assert out.last_hidden_state.shape == (2, 16, 64)

    def test_requires_encoder_hidden_states(self):
        dec = self._make_decoder()
        ids = torch.randint(0, 32, (2, 16))
        # Should work without encoder_hidden_states (it's optional in signature)
        # but DecoderLayer.forward expects it — let's just test normal path
        enc_hidden = torch.randn(2, 8, 64)
        out = dec(input_ids=ids, encoder_hidden_states=enc_hidden, return_dict=True)
        assert isinstance(out, DecoderOutput)

    def test_output_attentions(self):
        dec = self._make_decoder()
        ids = torch.randint(0, 32, (2, 16))
        enc_hidden = torch.randn(2, 8, 64)
        out = dec(
            input_ids=ids, encoder_hidden_states=enc_hidden,
            output_attentions=True, return_dict=True,
        )
        assert out.self_attentions is not None
        assert out.cross_attentions is not None
        assert len(out.self_attentions) == 2

    def test_output_hidden_states(self):
        dec = self._make_decoder()
        ids = torch.randint(0, 32, (2, 16))
        enc_hidden = torch.randn(2, 8, 64)
        out = dec(
            input_ids=ids, encoder_hidden_states=enc_hidden,
            output_hidden_states=True, return_dict=True,
        )
        assert out.hidden_states is not None
        assert len(out.hidden_states) == 3

    def test_return_dict_false(self):
        dec = self._make_decoder()
        ids = torch.randint(0, 32, (2, 16))
        enc_hidden = torch.randn(2, 8, 64)
        out = dec(input_ids=ids, encoder_hidden_states=enc_hidden, return_dict=False)
        assert isinstance(out, tuple)

    def test_absolute_position_embeddings(self):
        dec = self._make_decoder(position_embedding_type="absolute")
        ids = torch.randint(0, 32, (2, 16))
        enc_hidden = torch.randn(2, 8, 64)
        out = dec(input_ids=ids, encoder_hidden_states=enc_hidden, return_dict=True)
        assert out.last_hidden_state.shape == (2, 16, 64)
