# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for MutableEncoder."""

import torch

from mutable.config import MutableConfig
from mutable.models.encoder import MutableEncoder
from mutable.outputs import EncoderOutput


class TestMutableEncoder:
    @staticmethod
    def _make_encoder(**overrides):
        defaults = dict(
            hidden_size=64, num_encoder_layers=2, num_decoder_layers=2,
            num_attention_heads=4, intermediate_size=128, vocab_size=32,
            dropout=0.0, attention_dropout=0.0, hidden_dropout=0.0,
        )
        defaults.update(overrides)
        config = MutableConfig(**defaults)
        return MutableEncoder(config)

    def test_forward_output_shape(self):
        enc = self._make_encoder()
        ids = torch.randint(0, 32, (2, 16))
        out = enc(input_ids=ids, return_dict=True)
        assert out.last_hidden_state.shape == (2, 16, 64)

    def test_inputs_embeds_path(self):
        enc = self._make_encoder()
        embeds = torch.randn(2, 16, 64)
        out = enc(inputs_embeds=embeds, return_dict=True)
        assert out.last_hidden_state.shape == (2, 16, 64)

    def test_output_attentions(self):
        enc = self._make_encoder()
        ids = torch.randint(0, 32, (2, 16))
        out = enc(input_ids=ids, output_attentions=True, return_dict=True)
        assert out.attentions is not None
        assert len(out.attentions) == 2  # num_encoder_layers

    def test_output_hidden_states(self):
        enc = self._make_encoder()
        ids = torch.randint(0, 32, (2, 16))
        out = enc(input_ids=ids, output_hidden_states=True, return_dict=True)
        assert out.hidden_states is not None
        # num_layers + 1 (input embeddings + each layer output)
        assert len(out.hidden_states) == 3

    def test_return_dict_false(self):
        enc = self._make_encoder()
        ids = torch.randint(0, 32, (2, 16))
        out = enc(input_ids=ids, return_dict=False)
        assert isinstance(out, tuple)

    def test_attention_mask_propagated(self):
        enc = self._make_encoder()
        enc.eval()
        ids = torch.randint(4, 29, (1, 10))
        mask = torch.ones(1, 10, dtype=torch.long)
        mask[:, 8:] = 0
        out = enc(input_ids=ids, attention_mask=mask, return_dict=True)
        assert out.last_hidden_state.shape == (1, 10, 64)

    def test_absolute_position_embeddings(self):
        enc = self._make_encoder(position_embedding_type="absolute")
        ids = torch.randint(0, 32, (2, 16))
        out = enc(input_ids=ids, return_dict=True)
        assert out.last_hidden_state.shape == (2, 16, 64)
