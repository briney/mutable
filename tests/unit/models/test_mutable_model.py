# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for MutableModel (full encoder-bottleneck-decoder)."""

import torch

from mutable.outputs import MutableModelFullOutput, BottleneckOutput


class TestMutableModel:
    def test_encode_produces_latents(self, small_model):
        ids = torch.randint(4, 29, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        out = small_model.mutable.encode(input_ids=ids, attention_mask=mask, return_dict=True)
        assert isinstance(out, BottleneckOutput)
        assert out.latent_states.shape == (2, 8, 64)

    def test_decode_produces_output(self, small_model):
        dec_ids = torch.randint(4, 29, (2, 16))
        enc_hidden = torch.randn(2, 8, 64)
        out = small_model.mutable.decode(
            decoder_input_ids=dec_ids, encoder_hidden_states=enc_hidden, return_dict=True
        )
        assert out.last_hidden_state.shape == (2, 16, 64)

    def test_forward_returns_full_output(self, small_model):
        ids = torch.randint(4, 29, (2, 16))
        dec_ids = torch.randint(4, 29, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        out = small_model.mutable(
            input_ids=ids, attention_mask=mask,
            decoder_input_ids=dec_ids, return_dict=True,
        )
        assert isinstance(out, MutableModelFullOutput)
        assert out.last_hidden_state.shape == (2, 16, 64)
        assert out.latent_states.shape == (2, 8, 64)
        assert out.encoder_last_hidden_state.shape == (2, 16, 64)

    def test_encoder_decoder_separate_embeddings(self, small_model):
        enc_emb = small_model.mutable.encoder.word_embeddings
        dec_emb = small_model.mutable.decoder.word_embeddings
        # Should be separate nn.Embedding instances
        assert enc_emb is not dec_emb

    def test_output_attentions_flag(self, small_model):
        ids = torch.randint(4, 29, (2, 16))
        dec_ids = torch.randint(4, 29, (2, 16))
        out = small_model.mutable(
            input_ids=ids, decoder_input_ids=dec_ids,
            output_attentions=True, return_dict=True,
        )
        assert out.encoder_attentions is not None
        assert out.decoder_self_attentions is not None
        assert out.decoder_cross_attentions is not None

    def test_output_hidden_states_flag(self, small_model):
        ids = torch.randint(4, 29, (2, 16))
        dec_ids = torch.randint(4, 29, (2, 16))
        out = small_model.mutable(
            input_ids=ids, decoder_input_ids=dec_ids,
            output_hidden_states=True, return_dict=True,
        )
        assert out.encoder_hidden_states is not None
        assert out.decoder_hidden_states is not None

    def test_return_dict_false(self, small_model):
        ids = torch.randint(4, 29, (2, 16))
        dec_ids = torch.randint(4, 29, (2, 16))
        out = small_model.mutable(
            input_ids=ids, decoder_input_ids=dec_ids, return_dict=False,
        )
        assert isinstance(out, tuple)
