# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for MutableForDenoising."""

import torch

from mutable.config import MutableConfig
from mutable.models.mutable_denoising import MutableForDenoising
from mutable.outputs import DenoisingOutput


class TestMutableForDenoising:
    @staticmethod
    def _model_batch(sample_batch):
        """Strip non-model keys from sample_batch."""
        return {k: v for k, v in sample_batch.items() if k not in ("special_tokens_mask",)}

    def test_forward_with_labels_returns_loss(self, small_model, sample_batch):
        batch = self._model_batch(sample_batch)
        out = small_model(**batch, return_dict=True)
        assert isinstance(out, DenoisingOutput)
        assert out.loss is not None
        assert out.loss.dim() == 0  # scalar

    def test_logits_shape(self, small_model, sample_batch):
        batch = self._model_batch(sample_batch)
        out = small_model(**batch, return_dict=True)
        B, L = sample_batch["input_ids"].shape
        assert out.logits.shape == (B, L, 32)  # vocab_size=32

    def test_forward_without_labels(self, small_model):
        ids = torch.randint(4, 29, (2, 16))
        dec_ids = torch.randint(4, 29, (2, 16))
        out = small_model(
            input_ids=ids, decoder_input_ids=dec_ids, return_dict=True,
        )
        assert out.loss is None
        assert out.logits is not None

    def test_shift_right_prepends_bos(self, small_model):
        labels = torch.tensor([[5, 6, 7, 8, 9]])
        shifted = small_model._shift_right(labels)
        assert shifted[0, 0].item() == 0  # BOS

    def test_shift_right_drops_last_token(self, small_model):
        labels = torch.tensor([[5, 6, 7, 8, 9]])
        shifted = small_model._shift_right(labels)
        # last token of original (9) should not be in shifted
        assert shifted[0, -1].item() == labels[0, -2].item()

    def test_shift_right_replaces_neg100_with_pad(self, small_model):
        labels = torch.tensor([[5, -100, 7, -100, 9]])
        shifted = small_model._shift_right(labels)
        # -100 should become pad_token_id (1)
        assert (shifted == -100).sum() == 0
        assert (shifted == 1).sum() > 0

    def test_encoder_attentions_populated(self, small_model, sample_batch):
        batch = self._model_batch(sample_batch)
        out = small_model(**batch, output_attentions=True, return_dict=True)
        assert out.encoder_attentions is not None
        assert len(out.encoder_attentions) == 2  # num_encoder_layers

    def test_latent_states_shape(self, small_model, sample_batch):
        batch = self._model_batch(sample_batch)
        out = small_model(**batch, return_dict=True)
        assert out.latent_states.shape == (4, 8, 64)
