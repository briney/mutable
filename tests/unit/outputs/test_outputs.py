# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for output dataclasses."""

import torch

from mutable.outputs import (
    MutableModelOutput,
    EncoderOutput,
    BottleneckOutput,
    DecoderOutput,
    MutableModelFullOutput,
    DenoisingOutput,
    FlowMatchingOutput,
)


class TestMutableModelOutputTo:
    def test_to_moves_tensor_fields(self):
        out = DenoisingOutput(
            loss=torch.tensor(1.0),
            logits=torch.randn(2, 4, 32),
        )
        moved = out.to("cpu")
        assert moved.logits.device == torch.device("cpu")

    def test_to_moves_tuple_of_tensor_fields(self):
        attn = (torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8))
        out = EncoderOutput(
            last_hidden_state=torch.randn(2, 8, 64),
            attentions=attn,
        )
        moved = out.to("cpu")
        assert all(t.device == torch.device("cpu") for t in moved.attentions)

    def test_to_skips_none_fields(self):
        out = DenoisingOutput(loss=None, logits=torch.randn(2, 4, 32))
        moved = out.to("cpu")
        assert moved.loss is None
        assert moved.logits is not None


class TestOutputDataclasses:
    def test_encoder_output_fields(self):
        out = EncoderOutput()
        assert out.last_hidden_state is None
        assert out.hidden_states is None
        assert out.attentions is None

    def test_bottleneck_output_fields(self):
        out = BottleneckOutput()
        assert out.latent_states is None
        assert out.attentions is None

    def test_decoder_output_fields(self):
        out = DecoderOutput()
        assert out.last_hidden_state is None
        assert out.hidden_states is None
        assert out.self_attentions is None
        assert out.cross_attentions is None

    def test_full_output_fields(self):
        out = MutableModelFullOutput()
        assert out.last_hidden_state is None
        assert out.latent_states is None
        assert out.encoder_last_hidden_state is None

    def test_denoising_output_fields(self):
        out = DenoisingOutput()
        assert out.loss is None
        assert out.logits is None
        assert out.latent_states is None
        assert out.encoder_attentions is None

    def test_flow_matching_output_fields(self):
        out = FlowMatchingOutput()
        assert out.loss is None
        assert out.predicted_velocity is None
        assert out.target_velocity is None
        assert out.latent_t is None

    def test_denoising_output_with_values(self):
        out = DenoisingOutput(
            loss=torch.tensor(2.5),
            logits=torch.randn(2, 10, 32),
            latent_states=torch.randn(2, 8, 64),
        )
        assert out.loss.item() == 2.5
        assert out.logits.shape == (2, 10, 32)

    def test_flow_matching_output_with_values(self):
        out = FlowMatchingOutput(
            loss=torch.tensor(0.5),
            predicted_velocity=torch.randn(2, 8, 64),
            target_velocity=torch.randn(2, 8, 64),
        )
        assert out.loss.item() == 0.5
