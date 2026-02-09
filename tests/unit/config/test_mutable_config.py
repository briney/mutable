# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for MutableConfig."""

import pytest

from mutable.config import MutableConfig


class TestMutableConfig:
    def test_default_values(self):
        config = MutableConfig()
        assert config.hidden_size == 320
        assert config.vocab_size == 32
        assert config.num_encoder_layers == 6
        assert config.num_decoder_layers == 6
        assert config.num_attention_heads == 20
        assert config.num_latents == 32
        assert config.dropout == 0.1
        assert config.activation == "swiglu"
        assert config.position_embedding_type == "rotary"

    def test_intermediate_size_defaults_to_hidden_times_4(self):
        config = MutableConfig(hidden_size=64, num_attention_heads=4)
        assert config.intermediate_size == 256

    def test_intermediate_size_explicit(self):
        config = MutableConfig(hidden_size=64, num_attention_heads=4, intermediate_size=128)
        assert config.intermediate_size == 128

    def test_latent_dim_defaults_to_hidden_size(self):
        config = MutableConfig(hidden_size=64, num_attention_heads=4)
        assert config.latent_dim == 64

    def test_attention_dropout_cascades_from_dropout(self):
        config = MutableConfig(dropout=0.2)
        assert config.attention_dropout == 0.2
        assert config.hidden_dropout == 0.2

    def test_attention_dropout_explicit(self):
        config = MutableConfig(dropout=0.2, attention_dropout=0.1)
        assert config.attention_dropout == 0.1
        assert config.hidden_dropout == 0.2

    def test_type_conversions(self):
        config = MutableConfig(hidden_size="64", num_attention_heads="4")
        assert config.hidden_size == 64
        assert isinstance(config.hidden_size, int)
        assert config.num_attention_heads == 4

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Invalid activation"):
            MutableConfig(activation="badact")

    def test_invalid_position_embedding_raises(self):
        with pytest.raises(ValueError, match="Invalid position_embedding_type"):
            MutableConfig(position_embedding_type="badpos")

    def test_hidden_size_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            MutableConfig(hidden_size=65, num_attention_heads=4)

    def test_special_token_ids(self):
        config = MutableConfig()
        assert config.bos_token_id == 0
        assert config.pad_token_id == 1
        assert config.eos_token_id == 2
        assert config.sep_token_id == 29
        assert config.mask_token_id == 31

    def test_serialization_roundtrip(self):
        config = MutableConfig(hidden_size=64, num_attention_heads=4)
        d = config.to_dict()
        config2 = MutableConfig(**d)
        assert config2.hidden_size == 64
        assert config2.num_attention_heads == 4
