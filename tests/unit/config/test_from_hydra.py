# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for Hydra → HuggingFace config conversion utilities."""

import pytest
from omegaconf import OmegaConf

from mutable.config import MutableConfig, FlowMatchingConfig
from mutable.config.from_hydra import (
    _filter_to_known_params,
    mutable_config_from_dictconfig,
    flow_config_from_dictconfig,
    training_args_from_dictconfig,
)


class TestFilterToKnownParams:
    def test_unknown_keys_filtered(self):
        cfg = OmegaConf.create({"hidden_size": 64, "bogus_key": 99})
        result = _filter_to_known_params(cfg, MutableConfig)
        assert "bogus_key" not in result
        assert result["hidden_size"] == 64

    def test_known_keys_passed_through(self):
        cfg = OmegaConf.create({"hidden_size": 128, "num_encoder_layers": 3})
        result = _filter_to_known_params(cfg, MutableConfig)
        assert result["hidden_size"] == 128
        assert result["num_encoder_layers"] == 3

    def test_exclude_set_respected(self):
        cfg = OmegaConf.create({"hidden_size": 64, "vocab_size": 32})
        result = _filter_to_known_params(cfg, MutableConfig, exclude={"vocab_size"})
        assert "vocab_size" not in result
        assert result["hidden_size"] == 64


class TestMutableConfigFromDictConfig:
    def test_basic_conversion(self):
        cfg = OmegaConf.create({
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
        })
        config = mutable_config_from_dictconfig(cfg)
        assert isinstance(config, MutableConfig)
        assert config.hidden_size == 64

    def test_unknown_keys_ignored(self):
        cfg = OmegaConf.create({
            "hidden_size": 64,
            "num_attention_heads": 4,
            "not_a_real_key": "hello",
        })
        config = mutable_config_from_dictconfig(cfg)
        assert config.hidden_size == 64

    def test_defaults_fire_normally(self):
        cfg = OmegaConf.create({"hidden_size": 64, "num_attention_heads": 4})
        config = mutable_config_from_dictconfig(cfg)
        assert config.intermediate_size == 256  # 64 * 4


class TestFlowConfigFromDictConfig:
    def test_basic_conversion(self):
        model_config = MutableConfig(hidden_size=64, num_attention_heads=4, latent_dim=64)
        flow_cfg = OmegaConf.create({
            "hidden_size": 64,
            "num_layers": 2,
            "num_attention_heads": 4,
        })
        config = flow_config_from_dictconfig(flow_cfg, model_config)
        assert isinstance(config, FlowMatchingConfig)
        assert config.hidden_size == 64

    def test_hidden_size_none_resolved_from_model(self):
        model_config = MutableConfig(hidden_size=64, num_attention_heads=4, latent_dim=64)
        flow_cfg = OmegaConf.create({
            "hidden_size": None,
            "num_attention_heads": 4,
        })
        config = flow_config_from_dictconfig(flow_cfg, model_config)
        assert config.hidden_size == 64

    def test_num_heads_adapted_when_incompatible(self):
        model_config = MutableConfig(hidden_size=64, num_attention_heads=4, latent_dim=64)
        flow_cfg = OmegaConf.create({
            "hidden_size": 64,
            "num_attention_heads": 20,  # doesn't divide 64
        })
        config = flow_config_from_dictconfig(flow_cfg, model_config)
        assert config.hidden_size % config.num_attention_heads == 0


class TestTrainingArgsFromDictConfig:
    def test_basic_conversion(self, tmp_path):
        cfg = OmegaConf.create({
            "max_steps": 10,
            "per_device_train_batch_size": 4,
            "report_to": "none",
        })
        args = training_args_from_dictconfig(cfg, str(tmp_path))
        assert args.max_steps == 10
        assert args.per_device_train_batch_size == 4

    def test_output_dir_overridden(self, tmp_path):
        cfg = OmegaConf.create({
            "output_dir": "/wrong/path",
            "report_to": "none",
        })
        args = training_args_from_dictconfig(cfg, str(tmp_path))
        assert args.output_dir == str(tmp_path)

    def test_extra_keys_ignored(self, tmp_path):
        cfg = OmegaConf.create({
            "max_steps": 5,
            "pretrained_checkpoint": "/some/path",
            "freeze_backbone": True,
            "report_to": "none",
        })
        args = training_args_from_dictconfig(cfg, str(tmp_path))
        assert args.max_steps == 5
