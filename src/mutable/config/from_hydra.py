# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Conversion utilities: Hydra DictConfig → HuggingFace PretrainedConfig objects."""

from __future__ import annotations

import inspect
from typing import Any

from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from .mutable_config import MutableConfig
from .flow_config import FlowMatchingConfig

__all__ = [
    "mutable_config_from_dictconfig",
    "flow_config_from_dictconfig",
    "training_args_from_dictconfig",
]


def _filter_to_known_params(
    cfg: DictConfig, cls: type, exclude: set[str] | None = None
) -> dict[str, Any]:
    """Extract only the keys from cfg that match constructor params of cls.

    Parameters
    ----------
    cfg
        Hydra DictConfig section (e.g. cfg.model).
    cls
        Target class whose __init__ signature defines valid params.
    exclude
        Param names to skip.

    Returns
    -------
    dict
        Filtered keyword arguments suitable for ``cls(**kwargs)``.
    """
    sig = inspect.signature(cls.__init__)
    valid_params = {
        name
        for name, p in sig.parameters.items()
        if name not in ("self", "kwargs")
    }
    if exclude:
        valid_params -= exclude

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return {k: v for k, v in cfg_dict.items() if k in valid_params}


def mutable_config_from_dictconfig(model_cfg: DictConfig) -> MutableConfig:
    """Convert a Hydra ``model`` section to a :class:`MutableConfig`.

    All defaulting logic (``intermediate_size``, ``latent_dim``, etc.) fires
    normally inside ``MutableConfig.__init__`` when values are ``None``.

    Parameters
    ----------
    model_cfg
        The ``cfg.model`` DictConfig.

    Returns
    -------
    MutableConfig
    """
    kwargs = _filter_to_known_params(model_cfg, MutableConfig)
    return MutableConfig(**kwargs)


def flow_config_from_dictconfig(
    flow_cfg: DictConfig, model_config: MutableConfig
) -> FlowMatchingConfig:
    """Convert a Hydra ``flow`` section to a :class:`FlowMatchingConfig`.

    Resolves ``flow.hidden_size = None`` → ``model_config.latent_dim`` so that
    the flow network dimension matches the bottleneck output.

    Parameters
    ----------
    flow_cfg
        The ``cfg.flow`` DictConfig.
    model_config
        Already-constructed :class:`MutableConfig` (needed for ``latent_dim``).

    Returns
    -------
    FlowMatchingConfig
    """
    kwargs = _filter_to_known_params(flow_cfg, FlowMatchingConfig)

    # resolve hidden_size: default to model's latent_dim
    if kwargs.get("hidden_size") is None:
        kwargs["hidden_size"] = model_config.latent_dim

    # resolve num_attention_heads to be compatible with hidden_size
    # (only if using the default and it doesn't divide evenly)
    hidden = kwargs["hidden_size"]
    heads = kwargs.get("num_attention_heads", 20)
    if hidden % heads != 0:
        # fall back to model's num_attention_heads
        kwargs["num_attention_heads"] = model_config.num_attention_heads

    return FlowMatchingConfig(**kwargs)


def training_args_from_dictconfig(
    train_cfg: DictConfig, output_dir: str
) -> TrainingArguments:
    """Convert a Hydra ``train`` section to HF :class:`TrainingArguments`.

    Non-TrainingArguments keys (e.g. ``pretrained_checkpoint``,
    ``freeze_backbone``) are silently ignored.

    Parameters
    ----------
    train_cfg
        The ``cfg.train`` DictConfig.
    output_dir
        Resolved output directory (overrides any ``output_dir`` in the config).

    Returns
    -------
    TrainingArguments
    """
    cfg_dict = OmegaConf.to_container(train_cfg, resolve=True)

    # filter to valid TrainingArguments params
    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = {
        name
        for name, p in sig.parameters.items()
        if name not in ("self", "kwargs")
    }
    kwargs = {k: v for k, v in cfg_dict.items() if k in valid_params}

    # always override output_dir
    kwargs["output_dir"] = output_dir

    return TrainingArguments(**kwargs)
