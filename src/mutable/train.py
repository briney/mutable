# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Training entry points for denoising pre-training and flow matching."""

from __future__ import annotations

import importlib.resources
from contextlib import ExitStack
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from .config.from_hydra import (
    flow_config_from_dictconfig,
    mutable_config_from_dictconfig,
    training_args_from_dictconfig,
)


def _load_config(
    config: str | None,
    name: str,
    seed: int,
    output_dir: str,
    overrides: list[str] | None,
) -> DictConfig:
    """Load Hydra configuration from YAML files.

    Looks for configs in this order:
    1. Explicit path (file or directory) via ``config``
    2. Local ``configs/`` directory in the current working directory
    3. Bundled configs shipped with the ``mutable`` package

    Parameters
    ----------
    config
        Path to config file (.yaml/.yml) or config directory.
    name
        Experiment name.
    seed
        Random seed.
    output_dir
        Output directory.
    overrides
        List of Hydra config overrides.

    Returns
    -------
    DictConfig
        Loaded and merged configuration.
    """
    with ExitStack() as stack:
        if config is None or config == "configs":
            # check for local configs directory first
            local_configs = Path("configs").absolute()
            if local_configs.exists() and local_configs.is_dir():
                config_dir = local_configs
            else:
                # use bundled configs from package
                config_dir = stack.enter_context(
                    importlib.resources.as_file(
                        importlib.resources.files("mutable").joinpath("configs")
                    )
                )
            config_name = "config"
        else:
            config_path = Path(config).absolute()

            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config path '{config}' does not exist.\n"
                    f"Provide a config file (.yaml) or config directory via --config/-c"
                )

            if config_path.is_file():
                config_dir = config_path.parent
                config_name = config_path.stem
            else:
                config_dir = config_path
                config_name = "config"

        stack.enter_context(
            initialize_config_dir(config_dir=str(config_dir), version_base=None)
        )

        override_list = overrides or []
        override_list.extend([f"name={name}", f"seed={seed}"])
        # only override output_dir if explicitly provided (not default)
        # so that config's output_dir: outputs/${name} interpolation works
        if output_dir != "outputs":
            override_list.append(f"output_dir={output_dir}")

        return compose(config_name=config_name, overrides=override_list)


def run_denoising_training(
    config: str | None = None,
    output_dir: str = "outputs",
    name: str = "mutable_experiment",
    seed: int = 42,
    use_wandb: bool = True,
    overrides: list[str] | None = None,
) -> None:
    """Run Phase 1 denoising pre-training.

    Parameters
    ----------
    config
        Path to config file or config directory.
    output_dir
        Output directory for checkpoints and logs.
    name
        Experiment name.
    seed
        Random seed.
    use_wandb
        Whether to enable WandB logging.
    overrides
        Hydra config overrides. Must include ``data.train`` for training data.
    """
    import datasets as hf_datasets

    from .datasets import DenoisingCollator, DenoisingDataset
    from .masking import InformationWeightedMasker, UniformMasker
    from .models import MutableForDenoising
    from .modules.noise import BartNoiseFunction
    from .tokenizer import MutableTokenizer
    from .trainer import DenoisingTrainer

    # load config
    cfg = _load_config(config, name, seed, output_dir, overrides)
    set_seed(cfg.seed)

    # save resolved config
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path / "config.yaml")

    print(OmegaConf.to_yaml(cfg))

    # build model config
    model_config = mutable_config_from_dictconfig(cfg.model)
    model_config.save_pretrained(output_path)

    # create model
    model = MutableForDenoising(model_config)
    print(f"Model parameters: {model.count_parameters():,}")

    # create tokenizer
    tokenizer = MutableTokenizer()

    # masking setup
    use_weighted_masking = cfg.masking.use_weighted_masking

    # create noise function (only needed in default mode)
    noise_fn = None
    if not use_weighted_masking:
        noise_fn = BartNoiseFunction(
            mask_token_id=model_config.mask_token_id,
            pad_token_id=model_config.pad_token_id,
            vocab_size=model_config.vocab_size,
            mask_ratio=cfg.noise.mask_ratio,
            delete_ratio=cfg.noise.delete_ratio,
            infill_ratio=cfg.noise.infill_ratio,
            poisson_lambda=cfg.noise.poisson_lambda,
            permute_sentences=cfg.noise.permute_sentences,
            random_token_ratio=cfg.noise.random_token_ratio,
            sep_token_id=model_config.sep_token_id,
        )

    # create maskers (only needed in weighted mode)
    masker = None
    uniform_masker = None
    if use_weighted_masking:
        masker = InformationWeightedMasker(
            mask_rate=cfg.masking.mask_rate,
            mask_token_id=model_config.mask_token_id,
            cdr_weight_multiplier=cfg.masking.cdr_weight_multiplier,
            nongermline_weight_multiplier=cfg.masking.nongermline_weight_multiplier,
            v_weight_multiplier=cfg.masking.v_weight_multiplier,
            d_weight_multiplier=cfg.masking.d_weight_multiplier,
            j_weight_multiplier=cfg.masking.j_weight_multiplier,
            n_weight_multiplier=cfg.masking.n_weight_multiplier,
            selection_method=cfg.masking.selection_method,
        )
        uniform_masker = UniformMasker(
            mask_rate=cfg.masking.mask_rate,
            mask_token_id=model_config.mask_token_id,
        )

    # mask column names from data config
    mask_col_kwargs = {}
    if use_weighted_masking:
        mask_col_kwargs = dict(
            heavy_cdr_col=cfg.data.heavy_cdr_col,
            light_cdr_col=cfg.data.light_cdr_col,
            heavy_nongermline_col=cfg.data.heavy_nongermline_col,
            light_nongermline_col=cfg.data.light_nongermline_col,
            heavy_segment_col=cfg.data.heavy_segment_col,
            light_segment_col=cfg.data.light_segment_col,
        )

    # load datasets
    if cfg.data.train is None:
        raise ValueError(
            "Training data not specified. "
            "Use data.train=/path/to/data.csv override."
        )

    train_ds = hf_datasets.load_dataset(
        "csv", data_files=cfg.data.train, split="train"
    )
    train_dataset = DenoisingDataset(
        dataset=train_ds,
        tokenizer=tokenizer,
        noise_fn=noise_fn,
        max_length=cfg.data.max_length,
        heavy_col=cfg.data.heavy_col,
        light_col=cfg.data.light_col,
        use_weighted_masking=use_weighted_masking,
        **mask_col_kwargs,
    )

    eval_dataset = None
    if cfg.data.eval is not None:
        eval_ds = hf_datasets.load_dataset(
            "csv", data_files=cfg.data.eval, split="train"
        )
        eval_dataset = DenoisingDataset(
            dataset=eval_ds,
            tokenizer=tokenizer,
            noise_fn=noise_fn,
            max_length=cfg.data.max_length,
            heavy_col=cfg.data.heavy_col,
            light_col=cfg.data.light_col,
            use_weighted_masking=use_weighted_masking,
            **mask_col_kwargs,
        )

    # collator
    collator = DenoisingCollator(
        pad_token_id=model_config.pad_token_id,
        use_weighted_masking=use_weighted_masking,
    )

    # training args
    training_args = training_args_from_dictconfig(cfg.train, cfg.output_dir)

    # WandB
    if use_wandb and cfg.log.wandb.enabled:
        training_args.report_to = ["wandb"]
        import os

        os.environ.setdefault("WANDB_PROJECT", cfg.log.wandb.project)
        if cfg.log.wandb.entity:
            os.environ.setdefault("WANDB_ENTITY", cfg.log.wandb.entity)
        training_args.run_name = cfg.name
    else:
        training_args.report_to = ["none"]

    # create trainer
    trainer = DenoisingTrainer(
        masker=masker,
        uniform_masker=uniform_masker,
        use_weighted_masking=use_weighted_masking,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # train
    trainer.train()

    # save final model
    trainer.save_model(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))


def run_flow_training(
    config: str | None = None,
    output_dir: str = "outputs",
    name: str = "mutable_experiment",
    pretrained: str | None = None,
    seed: int = 42,
    use_wandb: bool = True,
    overrides: list[str] | None = None,
) -> None:
    """Run Phase 2 flow matching training.

    Parameters
    ----------
    config
        Path to config file or config directory.
    output_dir
        Output directory for checkpoints and logs.
    name
        Experiment name.
    pretrained
        Path to Phase 1 pretrained checkpoint directory.
    seed
        Random seed.
    use_wandb
        Whether to enable WandB logging.
    overrides
        Hydra config overrides. Must include ``data.train`` for training data.
    """
    import torch
    import datasets as hf_datasets

    from .datasets import FlowMatchingDataset
    from .models import MutableFlowMatching
    from .tokenizer import MutableTokenizer
    from .trainer import FlowMatchingTrainer

    # load config
    cfg = _load_config(config, name, seed, output_dir, overrides)
    set_seed(cfg.seed)

    # save resolved config
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path / "config.yaml")

    print(OmegaConf.to_yaml(cfg))

    # build configs
    model_config = mutable_config_from_dictconfig(cfg.model)
    flow_config = flow_config_from_dictconfig(cfg.flow, model_config)
    model_config.save_pretrained(output_path)

    # resolve pretrained checkpoint (CLI arg > config value)
    checkpoint_path = pretrained or cfg.train.get("pretrained_checkpoint")

    # create model
    model = MutableFlowMatching(model_config, flow_config)

    # load Phase 1 weights if provided
    if checkpoint_path is not None:
        print(f"Loading pretrained backbone from {checkpoint_path}")
        state_dict = torch.load(
            Path(checkpoint_path) / "pytorch_model.bin"
            if Path(checkpoint_path).is_dir()
            else checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        # load only backbone weights (mutable.* and lm_head.*)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys (expected for flow network): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    # freeze backbone if configured
    freeze = cfg.train.get("freeze_backbone", True)
    if freeze:
        model.freeze_encoder_decoder()
        print("Frozen encoder/decoder backbone")

    print(f"Total parameters: {model.count_parameters(only_trainable=False):,}")
    print(f"Trainable parameters: {model.count_parameters(only_trainable=True):,}")

    # create tokenizer
    tokenizer = MutableTokenizer()

    # load datasets
    if cfg.data.train is None:
        raise ValueError(
            "Training data not specified. "
            "Use data.train=/path/to/data.csv override."
        )

    train_ds = hf_datasets.load_dataset(
        "csv", data_files=cfg.data.train, split="train"
    )
    train_dataset = FlowMatchingDataset(
        dataset=train_ds,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        germline_heavy_col=cfg.data.germline_heavy_col,
        germline_light_col=cfg.data.germline_light_col,
        mutated_heavy_col=cfg.data.mutated_heavy_col,
        mutated_light_col=cfg.data.mutated_light_col,
        mu_col=cfg.data.mu_col,
    )

    eval_dataset = None
    if cfg.data.eval is not None:
        eval_ds = hf_datasets.load_dataset(
            "csv", data_files=cfg.data.eval, split="train"
        )
        eval_dataset = FlowMatchingDataset(
            dataset=eval_ds,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
            germline_heavy_col=cfg.data.germline_heavy_col,
            germline_light_col=cfg.data.germline_light_col,
            mutated_heavy_col=cfg.data.mutated_heavy_col,
            mutated_light_col=cfg.data.mutated_light_col,
            mu_col=cfg.data.mu_col,
        )

    # training args
    training_args = training_args_from_dictconfig(cfg.train, cfg.output_dir)

    # WandB
    if use_wandb and cfg.log.wandb.enabled:
        training_args.report_to = ["wandb"]
        import os

        os.environ.setdefault("WANDB_PROJECT", cfg.log.wandb.project)
        if cfg.log.wandb.entity:
            os.environ.setdefault("WANDB_ENTITY", cfg.log.wandb.entity)
        training_args.run_name = cfg.name
    else:
        training_args.report_to = ["none"]

    # create trainer
    trainer = FlowMatchingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # train
    trainer.train()

    # save final model
    trainer.save_model(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))
