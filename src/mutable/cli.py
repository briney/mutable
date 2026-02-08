# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Command-line interface for Mutable."""

from __future__ import annotations

from pathlib import Path

import click

from . import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Mutable: generative models for antibody somatic hypermutation."""
    pass


@main.command("train-denoising")
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    default="configs",
    help="Config file (.yaml) or config directory (default: configs)",
)
@click.option(
    "--output-dir", "-o", type=click.Path(), default="outputs", help="Output directory"
)
@click.option("--name", "-n", default="mutable_experiment", help="Experiment name")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--wandb/--no-wandb", default=True, help="Enable/disable WandB")
@click.argument("overrides", nargs=-1)
def train_denoising(
    config: str,
    output_dir: str,
    name: str,
    seed: int,
    wandb: bool,
    overrides: tuple[str, ...],
) -> None:
    """Phase 1: denoising pre-training.

    Training data must be specified in the config file (data.train) or via
    command-line override.

    Examples:

        mutable train-denoising data.train=/path/to/train.csv

        mutable train-denoising -c my_config.yaml data.train=/path/to/train.csv

        mutable train-denoising model=small train=debug data.train=/path/to/train.csv
    """
    from .train import run_denoising_training

    # inject phase-specific defaults (user overrides take precedence)
    override_list = ["data=denoising", "train=denoising"] + list(overrides)

    run_denoising_training(
        config=config,
        output_dir=output_dir,
        name=name,
        seed=seed,
        use_wandb=wandb,
        overrides=override_list,
    )


@main.command("train-flow")
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    default="configs",
    help="Config file (.yaml) or config directory (default: configs)",
)
@click.option(
    "--output-dir", "-o", type=click.Path(), default="outputs", help="Output directory"
)
@click.option("--name", "-n", default="mutable_experiment", help="Experiment name")
@click.option(
    "--pretrained",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Phase 1 pretrained checkpoint path",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--wandb/--no-wandb", default=True, help="Enable/disable WandB")
@click.argument("overrides", nargs=-1)
def train_flow(
    config: str,
    output_dir: str,
    name: str,
    pretrained: str | None,
    seed: int,
    wandb: bool,
    overrides: tuple[str, ...],
) -> None:
    """Phase 2: flow matching training.

    Requires a Phase 1 pretrained checkpoint (--pretrained) or
    train.pretrained_checkpoint in the config.

    Examples:

        mutable train-flow -p outputs/denoising/final data.train=/path/to/pairs.csv

        mutable train-flow -p checkpoint/ train=debug data.train=/path/to/pairs.csv
    """
    from .train import run_flow_training

    # inject phase-specific defaults (user overrides take precedence)
    override_list = ["data=flow", "train=flow"] + list(overrides)

    run_flow_training(
        config=config,
        output_dir=output_dir,
        name=name,
        pretrained=pretrained,
        seed=seed,
        use_wandb=wandb,
        overrides=override_list,
    )


@main.command()
def generate() -> None:
    """Generate mutated antibody sequences (not yet implemented)."""
    raise NotImplementedError(
        "Generation is not yet implemented. "
        "Use MutableFlowMatching.generate() directly for now."
    )


@main.command("model-size")
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    default="configs",
    help="Config file (.yaml) or config directory (default: configs)",
)
@click.argument("overrides", nargs=-1)
def model_size(
    config: str,
    overrides: tuple[str, ...],
) -> None:
    """Print parameter counts for a model configuration.

    Examples:

        mutable model-size

        mutable model-size model=small

        mutable model-size model=large model.hidden_size=768
    """
    import importlib.resources
    from contextlib import ExitStack

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from .config.from_hydra import (
        flow_config_from_dictconfig,
        mutable_config_from_dictconfig,
    )
    from .models import MutableForDenoising, MutableFlowMatching

    with ExitStack() as stack:
        if config is None or config == "configs":
            local_configs = Path("configs").absolute()
            if local_configs.exists() and local_configs.is_dir():
                config_dir = local_configs
            else:
                config_dir = stack.enter_context(
                    importlib.resources.as_file(
                        importlib.resources.files("mutable").joinpath("configs")
                    )
                )
            config_name = "config"
        else:
            config_path = Path(config).absolute()

            if not config_path.exists():
                raise click.ClickException(
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

        cfg = compose(config_name=config_name, overrides=list(overrides))

    # build configs
    model_config = mutable_config_from_dictconfig(cfg.model)
    flow_config = flow_config_from_dictconfig(cfg.flow, model_config)

    click.echo("Model configuration:")
    click.echo(OmegaConf.to_yaml(cfg.model))

    # denoising model
    denoising_model = MutableForDenoising(model_config)
    denoising_params = denoising_model.count_parameters(only_trainable=True)

    click.echo(f"Denoising model (Phase 1): {denoising_params:,} parameters")

    # flow model
    flow_model = MutableFlowMatching(model_config, flow_config)
    total_params = flow_model.count_parameters(only_trainable=False)
    flow_model.freeze_encoder_decoder()
    trainable_params = flow_model.count_parameters(only_trainable=True)

    click.echo(f"Flow model (Phase 2):      {total_params:,} total, {trainable_params:,} trainable")


if __name__ == "__main__":
    main()
