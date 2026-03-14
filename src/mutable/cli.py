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
@click.option(
    "--checkpoint",
    "-p",
    type=click.Path(exists=True),
    required=True,
    help="Trained flow matching checkpoint directory.",
)
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True),
    default=None,
    help="CSV file with germline sequences.",
)
@click.option(
    "--sequences",
    "-s",
    multiple=True,
    help="Direct heavy:light sequence pairs (can be repeated).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output CSV path (default: stdout).",
)
@click.option("--mu", type=float, default=1.0, help="Mutation intensity.")
@click.option(
    "--solver",
    type=click.Choice(["euler", "rk4", "adaptive"]),
    default=None,
    help="ODE solver (default: from config).",
)
@click.option("--steps", type=int, default=None, help="ODE integration steps (default: from config).")
@click.option("--batch-size", type=int, default=32, help="Generation batch size.")
@click.option("--device", type=str, default=None, help="Device: cpu/cuda (default: auto).")
@click.option(
    "--germline-heavy-col",
    type=str,
    default="germline_heavy",
    help="CSV column name for germline heavy chain.",
)
@click.option(
    "--germline-light-col",
    type=str,
    default="germline_light",
    help="CSV column name for germline light chain.",
)
def generate(
    checkpoint: str,
    input_file: str | None,
    sequences: tuple[str, ...],
    output: str | None,
    mu: float,
    solver: str | None,
    steps: int | None,
    batch_size: int,
    device: str | None,
    germline_heavy_col: str,
    germline_light_col: str,
) -> None:
    """Generate mutated antibody sequences using a trained flow matching model.

    Provide input via --input (CSV) or --sequences (heavy:light pairs).

    Examples:

        mutable generate -p outputs/flow/final -i germlines.csv

        mutable generate -p checkpoint/ -s "EVQLVESGG:DIQMTQSPS" --mu 0.5

        mutable generate -p checkpoint/ -i germlines.csv -o mutated.csv --solver rk4 --steps 200
    """
    import csv
    import sys

    import torch

    from .config import FlowMatchingConfig, MutableConfig
    from .models import MutableFlowMatching
    from .tokenizer import MutableTokenizer

    checkpoint_path = Path(checkpoint)

    if not input_file and not sequences:
        raise click.UsageError("Provide input via --input or --sequences.")

    # --- Load model ---
    model_config = MutableConfig.from_pretrained(checkpoint_path)

    # Load flow config — try checkpoint dir, fall back to defaults
    flow_config_path = checkpoint_path / "config.json"
    # FlowMatchingConfig may be saved alongside or we use defaults
    try:
        flow_config = FlowMatchingConfig.from_pretrained(checkpoint_path)
    except Exception:
        flow_config = FlowMatchingConfig()

    model = MutableFlowMatching.from_pretrained(
        checkpoint_path, flow_config=flow_config
    )
    tokenizer = MutableTokenizer.from_pretrained(checkpoint_path)

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # --- Collect input sequences ---
    pairs: list[tuple[str, str]] = []

    if input_file:
        with open(input_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                heavy = row[germline_heavy_col]
                light = row[germline_light_col]
                pairs.append((heavy, light))

    for seq_str in sequences:
        if ":" not in seq_str:
            raise click.UsageError(
                f"Sequence must be 'heavy:light' format, got: {seq_str}"
            )
        heavy, light = seq_str.split(":", 1)
        pairs.append((heavy, light))

    if not pairs:
        raise click.UsageError("No input sequences found.")

    # --- Generate in batches ---
    sep_token = "<sep>"
    results: list[tuple[str, str]] = []

    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start : batch_start + batch_size]

        # Tokenize
        seqs = [f"{h}{sep_token}{l}" for h, l in batch_pairs]
        encoding = tokenizer(
            seqs,
            truncation=True,
            max_length=model_config.max_position_embeddings
            if hasattr(model_config, "max_position_embeddings")
            else 512,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        mu_tensor = torch.full(
            (len(batch_pairs),), mu, dtype=torch.float32, device=device
        )

        # Generate
        output_ids = model.generate(
            germline_input_ids=input_ids,
            germline_attention_mask=attention_mask,
            mu=mu_tensor,
            num_steps=steps,
            solver=solver,
        )

        # Decode and split on <sep>
        for ids in output_ids:
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            # Strip BOS/EOS and split on <sep>
            decoded = decoded.replace("<cls>", "").replace("<eos>", "").replace("<pad>", "").strip()
            if sep_token in decoded:
                heavy_out, light_out = decoded.split(sep_token, 1)
            else:
                heavy_out = decoded
                light_out = ""
            results.append((heavy_out.strip(), light_out.strip()))

    # --- Write output ---
    if output:
        out_f = open(output, "w", newline="")
    else:
        out_f = sys.stdout

    try:
        writer = csv.writer(out_f)
        writer.writerow(["mutated_heavy", "mutated_light"])
        for heavy_out, light_out in results:
            writer.writerow([heavy_out, light_out])
    finally:
        if output:
            out_f.close()


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
