"""
Big Mood Detector CLI

Main command-line interface following Clean Architecture principles.
All commands are properly organized in the interfaces layer.
"""

import click

from big_mood_detector.interfaces.cli.commands import predict_command, process_command
from big_mood_detector.interfaces.cli.labeling import unified_label_group as label_group
from big_mood_detector.interfaces.cli.server import serve_command
from big_mood_detector.interfaces.cli.watch import watch_command


@click.group()
@click.version_option()
def cli() -> None:
    """
    Big Mood Detector - Clinical mood prediction from health data.

    Analyze Apple Health data to detect patterns associated with
    mood episodes using validated ML models.
    """
    pass


# Register all commands
cli.add_command(process_command)
cli.add_command(predict_command)
cli.add_command(serve_command)
cli.add_command(watch_command)
cli.add_command(label_group)


@cli.command(name="train")
@click.option(
    "--model-type",
    type=click.Choice(["xgboost", "pat"]),
    default="xgboost",
    help="Model type",
)
@click.option("--user-id", required=True, help="User identifier for the personal model")
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Training data (CSV for xgboost or NPY for PAT)",
)
@click.option(
    "--labels",
    type=click.Path(exists=True),
    required=True,
    help="Ground truth labels (CSV or NPY)",
)
def train_command(model_type: str, user_id: str, data: str, labels: str) -> None:
    """Train a personalized model using ``PersonalCalibrator``."""
    import numpy as np
    import pandas as pd

    # Check if PAT is requested but not available
    if model_type.lower() == "pat":
        from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE
        if not PAT_AVAILABLE:
            raise click.BadParameter(
                "PAT model requires PyTorch which is not installed. "
                "Install with: pip install torch transformers"
            )

    from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
        PersonalCalibrator,
    )

    click.echo(f"Training {model_type} model for user {user_id}...")

    calibrator = PersonalCalibrator(user_id=user_id, model_type=model_type)

    if labels.endswith(".npy"):
        y = np.load(labels)
    else:
        y = pd.read_csv(labels).iloc[:, 0].to_numpy()

    if model_type == "xgboost":
        features = pd.read_csv(data)
        metrics = calibrator.calibrate(features=features, labels=y)
    else:
        sequences = np.load(data)
        metrics = calibrator.calibrate(sequences=sequences, labels=y)

    calibrator.save_model(metrics)
    click.echo("✅ Personal model saved")


if __name__ == "__main__":
    cli()
