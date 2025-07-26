#!/usr/bin/env python3
"""
Create NHANES Scaler Statistics

This script loads the NHANES training data and computes the StandardScaler
statistics needed for production normalization.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_nhanes_data():
    """Load NHANES training data."""
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")

    if not cache_path.exists():
        logger.error(f"NHANES data not found at {cache_path}")
        logger.info("Run the PAT training scripts to generate this file")
        return None, None

    logger.info(f"Loading NHANES data from {cache_path}")
    data = np.load(cache_path)

    X_train = data['X_train']
    logger.info(f"Loaded training data: {X_train.shape}")

    return X_train, data


def compute_scaler_stats(X_train):
    """Compute StandardScaler statistics."""
    logger.info("Computing StandardScaler statistics...")

    # Flatten to 2D for sklearn
    n_samples = X_train.shape[0]
    X_flat = X_train.reshape(n_samples, -1)

    # Fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_flat)

    # Get statistics
    mean = scaler.mean_.astype(np.float32)
    std = scaler.scale_.astype(np.float32)  # scale_ is the std

    logger.info(f"Mean shape: {mean.shape}")
    logger.info(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    logger.info(f"Std shape: {std.shape}")
    logger.info(f"Std range: [{std.min():.4f}, {std.max():.4f}]")

    # Check for zero std
    zero_std_count = np.sum(std == 0)
    if zero_std_count > 0:
        logger.warning(f"Found {zero_std_count} features with zero std")
        std[std == 0] = 1.0  # Replace with 1 to avoid division by zero

    return mean, std


def save_scaler_stats(mean, std, output_path):
    """Save scaler statistics to JSON."""
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "shape": list(mean.shape),
        "dataset": "NHANES 2013-2014",
        "n_samples": -1,  # Will be filled later
        "created_by": "create_nhanes_scaler_stats.py",
        "notes": "StandardScaler statistics from PAT training data"
    }

    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved scaler statistics to {output_path}")


def verify_stats(stats_path):
    """Verify the saved statistics."""
    with open(stats_path) as f:
        loaded_stats = json.load(f)

    logger.info("\nVerifying saved statistics:")
    logger.info(f"  Shape: {loaded_stats['shape']}")
    logger.info(f"  Dataset: {loaded_stats['dataset']}")

    # Check a few values
    mean_sample = loaded_stats['mean'][:5]
    std_sample = loaded_stats['std'][:5]
    logger.info(f"  Mean sample: {mean_sample}")
    logger.info(f"  Std sample: {std_sample}")


def main():
    """Create NHANES scaler statistics."""
    logger.info("Creating NHANES Scaler Statistics")
    logger.info("=" * 60)

    # Load data
    X_train, data = load_nhanes_data()
    if X_train is None:
        return

    # Update sample count
    n_samples = X_train.shape[0]
    logger.info(f"Training samples: {n_samples}")

    # Compute statistics
    mean, std = compute_scaler_stats(X_train)

    # Save statistics
    output_path = Path("model_weights/production/nhanes_scaler_stats.json")
    save_scaler_stats(mean, std, output_path)

    # Update with sample count
    with open(output_path) as f:
        stats = json.load(f)
    stats['n_samples'] = int(n_samples)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Verify
    verify_stats(output_path)

    logger.info("\n✅ Successfully created NHANES scaler statistics!")
    logger.info("The ProductionPATLoader will now use proper normalization.")


if __name__ == "__main__":
    main()
