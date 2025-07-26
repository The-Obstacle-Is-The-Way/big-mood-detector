"""
NHANES Normalizer

Normalizes activity data using NHANES 2013-2014 statistics.
Critical for reproducing the paper's depression classification results.

Without proper normalization: Random predictions (AUC ~0.5)
With proper normalization: Clinical-grade predictions (AUC 0.5929)
"""

import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class NHANESNormalizer:
    """
    Normalizes activity sequences using NHANES population statistics.

    Uses StandardScaler approach as specified in the PAT paper:
    "All train, test, and validation sets were standardized separately
    using Sklearn's StandardScaler"
    """

    def __init__(self, stats_path: Path | None = None):
        """
        Initialize normalizer with saved statistics.

        Args:
            stats_path: Path to JSON file with mean/std statistics.
                       Defaults to production stats.
        """
        if stats_path is None:
            stats_path = Path("model_weights/production/nhanes_scaler_stats.json")

        self.stats_path = stats_path
        self.mean: NDArray[np.float32] | None = None
        self.std: NDArray[np.float32] | None = None
        self.fitted = False

        # Try to load existing statistics
        if self.stats_path.exists():
            self._load_statistics()
        else:
            logger.warning(
                f"No statistics found at {stats_path}. "
                "Call fit() with training data or load pre-computed stats."
            )

    def _load_statistics(self) -> None:
        """Load mean and std from JSON file."""
        try:
            with open(self.stats_path) as f:
                stats = json.load(f)

            self.mean = np.array(stats["mean"], dtype=np.float32)
            self.std = np.array(stats["std"], dtype=np.float32)

            # Handle zero std (constant features)
            self.std = np.where(self.std == 0, 1.0, self.std)

            self.fitted = True
            logger.info(f"Loaded NHANES statistics from {self.stats_path}")

            # Log metadata if available
            if "n_samples" in stats:
                logger.info(f"Statistics computed from {stats['n_samples']} samples")
            if "dataset" in stats:
                logger.info(f"Dataset: {stats['dataset']}")

        except Exception as e:
            logger.error(f"Failed to load statistics: {e}")
            raise

    def fit(self, X: NDArray[np.float32]) -> None:
        """
        Compute normalization statistics from training data.

        Args:
            X: Training data of shape (n_samples, 10080)
        """
        if X.shape[1] != 10080:
            raise ValueError(f"Expected 10080 timesteps, got {X.shape[1]}")

        # Compute statistics
        self.mean = np.mean(X, axis=0, dtype=np.float32)
        self.std = np.std(X, axis=0, dtype=np.float32, ddof=0)  # Population std

        # Handle zero std
        # Type assertion for mypy - we just assigned self.std above
        assert self.std is not None
        self.std = np.where(self.std == 0, 1.0, self.std)

        self.fitted = True
        logger.info(f"Fitted normalizer on {X.shape[0]} samples")

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize activity sequence(s).

        Args:
            X: Activity data of shape (10080,) or (n_samples, 10080)

        Returns:
            Normalized data with same shape as input
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Load statistics or call fit() first.")

        # Type assertion for mypy
        assert self.mean is not None and self.std is not None

        # Handle single sequence
        if X.ndim == 1:
            if X.shape[0] != 10080:
                raise ValueError(f"Expected 10080 timesteps, got {X.shape[0]}")
            return ((X - self.mean) / self.std).astype(np.float32)

        # Handle batch
        if X.ndim == 2:
            if X.shape[1] != 10080:
                raise ValueError(f"Expected 10080 timesteps, got {X.shape[1]}")
            return ((X - self.mean) / self.std).astype(np.float32)

        raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D")

    def transform_batch(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize batch of sequences.

        Alias for transform() that explicitly handles batches.

        Args:
            X: Batch of sequences (n_samples, 10080)

        Returns:
            Normalized batch
        """
        return self.transform(X)

    def save_statistics(self, path: str | Path) -> None:
        """
        Save computed statistics to JSON.

        Args:
            path: Output path for statistics JSON
        """
        if not self.fitted:
            raise ValueError("No statistics to save. Fit the normalizer first.")

        # Type assertion for mypy
        assert self.mean is not None and self.std is not None

        stats = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "shape": list(self.mean.shape),
            "dataset": "NHANES 2013-2014",  # Can be parameterized
        }

        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved normalization statistics to {path}")
