"""
Test NHANES Normalizer

Tests for normalizing activity data using NHANES statistics.
Critical for reproducing the paper's results!

Without proper normalization, we get random predictions.
With it, we unlock the power of temporal mood assessment.
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest


class TestNHANESNormalizer:
    """Test the normalizer that makes our predictions scientifically valid."""

    def test_nhanes_normalizer_class_exists(self):
        """The gateway to proper data preprocessing."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        assert NHANESNormalizer is not None

    def test_initializes_with_default_stats_path(self):
        """Should use production stats by default."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        normalizer = NHANESNormalizer()

        expected_path = Path("model_weights/production/nhanes_scaler_stats.json")
        assert normalizer.stats_path == expected_path

    def test_loads_statistics_from_json(self):
        """Should load mean and std from saved statistics."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        # Mock statistics file
        mock_stats = {
            "mean": [1.5] * 10080,  # 7 days of minute-level means
            "std": [0.5] * 10080,   # 7 days of minute-level stds
            "n_samples": 3077,
            "dataset": "NHANES 2013-2014"
        }

        mock_json_data = json.dumps(mock_stats)

        with patch("builtins.open", mock_open(read_data=mock_json_data)):
            with patch("pathlib.Path.exists", return_value=True):
                normalizer = NHANESNormalizer()

                assert hasattr(normalizer, 'mean')
                assert hasattr(normalizer, 'std')
                assert normalizer.mean.shape == (10080,)
                assert normalizer.std.shape == (10080,)
                assert normalizer.fitted is True

    def test_transform_applies_standardization(self):
        """Transform should apply (X - mean) / std."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        # Create normalizer with known stats
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps({
                "mean": [2.0] * 10080,
                "std": [0.5] * 10080
            }))):
                normalizer = NHANESNormalizer()

        # Test data
        X = np.array([3.0] * 10080, dtype=np.float32)  # All 3.0

        # Transform
        X_normalized = normalizer.transform(X)

        # (3.0 - 2.0) / 0.5 = 2.0
        expected = np.array([2.0] * 10080, dtype=np.float32)
        np.testing.assert_array_almost_equal(X_normalized, expected)

    def test_transform_raises_if_not_fitted(self):
        """Should raise error if trying to transform before fitting."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        with patch("pathlib.Path.exists", return_value=False):
            normalizer = NHANESNormalizer()

            X = np.random.randn(10080).astype(np.float32)

            with pytest.raises(ValueError, match="Normalizer not fitted"):
                normalizer.transform(X)

    def test_validates_input_shape(self):
        """Should validate input has 10080 timesteps."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps({
                "mean": [0.0] * 10080,
                "std": [1.0] * 10080
            }))):
                normalizer = NHANESNormalizer()

        # Wrong shape
        X_wrong = np.random.randn(5000).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 10080 timesteps"):
            normalizer.transform(X_wrong)

    def test_handles_batch_normalization(self):
        """Should handle batch of sequences (N, 10080)."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps({
                "mean": [1.0] * 10080,
                "std": [2.0] * 10080
            }))):
                normalizer = NHANESNormalizer()

        # Batch of 3 sequences
        X_batch = np.ones((3, 10080), dtype=np.float32) * 5.0

        X_normalized = normalizer.transform_batch(X_batch)

        assert X_normalized.shape == (3, 10080)
        # (5.0 - 1.0) / 2.0 = 2.0
        expected = np.ones((3, 10080), dtype=np.float32) * 2.0
        np.testing.assert_array_almost_equal(X_normalized, expected)

    def test_preserves_float32_dtype(self):
        """Should maintain float32 for PyTorch compatibility."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps({
                "mean": [0.0] * 10080,
                "std": [1.0] * 10080
            }))):
                normalizer = NHANESNormalizer()

        X = np.random.randn(10080).astype(np.float64)  # float64 input
        X_normalized = normalizer.transform(X)

        assert X_normalized.dtype == np.float32  # Should convert to float32

    def test_fit_computes_statistics_from_data(self):
        """Should compute mean and std from training data if needed."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        with patch("pathlib.Path.exists", return_value=False):
            normalizer = NHANESNormalizer()

            # Training data with known statistics
            X_train = np.array([
                [1.0] * 10080,
                [3.0] * 10080,
                [5.0] * 10080,
            ], dtype=np.float32)

            normalizer.fit(X_train)

            assert normalizer.fitted is True
            # Mean should be 3.0 for all timesteps
            np.testing.assert_array_almost_equal(
                normalizer.mean,
                np.array([3.0] * 10080, dtype=np.float32)
            )
            # Std should be sqrt(8/3) ≈ 1.633
            expected_std = np.std(X_train, axis=0, ddof=0)
            np.testing.assert_array_almost_equal(
                normalizer.std,
                expected_std
            )

    def test_save_statistics(self):
        """Should save computed statistics to JSON."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        normalizer = NHANESNormalizer()
        normalizer.mean = np.array([1.0] * 10080, dtype=np.float32)
        normalizer.std = np.array([0.5] * 10080, dtype=np.float32)
        normalizer.fitted = True

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            normalizer.save_statistics("test_stats.json")

        # Check that json.dump was called
        handle = mock_file()
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        saved_data = json.loads(written_data)

        assert "mean" in saved_data
        assert "std" in saved_data
        assert len(saved_data["mean"]) == 10080
        assert len(saved_data["std"]) == 10080

    @pytest.mark.parametrize("mean,std,input_val,expected", [
        (0.0, 1.0, 0.0, 0.0),      # No change
        (5.0, 2.0, 9.0, 2.0),      # (9-5)/2 = 2
        (10.0, 0.5, 11.0, 2.0),    # (11-10)/0.5 = 2
        (-2.0, 1.0, -2.0, 0.0),    # Negative mean
    ])
    def test_normalization_formula(self, mean, std, input_val, expected):
        """Test the standardization formula with various values."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps({
                "mean": [mean] * 10080,
                "std": [std] * 10080
            }))):
                normalizer = NHANESNormalizer()

        X = np.full(10080, input_val, dtype=np.float32)
        X_normalized = normalizer.transform(X)

        expected_array = np.full(10080, expected, dtype=np.float32)
        np.testing.assert_array_almost_equal(X_normalized, expected_array)

    def test_handles_zero_std_gracefully(self):
        """Should handle zero std (constant features) without division by zero."""
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps({
                "mean": [5.0] * 10080,
                "std": [0.0] * 10080  # Zero std!
            }))):
                normalizer = NHANESNormalizer()

        X = np.full(10080, 5.0, dtype=np.float32)

        # Should not raise division by zero
        X_normalized = normalizer.transform(X)

        # With zero std, output should be 0 (or handle specially)
        assert not np.any(np.isnan(X_normalized))
        assert not np.any(np.isinf(X_normalized))
