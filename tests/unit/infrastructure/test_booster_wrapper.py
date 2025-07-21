"""
Unit tests for BoosterPredictProbaWrapper.

Tests that the wrapper correctly adds predict_proba functionality to raw Booster objects.
"""

import numpy as np
import pytest


class TestBoosterWrapper:
    """Test the BoosterPredictProbaWrapper."""

    def test_wrapper_can_be_imported(self):
        """Test that the wrapper can be imported."""
        from big_mood_detector.infrastructure.ml_models.booster_wrapper import (
            BoosterPredictProbaWrapper,
        )

        assert BoosterPredictProbaWrapper is not None

    def test_wrapper_predict_proba(self):
        """Test that wrapper provides predict_proba method."""
        from big_mood_detector.infrastructure.ml_models.booster_wrapper import (
            BoosterPredictProbaWrapper,
        )

        # Mock booster that returns raw predictions
        class MockBooster:
            def predict(self, X):
                # Return mock probabilities
                if hasattr(X, 'num_row'):
                    n_samples = X.num_row()
                else:
                    n_samples = len(X)
                return np.array([0.3, 0.7, 0.2, 0.9, 0.5][:n_samples])

        # Create wrapper
        wrapped = BoosterPredictProbaWrapper(MockBooster())

        # Test single prediction
        X_single = [[1, 2, 3]]
        proba = wrapped.predict_proba(X_single)

        assert proba.shape == (1, 2)
        assert abs(proba[0, 0] - 0.7) < 1e-6  # P(class=0) = 1 - 0.3
        assert abs(proba[0, 1] - 0.3) < 1e-6  # P(class=1) = 0.3

        # Test batch prediction
        X_batch = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        proba_batch = wrapped.predict_proba(X_batch)

        assert proba_batch.shape == (3, 2)
        assert abs(proba_batch[0, 1] - 0.3) < 1e-6
        assert abs(proba_batch[1, 1] - 0.7) < 1e-6
        assert abs(proba_batch[2, 1] - 0.2) < 1e-6

        # Check probabilities sum to 1
        assert np.allclose(proba_batch.sum(axis=1), 1.0)

    def test_wrapper_delegates_attributes(self):
        """Test that wrapper delegates other attributes to booster."""
        from big_mood_detector.infrastructure.ml_models.booster_wrapper import (
            BoosterPredictProbaWrapper,
        )

        # Mock booster with custom attribute
        class MockBooster:
            def __init__(self):
                self.custom_attr = "test_value"
                self.feature_names = ["f1", "f2", "f3"]

            def predict(self, X):
                return np.array([0.5])

        wrapped = BoosterPredictProbaWrapper(MockBooster())

        # Check that attributes are accessible
        assert wrapped.custom_attr == "test_value"
        assert wrapped.feature_names == ["f1", "f2", "f3"]

    def test_wrapper_with_dmatrix(self):
        """Test wrapper handles DMatrix input correctly."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost not installed")

        from big_mood_detector.infrastructure.ml_models.booster_wrapper import (
            BoosterPredictProbaWrapper,
        )

        # Mock booster
        class MockBooster:
            def predict(self, X):
                assert isinstance(X, xgb.DMatrix)
                return np.array([0.6, 0.4])

        wrapped = BoosterPredictProbaWrapper(MockBooster())

        # Test with numpy array (should convert to DMatrix)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        proba = wrapped.predict_proba(X)

        assert proba.shape == (2, 2)
        assert abs(proba[0, 1] - 0.6) < 1e-6
        assert abs(proba[1, 1] - 0.4) < 1e-6
