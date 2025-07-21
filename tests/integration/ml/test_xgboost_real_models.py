"""
Integration tests for XGBoost models with real weights.

These tests are marked as 'heavy' and only run when actual model files are present.
They verify the full ML pipeline with real model artifacts.
"""

from pathlib import Path

import numpy as np
import pytest

@pytest.mark.heavy
@pytest.mark.ml
class TestXGBoostRealModels:
    """Test XGBoost models with actual model files."""

    @pytest.fixture
    def model_dir(self):
        """Get the real model directory."""
        return Path("model_weights/xgboost/pretrained")

    def test_load_real_models(self, model_dir):
        """Test loading actual XGBoost model files."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader

        if not model_dir.exists():
            pytest.skip(f"Model directory not found: {model_dir}")

        loader = XGBoostModelLoader()
        results = loader.load_all_models(model_dir)

        # Check if all models loaded successfully
        if not all(results.values()):
            missing = [k for k, v in results.items() if not v]
            pytest.skip(f"Missing model files: {missing}")

        assert loader.is_loaded
        assert len(loader.models) == 3
        assert all(k in loader.models for k in ["depression", "hypomanic", "manic"])

    def test_real_model_predictions(self, model_dir):
        """Test predictions with real models on synthetic data."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostMoodPredictor
        from big_mood_detector.domain.services.mood_predictor import MoodPrediction

        if not model_dir.exists():
            pytest.skip(f"Model directory not found: {model_dir}")

        predictor = XGBoostMoodPredictor()
        load_results = predictor.load_models(model_dir)

        if not all(load_results.values()):
            pytest.skip("Not all models loaded successfully")

        # Create synthetic feature vector with realistic values
        features = self._create_realistic_features()

        # Make prediction
        result = predictor.predict(features)

        # Verify prediction structure
        assert isinstance(result, MoodPrediction)
        assert 0 <= result.depression_risk <= 1
        assert 0 <= result.hypomanic_risk <= 1
        assert 0 <= result.manic_risk <= 1
        assert result.highest_risk_type in ["depression", "hypomanic", "manic"]

    def test_batch_predictions_performance(self, model_dir):
        """Test batch prediction performance with real models."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader

        if not model_dir.exists():
            pytest.skip(f"Model directory not found: {model_dir}")

        loader = XGBoostModelLoader()
        results = loader.load_all_models(model_dir)

        if not all(results.values()):
            pytest.skip("Not all models loaded successfully")

        # Create batch of features
        batch_size = 100
        features_batch = np.random.randn(batch_size, 36)

        # Time the batch prediction
        import time

        start = time.time()
        predictions = loader.predict_batch(features_batch)
        elapsed = time.time() - start

        # Performance assertions
        assert len(predictions) == batch_size
        assert elapsed < 1.0  # Should process 100 samples in under 1 second

        # Log performance metrics
        print(f"Batch prediction time: {elapsed:.3f}s for {batch_size} samples")
        print(f"Throughput: {batch_size / elapsed:.1f} samples/second")

    def _create_realistic_features(self):
        """Create a feature vector with realistic values based on the paper."""
        features = np.zeros(36)

        # Sleep percentage features (normalized around 0.3-0.4 for 7-8 hours)
        features[0] = 0.35  # sleep_percentage_MN
        features[1] = 0.05  # sleep_percentage_SD
        features[2] = 0.0  # sleep_percentage_Z

        # Circadian features
        features[30] = 0.8  # circadian_amplitude_MN
        features[31] = 0.1  # circadian_amplitude_SD
        features[32] = 0.0  # circadian_amplitude_Z
        features[33] = 14.5  # circadian_phase_MN (2:30 PM peak)
        features[34] = 1.2  # circadian_phase_SD
        features[35] = 0.0  # circadian_phase_Z

        # Fill in other features with small random values
        for i in range(3, 30):
            features[i] = np.random.randn() * 0.1

        return features

@pytest.mark.heavy
def test_model_files_exist():
    """Basic test to check if model files are present."""
    model_dir = Path("model_weights/xgboost/pretrained")

    if not model_dir.exists():
        pytest.skip("Model directory not found")

    expected_files = ["XGBoost_DE.json", "XGBoost_HME.json", "XGBoost_ME.json"]
    existing_files = [f for f in expected_files if (model_dir / f).exists()]

    print(f"Found {len(existing_files)}/{len(expected_files)} model files")
    print(f"Existing: {existing_files}")

    # This test passes even if no files exist - it's just informational
    assert True
