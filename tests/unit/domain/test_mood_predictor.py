"""
Unit tests for Mood Predictor Service
"""

import numpy as np
import pytest

from big_mood_detector.domain.services.mood_predictor import (
    MoodPrediction,
    MoodPredictor,
)


class TestMoodPredictor:
    """Test mood prediction with XGBoost models."""

    @pytest.fixture
    def predictor(self):
        """Create predictor with reference models."""
        return MoodPredictor()

    @pytest.fixture
    def sample_features(self):
        """Create sample 36-feature vector."""
        # Simulate reasonable feature values
        features = np.zeros(36)

        # Sleep features (mean, std, z-score pattern)
        features[0:3] = [0.33, 0.05, 0.5]  # sleep_percentage
        features[3:6] = [0.15, 0.03, -0.2]  # sleep_amplitude
        features[6:9] = [1.0, 0.2, 0.0]  # long_num
        features[9:12] = [8.0, 1.5, 0.3]  # long_len
        features[12:15] = [8.0, 1.5, 0.3]  # long_ST
        features[15:18] = [0.5, 0.1, -0.1]  # long_WT
        features[18:21] = [0.2, 0.1, -0.5]  # short_num
        features[21:24] = [1.5, 0.5, -0.3]  # short_len
        features[24:27] = [1.5, 0.5, -0.3]  # short_ST
        features[27:30] = [0.1, 0.05, 0.0]  # short_WT

        # Circadian features
        features[30:33] = [0.85, 0.1, 0.2]  # circadian_amplitude
        features[33:36] = [22.5, 1.2, 0.1]  # circadian_phase

        return features

    def test_predictor_initialization(self, predictor):
        """Test that predictor loads models."""
        assert predictor.is_loaded

        # Check model info
        info = predictor.get_model_info()
        assert "depression" in info
        assert "hypomanic" in info
        assert "manic" in info

    def test_predict_single_sample(self, predictor, sample_features):
        """Test prediction on single feature vector."""
        if not predictor.is_loaded:
            pytest.skip("Models not available")

        prediction = predictor.predict(sample_features)

        # Check prediction structure
        assert isinstance(prediction, MoodPrediction)
        assert 0 <= prediction.depression_risk <= 1
        assert 0 <= prediction.hypomanic_risk <= 1
        assert 0 <= prediction.manic_risk <= 1
        assert 0 <= prediction.confidence <= 1

        # Check highest risk identification
        assert prediction.highest_risk_type in ["depression", "hypomanic", "manic"]
        assert prediction.highest_risk_value >= 0

    def test_predict_batch(self, predictor, sample_features):
        """Test batch prediction."""
        if not predictor.is_loaded:
            pytest.skip("Models not available")

        # Create batch with variations
        batch = []
        for i in range(3):
            features = sample_features.copy()
            features[0] += i * 0.1  # Vary sleep percentage
            batch.append(features)

        predictions = predictor.predict_batch(batch)

        assert len(predictions) == 3
        assert all(isinstance(p, MoodPrediction) for p in predictions)

    def test_invalid_features(self, predictor):
        """Test handling of invalid features."""
        if not predictor.is_loaded:
            pytest.skip("Models not available")

        # Wrong shape
        with pytest.raises(ValueError, match="Expected 36 features"):
            predictor.predict(np.zeros(10))

        # Test with list instead of array (should convert)
        features_list = [0.5] * 36
        prediction = predictor.predict(features_list)
        assert isinstance(prediction, MoodPrediction)

    def test_prediction_to_dict(self, predictor, sample_features):
        """Test prediction serialization."""
        if not predictor.is_loaded:
            pytest.skip("Models not available")

        prediction = predictor.predict(sample_features)
        pred_dict = prediction.to_dict()

        assert "depression_risk" in pred_dict
        assert "hypomanic_risk" in pred_dict
        assert "manic_risk" in pred_dict
        assert "confidence" in pred_dict
        assert "highest_risk_type" in pred_dict
        assert "highest_risk_value" in pred_dict

        # Check types - some values are strings, others are floats
        for k, v in pred_dict.items():
            if k == "highest_risk_type":
                assert isinstance(v, str)
            else:
                assert isinstance(v, float)

    def test_extreme_features(self, predictor):
        """Test with extreme feature values."""
        if not predictor.is_loaded:
            pytest.skip("Models not available")

        # Very poor sleep pattern
        poor_sleep = np.zeros(36)
        poor_sleep[0] = 0.15  # Very low sleep percentage
        poor_sleep[3] = 0.5  # High sleep fragmentation
        poor_sleep[18] = 3.0  # Many short sleep windows
        poor_sleep[33] = 26.0  # Very late circadian phase

        prediction = predictor.predict(poor_sleep)

        # Should still get valid predictions
        assert 0 <= prediction.depression_risk <= 1
        assert 0 <= prediction.hypomanic_risk <= 1
        assert 0 <= prediction.manic_risk <= 1

    def test_confidence_calculation(self, predictor, sample_features):
        """Test confidence scoring."""
        if not predictor.is_loaded:
            pytest.skip("Models not available")

        # Good quality features
        good_prediction = predictor.predict(sample_features)

        # Poor quality features (many zeros)
        poor_features = np.zeros(36)
        poor_features[0] = 0.3  # Only set a few values
        poor_prediction = predictor.predict(poor_features)

        # Good features should have higher confidence
        assert good_prediction.confidence > poor_prediction.confidence
