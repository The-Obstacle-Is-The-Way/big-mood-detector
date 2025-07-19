"""
Test cases for XGBoost Model Infrastructure

Tests the loading and inference of XGBoost models for mood prediction.
Following TDD principles - tests written before implementation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class TestXGBoostModels:
    """Test the XGBoost model infrastructure."""

    def test_model_loader_initialization(self):
        """Test basic model loader initialization."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()

        assert loader.models == {}
        assert loader.is_loaded is False
        assert loader.feature_names is not None
        assert len(loader.feature_names) == 36  # 36 features from the paper

    def test_expected_feature_names(self):
        """Test that feature names match the paper specification."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()

        # Check for key features mentioned in the papers
        expected_features = [
            "sleep_percentage_MN",
            "sleep_percentage_SD",
            "sleep_percentage_Z",
            "circadian_amplitude_MN",
            "circadian_phase_MN",
            "long_num_MN",
            "short_num_MN",
        ]

        for feature in expected_features:
            assert feature in loader.feature_names

    @patch("pathlib.Path.exists")
    @patch("joblib.load")
    def test_load_single_model(self, mock_joblib_load, mock_exists):
        """Test loading a single XGBoost model."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        # Mock file exists
        mock_exists.return_value = True

        # Mock loaded model
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
        mock_joblib_load.return_value = mock_model

        loader = XGBoostModelLoader()
        model_path = Path("model_weights/xgboost/pretrained/depression_model.pkl")

        success = loader.load_model("depression", model_path)

        assert success is True
        assert "depression" in loader.models
        mock_joblib_load.assert_called_once_with(model_path)

    def test_load_model_file_not_found(self):
        """Test handling of missing model file."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()
        fake_path = Path("nonexistent/model.pkl")

        success = loader.load_model("depression", fake_path)

        assert success is False
        assert "depression" not in loader.models

    def test_load_all_models(self):
        """Test loading all three mood models."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        # Create a test loader
        loader = XGBoostModelLoader()
        
        # For unit tests, we'll mock the internal load method
        # This follows Eugene Yan's advice - we're testing behavior, not the actual model
        with patch.object(loader, '_load_single_model') as mock_load:
            # Mock successful loads
            mock_load.side_effect = [True, True, True]
            
            # Create mock models
            for mood in ["depression", "hypomanic", "manic"]:
                loader.models[mood] = MagicMock()
            
            model_dir = Path("model_weights/xgboost/pretrained")
            results = loader.load_all_models(model_dir)
            
            # Verify behavior
            assert len(results) == 3
            assert loader.is_loaded is True
            assert len(loader.models) == 3

    def test_predict_without_loaded_models(self):
        """Test that prediction fails gracefully without loaded models."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()
        features = np.random.randn(36)

        with pytest.raises(RuntimeError, match="Models not loaded"):
            loader.predict(features)

    def test_predict_with_loaded_models(self):
        """Test making predictions with loaded models."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()
        
        # Manually set up mock models - following Eugene Yan's advice
        # We're testing the prediction logic, not the actual models
        mock_models = {
            "depression": MagicMock(
                predict_proba=MagicMock(return_value=np.array([[0.3, 0.7]]))  # 70% risk
            ),
            "hypomanic": MagicMock(
                predict_proba=MagicMock(return_value=np.array([[0.8, 0.2]]))  # 20% risk
            ),
            "manic": MagicMock(
                predict_proba=MagicMock(return_value=np.array([[0.9, 0.1]]))  # 10% risk
            ),
        }
        
        # For XGBoost JSON models, we need to mock the predict method differently
        for model_type, mock_model in mock_models.items():
            mock_booster = MagicMock()
            # XGBoost Booster.predict returns raw probabilities
            if model_type == "depression":
                mock_booster.predict.return_value = np.array([0.7])
            elif model_type == "hypomanic":
                mock_booster.predict.return_value = np.array([0.2])
            else:  # manic
                mock_booster.predict.return_value = np.array([0.1])
            loader.models[model_type] = mock_booster
        
        loader.is_loaded = True

        # Make prediction
        features = np.random.randn(36)
        prediction = loader.predict(features)

        assert isinstance(prediction, MoodPrediction)
        assert prediction.depression_risk == 0.7
        assert prediction.hypomanic_risk == 0.2
        assert prediction.manic_risk == 0.1
        assert prediction.highest_risk_type == "depression"
        assert 0 <= prediction.confidence <= 1

    def test_feature_validation(self):
        """Test that feature vector validation works correctly."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()

        # Test wrong number of features
        with pytest.raises(ValueError, match="Expected 36 features"):
            loader._validate_features(np.random.randn(35))

        # Test correct features pass
        features = np.random.randn(36)
        loader._validate_features(features)  # Should not raise

    def test_feature_dict_to_array(self):
        """Test converting feature dictionary to array in correct order."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()

        # Create feature dict
        feature_dict = {name: i for i, name in enumerate(loader.feature_names)}

        # Convert to array
        feature_array = loader.dict_to_array(feature_dict)

        assert len(feature_array) == 36
        assert all(feature_array[i] == i for i in range(36))

    def test_batch_prediction(self):
        """Test making predictions on multiple samples."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()
        
        # Create XGBoost-style mock models that return raw probabilities
        batch_size = 5
        
        # Mock depression model predictions
        depression_mock = MagicMock()
        depression_mock.predict.return_value = np.array([0.7, 0.6, 0.8, 0.5, 0.9])
        
        # Mock hypomanic model predictions  
        hypomanic_mock = MagicMock()
        hypomanic_mock.predict.return_value = np.array([0.2, 0.3, 0.1, 0.4, 0.15])
        
        # Mock manic model predictions
        manic_mock = MagicMock()
        manic_mock.predict.return_value = np.array([0.1, 0.15, 0.05, 0.2, 0.08])
        
        # Set up the mocked models
        loader.models = {
            "depression": depression_mock,
            "hypomanic": hypomanic_mock,
            "manic": manic_mock
        }
        loader.is_loaded = True

        # Make batch prediction
        features_batch = np.random.randn(batch_size, 36)
        predictions = loader.predict_batch(features_batch)

        assert len(predictions) == batch_size
        assert all(isinstance(p, MoodPrediction) for p in predictions)
        assert predictions[2].depression_risk == 0.8  # Third sample

    def test_model_info(self):
        """Test retrieving model information."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostModelLoader,
        )

        loader = XGBoostModelLoader()
        info = loader.get_model_info()

        assert "num_features" in info
        assert "models_loaded" in info
        assert "feature_names" in info
        assert info["num_features"] == 36
        assert len(info["models_loaded"]) == 0  # No models loaded yet


class TestXGBoostMoodPredictor:
    """Test the domain service implementation."""

    def test_mood_predictor_implementation(self):
        """Test that XGBoostMoodPredictor implements the domain interface."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostMoodPredictor,
        )

        # Create predictor
        predictor = XGBoostMoodPredictor()
        
        # Manually set up mock models to test behavior
        # Following Eugene Yan's advice - test the interface, not the models
        mock_models = {
            "depression": MagicMock(predict=MagicMock(return_value=np.array([0.7]))),
            "hypomanic": MagicMock(predict=MagicMock(return_value=np.array([0.2]))),
            "manic": MagicMock(predict=MagicMock(return_value=np.array([0.1])))
        }
        
        # Inject the mocked models
        predictor.model_loader.models = mock_models
        predictor.model_loader.is_loaded = True

        # Test prediction
        features = np.random.randn(36)
        result = predictor.predict(features)

        assert isinstance(result, MoodPrediction)
        assert hasattr(predictor, "is_loaded")
        assert hasattr(predictor, "get_model_info")
        assert result.depression_risk == 0.7
        assert result.hypomanic_risk == 0.2
        assert result.manic_risk == 0.1
