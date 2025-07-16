"""
Test cases for XGBoost Model Infrastructure

Tests the loading and inference of XGBoost models for mood prediction.
Following TDD principles - tests written before implementation.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import joblib

from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class TestXGBoostModels:
    """Test the XGBoost model infrastructure."""

    def test_model_loader_initialization(self):
        """Test basic model loader initialization."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        loader = XGBoostModelLoader()
        
        assert loader.models == {}
        assert loader.is_loaded is False
        assert loader.feature_names is not None
        assert len(loader.feature_names) == 36  # 36 features from the paper

    def test_expected_feature_names(self):
        """Test that feature names match the paper specification."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        loader = XGBoostModelLoader()
        
        # Check for key features mentioned in the papers
        expected_features = [
            "sleep_percentage_MN",
            "sleep_percentage_SD", 
            "sleep_percentage_Z",
            "circadian_amplitude_MN",
            "circadian_phase_MN",
            "long_num_MN",
            "short_num_MN"
        ]
        
        for feature in expected_features:
            assert feature in loader.feature_names

    @patch('pathlib.Path.exists')
    @patch('joblib.load')
    def test_load_single_model(self, mock_joblib_load, mock_exists):
        """Test loading a single XGBoost model."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
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
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        loader = XGBoostModelLoader()
        fake_path = Path("nonexistent/model.pkl")
        
        success = loader.load_model("depression", fake_path)
        
        assert success is False
        assert "depression" not in loader.models

    @patch('pathlib.Path.exists')
    @patch('joblib.load')
    def test_load_all_models(self, mock_joblib_load, mock_exists):
        """Test loading all three mood models."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock loaded model
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model
        
        loader = XGBoostModelLoader()
        model_dir = Path("model_weights/xgboost/pretrained")
        
        results = loader.load_all_models(model_dir)
        
        assert len(results) == 3
        assert all(results.values())  # All should be True
        assert loader.is_loaded is True
        assert len(loader.models) == 3

    def test_predict_without_loaded_models(self):
        """Test that prediction fails gracefully without loaded models."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        loader = XGBoostModelLoader()
        features = np.random.randn(36)
        
        with pytest.raises(RuntimeError, match="Models not loaded"):
            loader.predict(features)

    @patch('pathlib.Path.exists')
    @patch('joblib.load')
    def test_predict_with_loaded_models(self, mock_joblib_load, mock_exists):
        """Test making predictions with loaded models."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        # Mock file exists
        mock_exists.return_value = True
        
        # Create mock models with different predictions
        mock_models = {
            "depression": MagicMock(predict_proba=MagicMock(
                return_value=np.array([[0.3, 0.7]])  # 70% risk
            )),
            "hypomanic": MagicMock(predict_proba=MagicMock(
                return_value=np.array([[0.8, 0.2]])  # 20% risk
            )),
            "manic": MagicMock(predict_proba=MagicMock(
                return_value=np.array([[0.9, 0.1]])  # 10% risk
            ))
        }
        
        # Set up joblib.load to return different models
        mock_joblib_load.side_effect = [
            mock_models["depression"],
            mock_models["hypomanic"],
            mock_models["manic"]
        ]
        
        loader = XGBoostModelLoader()
        loader.load_all_models(Path("model_weights/xgboost/pretrained"))
        
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
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        loader = XGBoostModelLoader()
        
        # Test wrong number of features
        with pytest.raises(ValueError, match="Expected 36 features"):
            loader._validate_features(np.random.randn(35))
        
        # Test correct features pass
        features = np.random.randn(36)
        loader._validate_features(features)  # Should not raise

    def test_feature_dict_to_array(self):
        """Test converting feature dictionary to array in correct order."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        loader = XGBoostModelLoader()
        
        # Create feature dict
        feature_dict = {name: i for i, name in enumerate(loader.feature_names)}
        
        # Convert to array
        feature_array = loader.dict_to_array(feature_dict)
        
        assert len(feature_array) == 36
        assert all(feature_array[i] == i for i in range(36))

    @patch('pathlib.Path.exists')
    @patch('joblib.load')
    def test_batch_prediction(self, mock_joblib_load, mock_exists):
        """Test making predictions on multiple samples."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        # Mock setup
        mock_exists.return_value = True
        
        # Create mock models
        batch_size = 5
        mock_depression = MagicMock()
        mock_depression.predict_proba.return_value = np.array([
            [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]
        ])
        
        mock_hypomanic = MagicMock()
        mock_hypomanic.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.7, 0.3], [0.9, 0.1], [0.6, 0.4], [0.85, 0.15]
        ])
        
        mock_manic = MagicMock()
        mock_manic.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.85, 0.15], [0.95, 0.05], [0.8, 0.2], [0.92, 0.08]
        ])
        
        mock_joblib_load.side_effect = [mock_depression, mock_hypomanic, mock_manic]
        
        loader = XGBoostModelLoader()
        loader.load_all_models(Path("model_weights/xgboost/pretrained"))
        
        # Make batch prediction
        features_batch = np.random.randn(batch_size, 36)
        predictions = loader.predict_batch(features_batch)
        
        assert len(predictions) == batch_size
        assert all(isinstance(p, MoodPrediction) for p in predictions)
        assert predictions[2].depression_risk == 0.8  # Third sample

    def test_model_info(self):
        """Test retrieving model information."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
        
        loader = XGBoostModelLoader()
        info = loader.get_model_info()
        
        assert "num_features" in info
        assert "models_loaded" in info
        assert "feature_names" in info
        assert info["num_features"] == 36
        assert len(info["models_loaded"]) == 0  # No models loaded yet


class TestXGBoostMoodPredictor:
    """Test the domain service implementation."""

    @patch('pathlib.Path.exists')
    @patch('joblib.load')
    def test_mood_predictor_implementation(self, mock_joblib_load, mock_exists):
        """Test that XGBoostMoodPredictor implements the domain interface."""
        from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostMoodPredictor
        
        # Mock setup
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_joblib_load.return_value = mock_model
        
        # Create predictor
        predictor = XGBoostMoodPredictor()
        model_dir = Path("model_weights/xgboost/pretrained")
        
        # Load models
        predictor.load_models(model_dir)
        
        # Test prediction
        features = np.random.randn(36)
        result = predictor.predict(features)
        
        assert isinstance(result, MoodPrediction)
        assert hasattr(predictor, 'is_loaded')
        assert hasattr(predictor, 'get_model_info')