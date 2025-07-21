"""
Test cases for Ensemble Orchestrator

Following TDD principles - tests written before implementation.
"""

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class TestEnsembleOrchestrator:
    """Test the ensemble orchestrator functionality."""

    @pytest.fixture
    def mock_xgboost_predictor(self):
        """Create a mock XGBoost predictor."""
        predictor = Mock()
        predictor.is_loaded = True
        predictor.predict.return_value = MoodPrediction(
            depression_risk=0.3, hypomanic_risk=0.4, manic_risk=0.2, confidence=0.85
        )
        return predictor

    @pytest.fixture
    def mock_pat_model(self):
        """Create a mock PAT model."""
        model = Mock()
        model.is_loaded = True
        model.extract_features.return_value = np.random.randn(96)
        return model

    @pytest.fixture
    def sample_activity_records(self):
        """Create sample activity records."""
        records = []
        base_date = datetime(2025, 5, 9, tzinfo=UTC)

        for hour in range(24):
            start = base_date + timedelta(hours=hour)
            end = start + timedelta(hours=1)
            records.append(
                ActivityRecord(
                    source_name="Test",
                    start_date=start,
                    end_date=end,
                    activity_type=ActivityType.STEP_COUNT,
                    value=np.random.uniform(0, 100),
                    unit="count",
                )
            )
        return records

    def test_orchestrator_initialization(self, mock_xgboost_predictor, mock_pat_model):
        """Test basic orchestrator initialization."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor, pat_model=mock_pat_model
        )

        assert orchestrator.xgboost_predictor == mock_xgboost_predictor
        assert orchestrator.pat_model == mock_pat_model
        assert orchestrator.config is not None
        assert orchestrator.executor is not None

    def test_orchestrator_with_custom_config(self, mock_xgboost_predictor):
        """Test orchestrator with custom configuration."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleConfig,
            EnsembleOrchestrator,
        )

        config = EnsembleConfig(xgboost_weight=0.7, pat_weight=0.3, pat_timeout=5.0)

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor, config=config
        )

        assert orchestrator.config.xgboost_weight == 0.7
        assert orchestrator.config.pat_weight == 0.3
        assert orchestrator.config.pat_timeout == 5.0

    def test_xgboost_only_prediction(self, mock_xgboost_predictor):
        """Test prediction with only XGBoost (no PAT)."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor, pat_model=None  # No PAT model
        )

        features = np.random.randn(36)
        result = orchestrator.predict(features)

        assert result.xgboost_prediction is not None
        assert result.pat_enhanced_prediction is None
        assert result.ensemble_prediction == result.xgboost_prediction
        assert "xgboost" in result.models_used
        assert len(result.models_used) == 1

    def test_ensemble_prediction_both_models(
        self, mock_xgboost_predictor, mock_pat_model, sample_activity_records
    ):
        """Test ensemble prediction with both models."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        # Setup PAT-enhanced prediction
        pat_enhanced_pred = MoodPrediction(
            depression_risk=0.25,
            hypomanic_risk=0.45,
            manic_risk=0.15,
            confidence=0.9,
        )

        # Mock to return different predictions for different inputs
        call_count = 0

        def predict_side_effect(features):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: regular XGBoost
                return mock_xgboost_predictor.predict.return_value
            else:
                # Second call: PAT-enhanced
                return pat_enhanced_pred

        mock_xgboost_predictor.predict.side_effect = predict_side_effect

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor, pat_model=mock_pat_model
        )

        features = np.random.randn(36)
        result = orchestrator.predict(
            features, activity_records=sample_activity_records
        )

        assert result.xgboost_prediction is not None
        assert result.pat_enhanced_prediction is not None
        assert len(result.models_used) == 2

        # Check ensemble calculation (weighted average)
        expected_depression = 0.6 * 0.3 + 0.4 * 0.25  # 0.28
        assert (
            abs(result.ensemble_prediction.depression_risk - expected_depression) < 0.01
        )

    def test_timeout_handling(self, mock_xgboost_predictor, mock_pat_model):
        """Test handling of model timeouts."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleConfig,
            EnsembleOrchestrator,
        )

        # Make PAT model timeout
        def slow_extract(*args, **kwargs):
            time.sleep(0.2)  # Longer than timeout
            return np.random.randn(96)

        mock_pat_model.extract_features = slow_extract

        config = EnsembleConfig(pat_timeout=0.1)  # Very short timeout

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=mock_pat_model,
            config=config,
        )

        features = np.random.randn(36)
        result = orchestrator.predict(features, activity_records=[])

        # Should fallback to XGBoost only
        assert result.xgboost_prediction is not None
        assert result.pat_enhanced_prediction is None
        assert "xgboost" in result.models_used
        assert "pat_enhanced" not in result.models_used

    def test_both_models_fail(self):
        """Test graceful degradation when both models fail."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        # Create failing predictors
        xgboost_predictor = Mock()
        xgboost_predictor.is_loaded = True
        xgboost_predictor.predict.side_effect = Exception("XGBoost failed")

        pat_model = Mock()
        pat_model.is_loaded = True
        pat_model.extract_features.side_effect = Exception("PAT failed")

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor, pat_model=pat_model
        )

        features = np.random.randn(36)
        result = orchestrator.predict(features)

        # Should return neutral prediction
        assert result.ensemble_prediction.depression_risk == 0.5
        assert result.ensemble_prediction.hypomanic_risk == 0.5
        assert result.ensemble_prediction.manic_risk == 0.5
        assert result.ensemble_prediction.confidence == 0.0

    def test_confidence_calculation(self, mock_xgboost_predictor, mock_pat_model):
        """Test confidence score calculation."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor, pat_model=mock_pat_model
        )

        features = np.random.randn(36)
        result = orchestrator.predict(features)

        assert "xgboost" in result.confidence_scores
        assert "ensemble" in result.confidence_scores
        assert result.confidence_scores["ensemble"] > 0

    def test_processing_time_tracking(self, mock_xgboost_predictor):
        """Test that processing times are tracked."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(xgboost_predictor=mock_xgboost_predictor)

        features = np.random.randn(36)
        result = orchestrator.predict(features)

        assert "total" in result.processing_time_ms
        assert "xgboost" in result.processing_time_ms
        assert result.processing_time_ms["total"] > 0

    def test_shutdown_cleanup(self, mock_xgboost_predictor):
        """Test proper resource cleanup."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(xgboost_predictor=mock_xgboost_predictor)

        # Should not raise any exceptions
        orchestrator.shutdown()


class TestEnsembleConfig:
    """Test the ensemble configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleConfig,
        )

        config = EnsembleConfig()

        assert config.xgboost_weight == 0.6
        assert config.pat_weight == 0.4
        assert config.xgboost_weight + config.pat_weight == 1.0
        assert config.use_pat_features is True
        assert config.pat_feature_dim == 16

    def test_custom_config(self):
        """Test custom configuration."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleConfig,
        )

        config = EnsembleConfig(
            xgboost_weight=0.8,
            pat_weight=0.2,
            use_pat_features=False,
            min_confidence_threshold=0.8,
        )

        assert config.xgboost_weight == 0.8
        assert config.pat_weight == 0.2
        assert config.use_pat_features is False
        assert config.min_confidence_threshold == 0.8


class TestEnsemblePrediction:
    """Test the ensemble prediction dataclass."""

    def test_ensemble_prediction_creation(self):
        """Test creating ensemble prediction result."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsemblePrediction,
        )
        from big_mood_detector.domain.services.mood_predictor import MoodPrediction

        xgb_pred = MoodPrediction(
            depression_risk=0.3,
            hypomanic_risk=0.4,
            manic_risk=0.2,
            confidence=0.85,
        )

        ensemble_result = EnsemblePrediction(
            xgboost_prediction=xgb_pred,
            pat_enhanced_prediction=None,
            ensemble_prediction=xgb_pred,
            models_used=["xgboost"],
            confidence_scores={"xgboost": 0.85, "ensemble": 0.85},
            processing_time_ms={"xgboost": 10.5, "total": 12.0},
        )

        assert ensemble_result.xgboost_prediction == xgb_pred
        assert ensemble_result.pat_enhanced_prediction is None
        assert len(ensemble_result.models_used) == 1
        assert ensemble_result.processing_time_ms["total"] == 12.0
