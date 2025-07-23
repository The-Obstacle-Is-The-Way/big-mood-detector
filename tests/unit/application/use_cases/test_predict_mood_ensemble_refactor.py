"""
Test cases for refactored Ensemble Orchestrator - Phase 1

These tests define the expected behavior for the true dual pipeline.
Written in TDD style before implementation.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class TestRefactoredEnsembleOrchestrator:
    """Test the refactored ensemble orchestrator for true dual pipeline."""

    @pytest.fixture
    def mock_xgboost_predictor(self):
        """Create a mock XGBoost predictor."""
        predictor = Mock()
        predictor.is_loaded = True
        predictor.predict.return_value = MoodPrediction(
            depression_risk=0.3,
            hypomanic_risk=0.4,
            manic_risk=0.2,
            confidence=0.85
        )
        return predictor

    @pytest.fixture
    def mock_pat_model(self):
        """Create a mock PAT model that only extracts embeddings."""
        model = Mock()
        model.is_loaded = True
        # PAT returns 96-dimensional embeddings
        model.extract_features.return_value = np.random.randn(96)
        # PAT cannot predict yet (no classification heads)
        model.predict_mood = Mock(side_effect=RuntimeError("No classification head loaded"))
        return model

    @pytest.fixture
    def sample_activity_records(self):
        """Create sample activity records for PAT."""
        records = []
        base_date = datetime(2025, 7, 16, tzinfo=UTC)  # 7 days ago

        for day in range(7):
            for hour in range(24):
                start = base_date + timedelta(days=day, hours=hour)
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

    def test_refactored_returns_pat_embeddings_separately(
        self, mock_xgboost_predictor, mock_pat_model, sample_activity_records
    ):
        """Test that PAT embeddings are returned separately, not fed to XGBoost."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
            EnsemblePrediction,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=mock_pat_model
        )

        features = np.random.randn(36)
        result = orchestrator.predict(
            statistical_features=features,
            activity_records=sample_activity_records,
            prediction_date=np.datetime64('2025-07-23')
        )

        # Key assertions for refactored behavior
        assert isinstance(result, EnsemblePrediction)

        # XGBoost should only be called with original features
        assert mock_xgboost_predictor.predict.call_count == 1
        call_args = mock_xgboost_predictor.predict.call_args[0][0]
        assert np.array_equal(call_args, features)

        # PAT should extract embeddings
        assert mock_pat_model.extract_features.called

        # NEW: Check for pat_embeddings field (needs to be added to dataclass)
        assert hasattr(result, 'pat_embeddings')
        assert result.pat_embeddings is not None
        assert result.pat_embeddings.shape == (96,)

        # PAT prediction should be None (no classification head yet)
        assert hasattr(result, 'pat_prediction')
        assert result.pat_prediction is None

        # XGBoost prediction should work as before
        assert result.xgboost_prediction is not None
        assert result.xgboost_prediction.depression_risk == 0.3

        # Ensemble should fallback to XGBoost only
        assert result.ensemble_prediction == result.xgboost_prediction

    def test_xgboost_not_contaminated_with_pat_features(
        self, mock_xgboost_predictor, mock_pat_model, sample_activity_records
    ):
        """Ensure XGBoost never sees PAT embeddings in the refactored version."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=mock_pat_model
        )

        features = np.random.randn(36)
        _ = orchestrator.predict(
            statistical_features=features,
            activity_records=sample_activity_records
        )

        # Critical: XGBoost should NEVER receive concatenated features
        call_args = mock_xgboost_predictor.predict.call_args[0][0]
        assert call_args.shape == (36,)  # Original features only
        assert np.array_equal(call_args, features)

    def test_temporal_context_added_to_predictions(
        self, mock_xgboost_predictor, mock_pat_model, sample_activity_records
    ):
        """Test that temporal context is clearly labeled (addresses Issue #25)."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=mock_pat_model
        )

        result = orchestrator.predict(
            statistical_features=np.random.randn(36),
            activity_records=sample_activity_records
        )

        # NEW: Check for temporal_context field
        assert hasattr(result, 'temporal_context')
        assert result.temporal_context is not None
        assert result.temporal_context['xgboost'] == 'next_24_hours'
        assert result.temporal_context['pat'] == 'embeddings_only'  # Will be 'current_state' after training

    def test_pat_embeddings_without_activity_records(
        self, mock_xgboost_predictor, mock_pat_model
    ):
        """Test behavior when no activity records provided."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=mock_pat_model
        )

        result = orchestrator.predict(
            statistical_features=np.random.randn(36),
            activity_records=None  # No activity data
        )

        # Should work with XGBoost only
        assert result.xgboost_prediction is not None
        assert result.pat_embeddings is None
        assert result.pat_prediction is None
        assert 'xgboost' in result.models_used
        assert 'pat' not in result.models_used

    def test_refactored_dataclass_structure(self):
        """Test the updated EnsemblePrediction dataclass structure."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsemblePrediction,
        )
        from big_mood_detector.domain.services.mood_predictor import MoodPrediction

        # Test creating the refactored prediction
        xgb_pred = MoodPrediction(
            depression_risk=0.3,
            hypomanic_risk=0.4,
            manic_risk=0.2,
            confidence=0.85
        )

        embeddings = np.random.randn(96)

        result = EnsemblePrediction(
            xgboost_prediction=xgb_pred,
            pat_enhanced_prediction=None,  # Deprecated but required
            pat_embeddings=embeddings,  # NEW field
            pat_prediction=None,  # NEW field (replaces pat_enhanced_prediction)
            ensemble_prediction=xgb_pred,
            models_used=['xgboost'],
            confidence_scores={'xgboost': 0.85, 'ensemble': 0.85},
            processing_time_ms={'xgboost': 10.0, 'total': 15.0},
            temporal_context={  # NEW field
                'xgboost': 'next_24_hours',
                'pat': 'embeddings_only'
            }
        )

        assert result.pat_embeddings.shape == (96,)
        assert result.pat_prediction is None
        assert result.temporal_context['xgboost'] == 'next_24_hours'

    def test_models_used_reflects_actual_predictions(
        self, mock_xgboost_predictor, mock_pat_model, sample_activity_records
    ):
        """Test that models_used only includes models making predictions."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=mock_pat_model
        )

        result = orchestrator.predict(
            statistical_features=np.random.randn(36),
            activity_records=sample_activity_records
        )

        # Only XGBoost makes predictions currently
        assert 'xgboost' in result.models_used
        assert 'pat_embeddings' in result.models_used  # Indicates embeddings extracted
        assert 'pat_enhanced' not in result.models_used  # OLD naming removed
        assert 'pat_prediction' not in result.models_used  # Not available yet

    def test_no_feature_concatenation_in_refactored_version(
        self, mock_xgboost_predictor, mock_pat_model, sample_activity_records
    ):
        """Ensure the problematic feature concatenation is removed."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        # Spy on XGBoost to track all calls
        call_history = []
        def track_calls(features):
            call_history.append(features.copy())
            return MoodPrediction(
                depression_risk=0.3,
                hypomanic_risk=0.4,
                manic_risk=0.2,
                confidence=0.85
            )

        mock_xgboost_predictor.predict.side_effect = track_calls

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=mock_pat_model
        )

        features = np.random.randn(36)
        _ = orchestrator.predict(
            statistical_features=features,
            activity_records=sample_activity_records
        )

        # Should only have ONE call to XGBoost (not two like before)
        assert len(call_history) == 1

        # That call should use original features only
        assert call_history[0].shape == (36,)
        assert np.array_equal(call_history[0], features)

        # No concatenated features should exist
        for call_features in call_history:
            assert len(call_features) == 36  # Not 36 (20+16)

    def test_backwards_compatibility_maintained(
        self, mock_xgboost_predictor
    ):
        """Ensure XGBoost-only mode still works (backwards compatibility)."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )

        # No PAT model
        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=mock_xgboost_predictor,
            pat_model=None
        )

        features = np.random.randn(36)
        result = orchestrator.predict(statistical_features=features)

        # Should work exactly as before
        assert result.xgboost_prediction is not None
        assert result.pat_embeddings is None
        assert result.pat_prediction is None
        assert result.ensemble_prediction == result.xgboost_prediction
        assert len(result.models_used) == 1
        assert 'xgboost' in result.models_used

