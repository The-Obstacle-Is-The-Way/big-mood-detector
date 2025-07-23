"""
Test Temporal Ensemble Orchestrator

Tests for orchestrating PAT (current state) and XGBoost (future prediction)
in a temporally-aware manner.
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from big_mood_detector.domain.services.mood_predictor import MoodPrediction
from big_mood_detector.domain.services.pat_predictor import PATBinaryPredictions
from big_mood_detector.domain.value_objects.temporal_mood_assessment import (
    CurrentMoodState,
    FutureMoodRisk,
    TemporalMoodAssessment,
)


class TestTemporalEnsembleOrchestrator:
    """Test the temporal ensemble orchestrator."""

    def test_orchestrator_can_be_imported(self):
        """Test that orchestrator can be imported."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        assert TemporalEnsembleOrchestrator is not None

    @pytest.fixture
    def mock_pat_predictor(self):
        """Create mock PAT predictor."""
        predictor = MagicMock()
        predictor.predict_from_embeddings.return_value = PATBinaryPredictions(
            depression_probability=0.7,
            benzodiazepine_probability=0.2,
            confidence=0.85
        )
        return predictor

    @pytest.fixture
    def mock_xgboost_predictor(self):
        """Create mock XGBoost predictor."""
        predictor = MagicMock()
        predictor.predict.return_value = MoodPrediction(
            depression_risk=0.3,
            hypomanic_risk=0.6,
            manic_risk=0.1,
            confidence=0.8
        )
        return predictor

    @pytest.fixture
    def mock_pat_encoder(self):
        """Create mock PAT encoder."""
        encoder = MagicMock()
        encoder.encode.return_value = np.random.rand(96)
        return encoder

    def test_orchestrator_initialization(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator initialization."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        assert orchestrator.pat_predictor is mock_pat_predictor
        assert orchestrator.xgboost_predictor is mock_xgboost_predictor
        assert orchestrator.pat_encoder is mock_pat_encoder

    def test_orchestrator_returns_temporal_assessment(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator returns TemporalMoodAssessment."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Mock inputs
        xgboost_features = np.random.rand(36)
        activity_sequences = np.random.rand(7, 1440)
        user_id = "test_user_123"

        # Execute
        result = orchestrator.predict(
            pat_sequence=activity_sequences,
            statistical_features=xgboost_features,
            user_id=user_id
        )

        # Verify
        assert isinstance(result, TemporalMoodAssessment)
        assert isinstance(result.current_state, CurrentMoodState)
        assert isinstance(result.future_risk, FutureMoodRisk)
        assert result.user_id == user_id
        assert isinstance(result.assessment_timestamp, datetime)

    def test_pat_assesses_current_state(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test PAT is used for current state assessment."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Execute
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=np.random.rand(7, 1440),
            user_id="test_user"
        )

        # Verify PAT was used for current state
        mock_pat_encoder.encode.assert_called_once()
        mock_pat_predictor.predict_from_embeddings.assert_called_once()

        # Check current state values match PAT output
        assert result.current_state.depression_probability == 0.7
        assert result.current_state.on_benzodiazepine_probability == 0.2
        assert result.current_state.confidence == 0.85

    def test_xgboost_predicts_future_risk(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test XGBoost is used for future risk prediction."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Execute
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=np.random.rand(7, 1440),
            user_id="test_user"
        )

        # Verify XGBoost was used for future risk
        mock_xgboost_predictor.predict.assert_called_once()

        # Check future risk values match XGBoost output
        assert result.future_risk.depression_risk == 0.3
        assert result.future_risk.hypomanic_risk == 0.6
        assert result.future_risk.manic_risk == 0.1
        assert result.future_risk.confidence == 0.8
        # Hypomanic has highest probability (0.6)

    @pytest.mark.xfail(reason="Phase 4: Handle missing PAT sequences")
    def test_orchestrator_handles_missing_pat_data(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator handles missing activity sequences gracefully."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Execute with None activity sequences
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=None,
            user_id="test_user"
        )

        # When None is passed, PAT will try to encode and fail
        # Our implementation provides default values on failure
        assert result.current_state.depression_probability == 0.5
        assert result.current_state.on_benzodiazepine_probability == 0.5
        assert result.current_state.confidence == 0.0

        # Future risk should still work
        assert result.future_risk.depression_risk == 0.3

    def test_orchestrator_handles_pat_failure(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator handles PAT prediction failure gracefully."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        # Make PAT predictor raise an exception
        mock_pat_predictor.predict_from_embeddings.side_effect = Exception("PAT model error")

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Execute
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=np.random.rand(7, 1440),
            user_id="test_user"
        )

        # Should not raise exception
        assert isinstance(result, TemporalMoodAssessment)

        # Current state should have default values on failure
        assert result.current_state.depression_probability == 0.5
        assert result.current_state.confidence == 0.0

        # Future risk should still work
        assert result.future_risk.depression_risk == 0.3

    def test_orchestrator_handles_xgboost_failure(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator handles XGBoost prediction failure gracefully."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        # Make XGBoost predictor raise an exception
        mock_xgboost_predictor.predict.side_effect = Exception("XGBoost model error")

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Execute
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=np.random.rand(7, 1440),
            user_id="test_user"
        )

        # Should not raise exception
        assert isinstance(result, TemporalMoodAssessment)

        # Current state should still work
        assert result.current_state.depression_probability == 0.7

        # Future risk should have default values on failure
        assert result.future_risk.depression_risk == 0.33
        assert result.future_risk.hypomanic_risk == 0.33
        assert result.future_risk.manic_risk == 0.34
        assert result.future_risk.confidence == 0.0

    def test_temporal_consistency_tracking(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator tracks temporal consistency between assessments."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Execute
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=np.random.rand(7, 1440),
            user_id="test_user"
        )

        # Check temporal assessment has required properties
        assert hasattr(result, 'temporal_concordance')
        assert hasattr(result, 'requires_immediate_intervention')
        assert hasattr(result, 'requires_preventive_action')

    @pytest.mark.xfail(reason="Phase 4: Implement short sequence handling")
    def test_orchestrator_with_insufficient_activity_days(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator handles insufficient activity data (< 7 days)."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Only 3 days of data
        short_sequences = np.random.rand(3, 1440)

        # Execute
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=short_sequences,
            user_id="test_user"
        )

        # Current state should indicate insufficient data
        assert result.current_state.data_sufficiency == "partial"
        assert result.current_state.confidence < 0.5  # Low confidence

        # Future risk should still work with XGBoost features
        assert result.future_risk.depression_risk == 0.3

    @pytest.mark.xfail(reason="Phase 4: Implement clinical alert thresholds")
    def test_critical_alert_generation(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator generates alerts for critical patterns."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        # Set up critical scenario: Currently depressed + High future mania risk
        mock_pat_predictor.predict_from_embeddings.return_value = PATBinaryPredictions(
            depression_probability=0.85,  # Currently depressed
            benzodiazepine_probability=0.1,
            confidence=0.9
        )

        mock_xgboost_predictor.predict.return_value = MoodPrediction(
            depression_probability=0.1,
            hypomania_probability=0.2,
            mania_probability=0.7,  # High future mania risk
            confidence=0.85
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Execute
        result = orchestrator.predict(
            statistical_features=np.random.rand(36),
            pat_sequence=np.random.rand(7, 1440),
            user_id="test_user"
        )

        # Should generate alert for rapid mood cycling risk
        assert hasattr(result, 'clinical_alerts')
        assert len(result.clinical_alerts) > 0
        assert any('rapid cycling' in alert.lower() for alert in result.clinical_alerts)

    @pytest.mark.xfail(reason="Phase 4: Full pipeline integration")
    def test_integration_with_pipelines(self, mock_pat_predictor, mock_xgboost_predictor, mock_pat_encoder):
        """Test orchestrator integrates with existing pipeline structure."""
        from big_mood_detector.application.services.temporal_ensemble_orchestrator import (
            TemporalEnsembleOrchestrator,
        )

        orchestrator = TemporalEnsembleOrchestrator(
            pat_predictor=mock_pat_predictor,
            xgboost_predictor=mock_xgboost_predictor,
            pat_encoder=mock_pat_encoder
        )

        # Simulate pipeline output format
        pipeline_features = {
            'xgboost_features': np.random.rand(36),
            'activity_sequences': np.random.rand(7, 1440),
            'metadata': {
                'user_id': 'pipeline_user',
                'processing_date': datetime.now()
            }
        }

        # Execute with pipeline data
        result = orchestrator.predict_from_pipeline(pipeline_features)

        # Verify integration
        assert isinstance(result, TemporalMoodAssessment)
        assert result.user_id == 'pipeline_user'
        assert result.current_state.depression_probability == 0.7
        assert result.future_risk.hypomanic_risk == 0.6
