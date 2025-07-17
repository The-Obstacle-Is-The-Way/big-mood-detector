"""
Test Pipeline Ensemble Integration

TDD for connecting ensemble orchestrator to main pipeline.
"""

from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    EnsembleConfig,
    EnsembleOrchestrator,
    EnsemblePrediction,
)
from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class TestPipelineEnsembleIntegration:
    """Test that ensemble models are properly integrated."""

    def test_pipeline_uses_ensemble_when_configured(self):
        """Test that pipeline uses ensemble orchestrator when enabled."""
        # Create pipeline with ensemble enabled
        config = PipelineConfig(
            include_pat_sequences=True,  # This should trigger ensemble
            model_dir=Path("models"),
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Verify ensemble orchestrator is created
        assert hasattr(pipeline, "ensemble_orchestrator")
        assert pipeline.ensemble_orchestrator is not None
        assert isinstance(pipeline.ensemble_orchestrator, EnsembleOrchestrator)

    def test_pipeline_uses_ensemble_for_predictions(self):
        """Test that predictions go through ensemble orchestrator."""
        config = PipelineConfig(
            include_pat_sequences=True,
            model_dir=Path("models"),
        )

        with patch(
            "big_mood_detector.application.use_cases.process_health_data_use_case.EnsembleOrchestrator"
        ) as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble

            # Mock ensemble prediction
            mock_ensemble.predict.return_value = EnsemblePrediction(
                xgboost_prediction=MoodPrediction(
                    depression_risk=0.3,
                    hypomanic_risk=0.1,
                    manic_risk=0.05,
                    confidence=0.8,
                ),
                pat_enhanced_prediction=MoodPrediction(
                    depression_risk=0.35,
                    hypomanic_risk=0.15,
                    manic_risk=0.07,
                    confidence=0.75,
                ),
                ensemble_prediction=MoodPrediction(
                    depression_risk=0.32,
                    hypomanic_risk=0.12,
                    manic_risk=0.06,
                    confidence=0.85,
                ),
                models_used=["xgboost", "pat"],
                confidence_scores={"xgboost": 0.8, "pat": 0.75},
                processing_time_ms={"xgboost": 10, "pat": 40},
            )

            pipeline = MoodPredictionPipeline(config=config)

            # Process some data
            result = pipeline.process_health_data(
                sleep_records=[],
                activity_records=[],
                heart_records=[],
                target_date=date.today(),
            )

            # Verify ensemble was used
            assert mock_ensemble.predict.called

            # Verify combined predictions are used
            if result.daily_predictions:
                pred = list(result.daily_predictions.values())[0]
                assert pred["depression_risk"] == 0.32  # Combined prediction
                assert pred["model_agreement"] == 0.9

    def test_pipeline_falls_back_to_xgboost_when_pat_disabled(self):
        """Test fallback to XGBoost-only when PAT is disabled."""
        config = PipelineConfig(
            include_pat_sequences=False,  # Disable ensemble
            model_dir=Path("models"),
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Should not have ensemble orchestrator
        assert (
            not hasattr(pipeline, "ensemble_orchestrator")
            or pipeline.ensemble_orchestrator is None
        )

        # Should still have basic mood predictor
        assert hasattr(pipeline, "mood_predictor")
        assert pipeline.mood_predictor is not None

    def test_ensemble_handles_pat_model_failure(self):
        """Test that ensemble gracefully handles PAT model failures."""
        config = PipelineConfig(
            include_pat_sequences=True,
            model_dir=Path("models"),
        )

        with patch(
            "big_mood_detector.application.use_cases.process_health_data_use_case.EnsembleOrchestrator"
        ) as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble

            # Simulate PAT failure - ensemble returns XGBoost-only result
            mock_ensemble.predict.return_value = EnsemblePrediction(
                xgboost_prediction=MoodPrediction(
                    depression_risk=0.3,
                    hypomanic_risk=0.1,
                    manic_risk=0.05,
                    confidence=0.8,
                ),
                pat_prediction=None,  # PAT failed
                combined_prediction=MoodPrediction(
                    depression_risk=0.3,  # Falls back to XGBoost
                    hypomanic_risk=0.1,
                    manic_risk=0.05,
                    confidence=0.6,  # Lower confidence
                ),
                model_agreement=0.0,  # No agreement since only one model
                processing_time_ms=20,
            )

            pipeline = MoodPredictionPipeline(config=config)

            result = pipeline.process_health_data(
                sleep_records=[],
                activity_records=[],
                heart_records=[],
                target_date=date.today(),
            )

            # Should still get predictions
            assert len(result.daily_predictions) > 0
            # But with warnings
            assert "PAT model unavailable" in result.warnings

    def test_ensemble_weight_configuration(self):
        """Test that ensemble weights can be configured."""
        ensemble_config = EnsembleConfig(
            xgboost_weight=0.7,
            pat_weight=0.3,
        )

        config = PipelineConfig(
            include_pat_sequences=True,
            ensemble_config=ensemble_config,
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Verify custom weights are used
        assert pipeline.ensemble_orchestrator.config.xgboost_weight == 0.7
        assert pipeline.ensemble_orchestrator.config.pat_weight == 0.3

    def test_pipeline_passes_activity_data_to_ensemble(self):
        """Test that activity records are passed to ensemble for PAT."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        config = PipelineConfig(
            include_pat_sequences=True,
        )

        # Create test activity records
        activity_records = [
            ActivityRecord(
                type=ActivityType.STEP_COUNT,
                value=5000.0,
                unit="count",
                start_date=date.today(),
                end_date=date.today(),
                source_name="test",
            )
        ]

        with patch(
            "big_mood_detector.application.use_cases.process_health_data_use_case.EnsembleOrchestrator"
        ) as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble

            mock_ensemble.predict.return_value = EnsemblePrediction(
                xgboost_prediction=MoodPrediction(0.1, 0.1, 0.1, 0.8),
                pat_prediction=MoodPrediction(0.1, 0.1, 0.1, 0.8),
                combined_prediction=MoodPrediction(0.1, 0.1, 0.1, 0.8),
                model_agreement=1.0,
                processing_time_ms=10,
            )

            pipeline = MoodPredictionPipeline(config=config)

            _ = pipeline.process_health_data(
                sleep_records=[],
                activity_records=activity_records,
                heart_records=[],
                target_date=date.today(),
            )

            # Verify activity records were passed to ensemble
            call_args = mock_ensemble.predict.call_args
            assert "activity_records" in call_args.kwargs
            assert len(call_args.kwargs["activity_records"]) == 1
