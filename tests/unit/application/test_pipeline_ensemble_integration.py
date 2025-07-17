"""
Test Pipeline Ensemble Integration

TDD for connecting ensemble orchestrator to main pipeline.
"""

from datetime import date
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

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

    @patch(
        "big_mood_detector.infrastructure.ml_models.xgboost_models.XGBoostMoodPredictor"
    )
    @patch("big_mood_detector.infrastructure.ml_models.pat_model.PATModel")
    @patch("pathlib.Path.exists")
    def test_pipeline_uses_ensemble_when_configured(
        self, mock_exists, mock_pat_class, mock_xgb_class
    ):
        """Test that pipeline uses ensemble orchestrator when enabled."""
        # Mock file existence check
        mock_exists.return_value = True

        # Mock XGBoost predictor
        mock_xgb_instance = Mock()
        mock_xgb_instance.load_models.return_value = {
            "depression": True,
            "hypomanic": True,
            "manic": True,
        }
        mock_xgb_instance.is_loaded = True
        mock_xgb_class.return_value = mock_xgb_instance

        # Mock PAT model
        mock_pat_instance = Mock()
        mock_pat_instance.load_pretrained_weights.return_value = True
        mock_pat_class.return_value = mock_pat_instance

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

    @patch(
        "big_mood_detector.domain.services.mood_predictor.MoodPredictor._load_models"
    )
    @patch(
        "big_mood_detector.infrastructure.ml_models.xgboost_models.XGBoostMoodPredictor"
    )
    @patch("big_mood_detector.infrastructure.ml_models.pat_model.PATModel")
    @patch("pathlib.Path.exists")
    def test_pipeline_uses_ensemble_for_predictions(
        self, mock_exists, mock_pat_class, mock_xgb_class, mock_load_models
    ):
        """Test that predictions go through ensemble orchestrator."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock domain model loading
        mock_load_models.return_value = None

        # Mock XGBoost predictor
        mock_xgb_instance = Mock()
        mock_xgb_instance.load_models.return_value = {
            "depression": True,
            "hypomanic": True,
            "manic": True,
        }
        mock_xgb_instance.is_loaded = True
        mock_xgb_class.return_value = mock_xgb_instance

        # Mock PAT model
        mock_pat_instance = Mock()
        mock_pat_instance.load_pretrained_weights.return_value = True
        mock_pat_class.return_value = mock_pat_instance

        config = PipelineConfig(
            include_pat_sequences=True,
            model_dir=Path("model_weights/xgboost/pretrained"),
        )

        # Mock the clinical feature extractor to return features
        import numpy as np

        from big_mood_detector.domain.services.clinical_feature_extractor import (
            ClinicalFeatureSet,
            SeoulXGBoostFeatures,
        )

        with (
            patch(
                "big_mood_detector.application.use_cases.process_health_data_use_case.EnsembleOrchestrator"
            ) as mock_ensemble_class,
            patch(
                "big_mood_detector.domain.services.mood_predictor.MoodPredictor.is_loaded",
                new_callable=PropertyMock,
                return_value=True,
            ),
        ):
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

            # Create pipeline after setting up the mock
            pipeline = MoodPredictionPipeline(config=config)

            with patch.object(
                pipeline.clinical_extractor, "extract_clinical_features"
            ) as mock_extract:
                # Create mock features
                mock_features = Mock(spec=ClinicalFeatureSet)
                mock_seoul = Mock(spec=SeoulXGBoostFeatures)
                mock_seoul.to_xgboost_features.return_value = np.zeros(36)
                mock_features.seoul_features = mock_seoul
                mock_extract.return_value = mock_features

                # Also need to provide some sleep records for date calculation
                from datetime import datetime, timedelta

                from big_mood_detector.domain.entities.sleep_record import (
                    SleepRecord,
                    SleepState,
                )

                target_date = date.today()
                sleep_records = [
                    SleepRecord(
                        source_name="test",
                        start_date=datetime.combine(
                            target_date - timedelta(days=i), datetime.min.time()
                        ),
                        end_date=datetime.combine(
                            target_date - timedelta(days=i), datetime.min.time()
                        )
                        + timedelta(hours=8),
                        state=SleepState.ASLEEP,
                    )
                    for i in range(7)
                ]

                # Process some data
                result = pipeline.process_health_data(
                    sleep_records=sleep_records,
                    activity_records=[],
                    heart_records=[],
                    target_date=target_date,
                )

                # Verify ensemble was used
                assert mock_ensemble.predict.called

            # Verify combined predictions are used
            if result.daily_predictions:
                pred = list(result.daily_predictions.values())[0]
                assert pred["depression_risk"] == 0.32  # Combined prediction
                assert "models_used" in pred  # Should have ensemble metadata
                assert "confidence_scores" in pred

    def test_pipeline_falls_back_to_xgboost_when_pat_disabled(self):
        """Test fallback to XGBoost-only when PAT is disabled."""
        config = PipelineConfig(
            include_pat_sequences=False,  # Disable ensemble
            model_dir=Path("model_weights/xgboost/pretrained"),
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Should not have ensemble orchestrator
        assert pipeline.ensemble_orchestrator is None

        # Should still have basic mood predictor
        assert hasattr(pipeline, "mood_predictor")
        assert pipeline.mood_predictor is not None

    @patch(
        "big_mood_detector.domain.services.mood_predictor.MoodPredictor._load_models"
    )
    @patch(
        "big_mood_detector.infrastructure.ml_models.xgboost_models.XGBoostMoodPredictor"
    )
    @patch("big_mood_detector.infrastructure.ml_models.pat_model.PATModel")
    @patch("pathlib.Path.exists")
    def test_ensemble_handles_pat_model_failure(
        self, mock_exists, mock_pat_class, mock_xgb_class, mock_load_models
    ):
        """Test that ensemble gracefully handles PAT model failures."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock domain model loading
        mock_load_models.return_value = None

        # Mock XGBoost predictor
        mock_xgb_instance = Mock()
        mock_xgb_instance.load_models.return_value = {
            "depression": True,
            "hypomanic": True,
            "manic": True,
        }
        mock_xgb_instance.is_loaded = True
        mock_xgb_class.return_value = mock_xgb_instance

        # Mock PAT model
        mock_pat_instance = Mock()
        mock_pat_instance.load_pretrained_weights.return_value = True
        mock_pat_class.return_value = mock_pat_instance

        config = PipelineConfig(
            include_pat_sequences=True,
            model_dir=Path("model_weights/xgboost/pretrained"),
        )

        with (
            patch(
                "big_mood_detector.application.use_cases.process_health_data_use_case.EnsembleOrchestrator"
            ) as mock_ensemble_class,
            patch(
                "big_mood_detector.domain.services.mood_predictor.MoodPredictor.is_loaded",
                new_callable=PropertyMock,
                return_value=True,
            ),
        ):
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
                pat_enhanced_prediction=None,  # PAT failed
                ensemble_prediction=MoodPrediction(
                    depression_risk=0.3,  # Falls back to XGBoost
                    hypomanic_risk=0.1,
                    manic_risk=0.05,
                    confidence=0.6,  # Lower confidence
                ),
                models_used=["xgboost"],  # Only XGBoost
                confidence_scores={"xgboost": 0.8},
                processing_time_ms={"xgboost": 20},
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

    @patch(
        "big_mood_detector.infrastructure.ml_models.xgboost_models.XGBoostMoodPredictor"
    )
    @patch("big_mood_detector.infrastructure.ml_models.pat_model.PATModel")
    @patch("pathlib.Path.exists")
    def test_ensemble_weight_configuration(
        self, mock_exists, mock_pat_class, mock_xgb_class
    ):
        """Test that ensemble weights can be configured."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock XGBoost predictor
        mock_xgb_instance = Mock()
        mock_xgb_instance.load_models.return_value = {
            "depression": True,
            "hypomanic": True,
            "manic": True,
        }
        mock_xgb_instance.is_loaded = True
        mock_xgb_class.return_value = mock_xgb_instance

        # Mock PAT model
        mock_pat_instance = Mock()
        mock_pat_instance.load_pretrained_weights.return_value = True
        mock_pat_class.return_value = mock_pat_instance

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

    @patch(
        "big_mood_detector.domain.services.mood_predictor.MoodPredictor._load_models"
    )
    @patch(
        "big_mood_detector.infrastructure.ml_models.xgboost_models.XGBoostMoodPredictor"
    )
    @patch("big_mood_detector.infrastructure.ml_models.pat_model.PATModel")
    @patch("pathlib.Path.exists")
    def test_pipeline_passes_activity_data_to_ensemble(
        self, mock_exists, mock_pat_class, mock_xgb_class, mock_load_models
    ):
        """Test that activity records are passed to ensemble for PAT."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        # Mock file existence
        mock_exists.return_value = True

        # Mock domain model loading
        mock_load_models.return_value = None

        # Mock XGBoost predictor
        mock_xgb_instance = Mock()
        mock_xgb_instance.load_models.return_value = {
            "depression": True,
            "hypomanic": True,
            "manic": True,
        }
        mock_xgb_instance.is_loaded = True
        mock_xgb_class.return_value = mock_xgb_instance

        # Mock PAT model
        mock_pat_instance = Mock()
        mock_pat_instance.load_pretrained_weights.return_value = True
        mock_pat_class.return_value = mock_pat_instance

        config = PipelineConfig(
            include_pat_sequences=True,
        )

        # Create test activity records
        from datetime import datetime

        today = date.today()
        activity_records = [
            ActivityRecord(
                activity_type=ActivityType.STEP_COUNT,
                value=5000.0,
                unit="count",
                start_date=datetime.combine(today, datetime.min.time()),
                end_date=datetime.combine(today, datetime.max.time()),
                source_name="test",
            )
        ]

        with (
            patch(
                "big_mood_detector.application.use_cases.process_health_data_use_case.EnsembleOrchestrator"
            ) as mock_ensemble_class,
            patch(
                "big_mood_detector.domain.services.mood_predictor.MoodPredictor.is_loaded",
                new_callable=PropertyMock,
                return_value=True,
            ),
        ):
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble

            mock_ensemble.predict.return_value = EnsemblePrediction(
                xgboost_prediction=MoodPrediction(0.1, 0.1, 0.1, 0.8),
                pat_enhanced_prediction=MoodPrediction(0.1, 0.1, 0.1, 0.8),
                ensemble_prediction=MoodPrediction(0.1, 0.1, 0.1, 0.8),
                models_used=["xgboost", "pat"],
                confidence_scores={"xgboost": 0.8, "pat": 0.8},
                processing_time_ms={"xgboost": 5, "pat": 5},
            )

            pipeline = MoodPredictionPipeline(config=config)

            # Mock the clinical feature extractor
            with patch.object(
                pipeline.clinical_extractor, "extract_clinical_features"
            ) as mock_extract:
                # Create mock features
                import numpy as np

                from big_mood_detector.domain.services.clinical_feature_extractor import (
                    ClinicalFeatureSet,
                    SeoulXGBoostFeatures,
                )

                mock_features = Mock(spec=ClinicalFeatureSet)
                mock_seoul = Mock(spec=SeoulXGBoostFeatures)
                mock_seoul.to_xgboost_features.return_value = np.zeros(36)
                mock_features.seoul_features = mock_seoul
                mock_extract.return_value = mock_features

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
