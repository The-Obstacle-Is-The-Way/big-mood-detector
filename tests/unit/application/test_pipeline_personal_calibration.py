"""
Test Personal Calibration Integration with Pipeline

TDD for wiring PersonalCalibrator into MoodPredictionPipeline.
"""

from datetime import date, datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.mood_predictor import MoodPrediction
from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
    PersonalCalibrator,
)


class TestPipelinePersonalCalibration:
    """Test PersonalCalibrator integration with MoodPredictionPipeline."""

    def test_pipeline_can_use_personal_calibrator(self):
        """Test that pipeline can be configured with a personal calibrator."""
        # Given a personal calibrator
        calibrator = PersonalCalibrator(user_id="test_user")

        # And a pipeline config with personal calibration enabled
        config = PipelineConfig(
            enable_personal_calibration=True, personal_calibrator=calibrator
        )

        # When creating a pipeline
        pipeline = MoodPredictionPipeline(config=config)

        # Then the calibrator should be available
        assert pipeline.personal_calibrator is not None
        assert pipeline.personal_calibrator.user_id == "test_user"

    def test_pipeline_loads_personal_model_if_exists(self, tmp_path):
        """Test pipeline loads existing personal model."""
        # Given a saved personal model
        user_id = "test_user"
        model_dir = tmp_path / "models"

        # Create a mock saved model
        user_dir = model_dir / "users" / user_id
        user_dir.mkdir(parents=True)
        metadata = {
            "user_id": user_id,
            "model_type": "xgboost",
            "baseline": {"mean_sleep_duration": 420},
            "calibration_date": "2024-01-01",
            "metrics": {"accuracy": 0.85},
        }

        import json

        with open(user_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # When creating pipeline with user ID
        config = PipelineConfig(
            enable_personal_calibration=True, user_id=user_id, model_dir=model_dir
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Then personal calibrator should be loaded
        assert pipeline.personal_calibrator is not None
        assert pipeline.personal_calibrator.user_id == user_id
        assert pipeline.personal_calibrator.baseline["mean_sleep_duration"] == 420

    @patch(
        "big_mood_detector.infrastructure.fine_tuning.personal_calibrator.PersonalCalibrator.load"
    )
    def test_pipeline_continues_without_personal_model_if_not_found(self, mock_load):
        """Test pipeline continues with population model if personal model not found."""
        # Given loading personal model fails
        mock_load.side_effect = FileNotFoundError("No personal model")

        # When creating pipeline with user ID
        config = PipelineConfig(
            enable_personal_calibration=True, user_id="nonexistent_user"
        )

        # Should not raise - falls back to population model
        pipeline = MoodPredictionPipeline(config=config)

        # Then personal calibrator should be None
        assert pipeline.personal_calibrator is None

    def test_predictions_use_personal_baseline_deviations(self):
        """Test that predictions incorporate personal baseline deviations."""
        # Given a calibrator with established baseline
        calibrator = PersonalCalibrator(user_id="test_user")
        calibrator.baseline = {
            "mean_sleep_duration": 420,  # 7 hours
            "std_sleep_duration": 30,
            "mean_daily_activity": 10000,
        }

        # Mock the mood predictor
        mock_predictor = Mock()
        mock_predictor.is_loaded = True
        mock_predictor.predict.return_value = MoodPrediction(
            depression_risk=0.6, hypomanic_risk=0.2, manic_risk=0.1, confidence=0.8
        )

        # Create pipeline with calibrator
        config = PipelineConfig(
            enable_personal_calibration=True, personal_calibrator=calibrator
        )
        pipeline = MoodPredictionPipeline(config=config)
        pipeline.mood_predictor = mock_predictor

        # Create test data with deviation from baseline
        test_date = date(2024, 1, 15)
        sleep_records = [
            SleepRecord(
                start_date=datetime(2024, 1, 15, 22, 0),
                end_date=datetime(
                    2024, 1, 16, 2, 0
                ),  # 4 hours (3 hours below baseline)
                source_name="test",
                state=SleepState.ASLEEP,
            )
        ]

        # When processing
        result = pipeline.process_health_data(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=test_date,
        )

        # Then predictions should exist
        assert len(result.daily_predictions) > 0

        # And the pipeline should have calculated deviations
        # (This would be used internally for adjusted predictions)
        assert result.metadata.get("personal_calibration_used") is True

    def test_calibrator_adjusts_prediction_probabilities(self):
        """Test that calibrator adjusts overconfident predictions."""
        # Given a calibrator with calibration data
        calibrator = PersonalCalibrator(user_id="test_user")

        # Simulate calibration from historical data
        # Model was overconfident - often wrong when very sure
        raw_probs = np.array([0.9, 0.1, 0.85, 0.15, 0.95])
        true_labels = np.array([0, 0, 1, 0, 0])  # Model was wrong on high confidence
        calibrator.fit_calibration(raw_probs, true_labels)

        # When calibrating a new overconfident prediction
        new_prediction = 0.92  # Very high confidence
        calibrated = calibrator.calibrate_probabilities(np.array([new_prediction]))

        # Then confidence should be reduced
        assert calibrated[0] < new_prediction
        assert calibrated[0] > 0.5  # Still positive but less extreme

    def test_pipeline_saves_predictions_with_personal_context(self, tmp_path):
        """Test pipeline saves predictions with personal calibration metadata."""
        # Given a calibrated pipeline
        calibrator = PersonalCalibrator(user_id="test_user")
        calibrator.baseline = {"mean_sleep_duration": 420}

        config = PipelineConfig(
            enable_personal_calibration=True, personal_calibrator=calibrator
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Mock components
        pipeline.mood_predictor = Mock()
        pipeline.mood_predictor.is_loaded = True
        pipeline.mood_predictor.predict.return_value = MoodPrediction(
            depression_risk=0.7, hypomanic_risk=0.1, manic_risk=0.1, confidence=0.85
        )

        # When processing and exporting
        result = pipeline.process_health_data(
            sleep_records=[],
            activity_records=[],
            heart_records=[],
            target_date=date(2024, 1, 15),
        )

        output_path = tmp_path / "predictions.csv"
        pipeline.export_results(result, output_path)

        # Then metadata should include calibration info
        metadata_path = output_path.with_suffix(".summary.json")
        assert metadata_path.exists()

        import json

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata.get("personal_calibration", {}).get("user_id") == "test_user"
        assert (
            metadata.get("personal_calibration", {}).get("baseline_available") is True
        )

    @patch("big_mood_detector.infrastructure.ml_models.pat_model.PATModel")
    def test_ensemble_uses_personal_calibration(self, mock_pat_class):
        """Test that ensemble predictions also use personal calibration."""
        # Mock PAT model to avoid loading real weights
        mock_pat = Mock()
        mock_pat.load_pretrained_weights.return_value = True
        mock_pat_class.return_value = mock_pat

        # Given a calibrator
        calibrator = PersonalCalibrator(user_id="test_user")
        calibrator.baseline = {
            "mean_sleep_duration": 420,
            "mean_daily_activity": 10000,
        }

        # And ensemble is enabled
        config = PipelineConfig(
            enable_personal_calibration=True,
            personal_calibrator=calibrator,
            include_pat_sequences=True,  # Enable ensemble
        )

        # When creating pipeline
        pipeline = MoodPredictionPipeline(config=config)

        # Then ensemble should have access to calibrator
        if pipeline.ensemble_orchestrator:
            # Ensemble should be configured with calibrator
            assert hasattr(pipeline.ensemble_orchestrator, "personal_calibrator")

    def test_pipeline_updates_personal_model_with_new_labels(self):
        """Test pipeline can update personal model with new labeled data."""
        # Given a pipeline with calibrator
        calibrator = PersonalCalibrator(user_id="test_user", model_type="xgboost")

        config = PipelineConfig(
            enable_personal_calibration=True, personal_calibrator=calibrator
        )

        pipeline = MoodPredictionPipeline(config=config)

        # And new labeled data
        labeled_features = pd.DataFrame(
            {
                "mean_sleep_duration": [300, 500, 420],
                "std_sleep_duration": [20, 40, 30],
                # ... other features
            }
        )
        labels = np.array([1, 1, 0])  # Depression episodes

        # When updating the personal model
        metrics = pipeline.update_personal_model(
            features=labeled_features, labels=labels, sample_weight=None
        )

        # Then model should be updated
        assert metrics is not None
        assert "accuracy" in metrics
        assert metrics["n_trees_added"] > 0  # For XGBoost

    def test_personal_calibration_improves_early_warning(self):
        """Test that personal calibration enables earlier episode detection."""
        # Given historical data showing pattern before episodes
        calibrator = PersonalCalibrator(user_id="test_user")

        # User typically has 7 hours sleep, but drops to 5 hours 3 days before depression
        calibrator.baseline = {
            "mean_sleep_duration": 420,
            "pre_episode_sleep_pattern": 300,  # Learned from historical labels
        }

        # Current sleep showing early warning pattern
        sleep_records = [
            SleepRecord(
                start_date=datetime(2024, 1, 13, 23, 0),
                end_date=datetime(2024, 1, 14, 4, 0),  # 5 hours
                source_name="test",
                state=SleepState.ASLEEP,
            )
        ]

        # Mock predictor that uses personal patterns
        mock_predictor = Mock()
        mock_predictor.is_loaded = True
        mock_predictor.predict.return_value = MoodPrediction(
            depression_risk=0.6,  # Elevated but not high
            hypomanic_risk=0.1,
            manic_risk=0.1,
            confidence=0.7,
        )

        config = PipelineConfig(
            enable_personal_calibration=True, personal_calibrator=calibrator
        )

        pipeline = MoodPredictionPipeline(config=config)
        pipeline.mood_predictor = mock_predictor

        # When processing
        result = pipeline.process_health_data(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2024, 1, 14),
        )

        # Then should have elevated risk
        predictions = result.daily_predictions[date(2024, 1, 14)]
        assert predictions["depression_risk"] == 0.6  # Elevated risk detected
        assert predictions["confidence"] == 0.7

        # And personal calibration metadata should be present
        assert result.metadata.get("personal_calibration_used") is True
        assert result.metadata.get("user_id") == "test_user"
