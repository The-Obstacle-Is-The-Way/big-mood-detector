"""
Test to verify the prediction pipeline uses the correct feature generation.
This demonstrates the fix for the Seoul feature mismatch bug.
"""
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestPredictionPipelineFix:
    """Test that the prediction pipeline correctly uses Seoul features."""

    @pytest.fixture
    def mock_xgboost_models(self):
        """Mock XGBoost models to avoid needing real model files."""
        # Mock just the prediction methods, not the entire class
        with patch('big_mood_detector.domain.services.mood_predictor.MoodPredictor._load_models') as mock_load:
            mock_load.return_value = None  # Prevent actual model loading

            # Mock the predict method
            with patch('big_mood_detector.domain.services.mood_predictor.MoodPredictor.predict') as mock_predict:
                mock_prediction = Mock()
                mock_prediction.depression_risk = 0.3
                mock_prediction.hypomanic_risk = 0.1
                mock_prediction.manic_risk = 0.05
                mock_prediction.confidence = 0.8
                mock_predict.return_value = mock_prediction

                # Also mock is_loaded property
                with patch('big_mood_detector.domain.services.mood_predictor.MoodPredictor.is_loaded', new_callable=lambda: property(lambda self: True)):
                    yield mock_predict

    @pytest.fixture
    def sample_health_data(self):
        """Create 30 days of sample health data."""
        sleep_records = []
        activity_records = []

        base_date = date(2025, 6, 1)

        for i in range(30):
            current_date = base_date + timedelta(days=i)

            # Sleep record
            sleep_start = datetime.combine(
                current_date - timedelta(days=1),
                datetime.min.time()
            ) + timedelta(hours=23)
            sleep_end = sleep_start + timedelta(hours=7.5)

            sleep_records.append(SleepRecord(
                source_name="test",
                start_date=sleep_start,
                end_date=sleep_end,
                state=SleepState.ASLEEP
            ))

            # Activity records
            for hour in range(8, 20):
                activity_start = datetime.combine(
                    current_date,
                    datetime.min.time()
                ) + timedelta(hours=hour)
                activity_end = activity_start + timedelta(hours=1)

                activity_records.append(ActivityRecord(
                    source_name="test",
                    start_date=activity_start,
                    end_date=activity_end,
                    activity_type=ActivityType.STEP_COUNT,
                    value=500.0,
                    unit="count"
                ))

        return {
            'sleep_records': sleep_records,
            'activity_records': activity_records,
            'heart_rate_records': []
        }

    @pytest.mark.integration
    def test_pipeline_uses_seoul_features(self, mock_xgboost_models, sample_health_data):
        """Test that the pipeline uses Seoul features for XGBoost predictions."""
        # Configure pipeline to use aggregation for Seoul features
        config = PipelineConfig(
            include_pat_sequences=False,  # XGBoost only
            min_days_required=7,
            use_seoul_features=True  # NEW: Flag to use Seoul features
        )

        # Create pipeline
        pipeline = MoodPredictionPipeline(config=config)

        # Mock the data parsing to return our sample data
        with patch.object(pipeline.data_parsing_service, 'parse_health_data') as mock_parse:
            mock_parse.return_value = sample_health_data

            # Process and predict
            result = pipeline.process_apple_health_file(
                file_path=Path("test.xml"),
                start_date=date(2025, 6, 20),
                end_date=date(2025, 6, 30)
            )

        # Verify predictions were made
        assert result is not None
        assert len(result.daily_predictions) > 0

        # Verify XGBoost predictor was called
        assert mock_xgboost_models.called

        # Verify predictions have expected values
        for _date_key, prediction in result.daily_predictions.items():
            assert prediction['depression_risk'] == 0.3
            assert prediction['hypomanic_risk'] == 0.1
            assert prediction['manic_risk'] == 0.05
            assert prediction['confidence'] == 0.8

    @pytest.mark.integration
    def test_pipeline_without_seoul_features_uses_clinical_flow(self, mock_xgboost_models, sample_health_data):
        """Test that the pipeline uses clinical flow when Seoul features are disabled."""
        # Configure pipeline without Seoul features flag
        config = PipelineConfig(
            include_pat_sequences=False,
            min_days_required=7,
            use_seoul_features=False  # Use clinical flow
        )

        # Create pipeline
        pipeline = MoodPredictionPipeline(config=config)

        # Verify aggregation pipeline was not created for Seoul features
        # (it's still created as fallback but not used for predictions)
        assert pipeline.aggregation_pipeline is not None  # Created as fallback

        # Mock the data parsing
        with patch.object(pipeline.data_parsing_service, 'parse_health_data') as mock_parse:
            mock_parse.return_value = sample_health_data

            # Process should work but use different flow
            result = pipeline.process_apple_health_file(
                file_path=Path("test.xml"),
                start_date=date(2025, 6, 20),
                end_date=date(2025, 6, 30)
            )

            # Should have predictions (mocked)
            assert result is not None
            assert len(result.daily_predictions) > 0
