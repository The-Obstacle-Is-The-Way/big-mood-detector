"""
Unit tests for Mood Prediction Pipeline
Tests the full orchestration from Apple Health data to mood predictions.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
    PipelineResult,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestMoodPredictionPipeline:
    """Test the complete mood prediction pipeline."""

    def test_pipeline_initialization(self):
        """Pipeline should initialize with default config."""
        pipeline = MoodPredictionPipeline()

        assert pipeline is not None
        assert pipeline.config.min_days_required == 7
        assert pipeline.config.include_pat_sequences is False
        assert pipeline.config.confidence_threshold == 0.7

    @patch("big_mood_detector.infrastructure.ml_models.pat_model.PATModel")
    def test_pipeline_with_custom_config(self, mock_pat_class):
        """Pipeline should accept custom configuration."""
        # Mock PAT model to avoid loading real weights
        mock_pat = Mock()
        mock_pat.load_pretrained_weights.return_value = True
        mock_pat_class.return_value = mock_pat

        config = PipelineConfig(
            min_days_required=14,
            include_pat_sequences=True,
            confidence_threshold=0.8,
            model_dir=Path("/custom/models"),
        )

        pipeline = MoodPredictionPipeline(config)

        assert pipeline.config.min_days_required == 14
        assert pipeline.config.include_pat_sequences is True
        assert pipeline.config.confidence_threshold == 0.8

    def test_process_apple_health_xml(self):
        """Process Apple Health XML export and generate predictions."""
        # Create test data
        target_date = date(2025, 7, 16)
        sleep_records = self._create_test_sleep_records(target_date, days=14)
        activity_records = self._create_test_activity_records(target_date, days=14)
        heart_records = self._create_test_heart_records(target_date, days=14)

        # Mock DataParsingService
        mock_parsing_service = Mock()
        mock_parsing_service.parse_health_data.return_value = {
            "sleep_records": sleep_records,
            "activity_records": activity_records,
            "heart_rate_records": heart_records,
            "errors": []
        }

        # Create pipeline for testing with mock predictor
        from big_mood_detector.test_support.predictors import ConstantMoodPredictor
        
        pipeline = MoodPredictionPipeline.for_testing(
            predictor=ConstantMoodPredictor(),
            config=PipelineConfig(min_days_required=7),
            disable_ensemble=True
        )
        pipeline.data_parsing_service = mock_parsing_service

        # Act
        result = pipeline.process_apple_health_file(
            file_path=Path("test_export.xml"),
            start_date=target_date - timedelta(days=13),
            end_date=target_date,
        )

        # Assert
        assert result is not None
        # Pipeline needs 7 days of data to make a prediction, so with 14 days of data
        # we expect at least 5 predictions (days 8-14 can have predictions)
        assert len(result.daily_predictions) >= 5
        assert result.overall_summary is not None
        assert 0.0 <= result.confidence_score <= 1.0

        # Check that predictions have required fields
        for _date_key, prediction in result.daily_predictions.items():
            assert "depression_risk" in prediction
            assert "hypomanic_risk" in prediction
            assert "manic_risk" in prediction
            assert "confidence" in prediction

    def test_batch_feature_extraction(self):
        """Extract features for multiple days efficiently."""
        pipeline = MoodPredictionPipeline()

        target_date = date(2025, 7, 16)
        sleep_records = self._create_test_sleep_records(target_date, days=30)
        activity_records = self._create_test_activity_records(target_date, days=30)
        heart_records = self._create_test_heart_records(target_date, days=30)

        # Act
        features = pipeline.extract_features_batch(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            start_date=target_date - timedelta(days=6),
            end_date=target_date,
        )

        # Assert
        assert len(features) == 7  # 7 days of features
        for _date_key, feature_set in features.items():
            assert feature_set.seoul_features is not None
            assert len(feature_set.seoul_features.to_xgboost_features()) == 36

    @patch("big_mood_detector.domain.services.mood_predictor.MoodPredictor.predict")
    def test_prediction_with_insufficient_data(self, mock_predict):
        """Handle cases with insufficient data gracefully."""
        pipeline = MoodPredictionPipeline()

        # Only 3 days of data (less than min_days_required)
        target_date = date(2025, 7, 16)
        sleep_records = self._create_test_sleep_records(target_date, days=3)
        activity_records = self._create_test_activity_records(target_date, days=3)
        heart_records = self._create_test_heart_records(target_date, days=3)

        # Act
        result = pipeline.process_health_data(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
        )

        # Assert
        assert result is not None
        assert result.has_warnings is True
        assert "Insufficient data" in result.warnings[0]
        # With insufficient data, confidence should be reduced
        assert result.confidence_score <= 0.7  # Reduced from 1.0 due to warnings

    def test_pipeline_with_sparse_data(self):
        """Handle sparse data with missing days."""
        pipeline = MoodPredictionPipeline()

        target_date = date(2025, 7, 16)
        # Create sparse data - only every 3rd day has data
        sleep_records = []
        activity_records = []

        for i in range(0, 21, 3):  # Days 0, 3, 6, 9, 12, 15, 18
            day = target_date - timedelta(days=20 - i)
            sleep_records.extend(self._create_test_sleep_records(day, days=1))
            activity_records.extend(self._create_test_activity_records(day, days=1))

        # Act
        result = pipeline.process_health_data(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            target_date=target_date,
        )

        # Assert
        assert result is not None
        assert result.has_warnings is True
        assert any("sparse" in w.lower() for w in result.warnings)

    @patch(
        "big_mood_detector.application.use_cases.process_health_data_use_case.MoodPredictor"
    )
    def test_pipeline_without_models(self, MockPredictor):
        """Handle missing ML models gracefully."""
        # Create mock predictor instance
        mock_predictor_instance = Mock()
        mock_predictor_instance.is_loaded = False
        MockPredictor.return_value = mock_predictor_instance

        pipeline = MoodPredictionPipeline()

        target_date = date(2025, 7, 16)
        sleep_records = self._create_test_sleep_records(target_date, days=14)
        activity_records = self._create_test_activity_records(target_date, days=14)
        heart_records = self._create_test_heart_records(target_date, days=14)

        # Act
        result = pipeline.process_health_data(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
        )

        # Assert
        assert result is not None
        assert result.has_errors is True
        assert "models not loaded" in result.errors[0].lower()

    def test_export_results_to_csv(self, tmp_path):
        """Export pipeline results to CSV format."""
        pipeline = MoodPredictionPipeline()

        # Create mock result
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 14): {
                    "depression_risk": 0.2,
                    "hypomanic_risk": 0.1,
                    "manic_risk": 0.05,
                    "confidence": 0.85,
                },
                date(2025, 7, 15): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.15,
                    "manic_risk": 0.08,
                    "confidence": 0.82,
                },
            },
            overall_summary={
                "avg_depression_risk": 0.25,
                "avg_hypomanic_risk": 0.125,
                "avg_manic_risk": 0.065,
                "days_analyzed": 2,
            },
            confidence_score=0.835,
            processing_time_seconds=1.5,
        )

        # Act
        csv_path = tmp_path / "results.csv"
        pipeline.export_results(result, csv_path)

        # Assert
        assert csv_path.exists()
        with open(csv_path) as f:
            content = f.read()
            assert "2025-07-14" in content
            assert "0.2" in content  # depression risk
            assert "0.85" in content  # confidence

    def test_pipeline_performance_metrics(self):
        """Track pipeline performance metrics."""
        pipeline = MoodPredictionPipeline()

        target_date = date(2025, 7, 16)
        sleep_records = self._create_test_sleep_records(target_date, days=30)
        activity_records = self._create_test_activity_records(target_date, days=30)
        heart_records = self._create_test_heart_records(target_date, days=30)

        # Act
        result = pipeline.process_health_data(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
        )

        # Assert
        assert result.processing_time_seconds > 0
        assert result.records_processed > 0
        assert result.features_extracted > 0

    # Helper methods
    def _create_test_sleep_records(
        self, end_date: date, days: int
    ) -> list[SleepRecord]:
        """Create test sleep records."""
        records = []
        for i in range(days):
            day = end_date - timedelta(days=days - 1 - i)
            records.append(
                SleepRecord(
                    source_name="test",
                    start_date=datetime.combine(day, datetime.min.time()).replace(
                        hour=23
                    ),
                    end_date=datetime.combine(
                        day + timedelta(days=1), datetime.min.time()
                    ).replace(hour=7),
                    state=SleepState.ASLEEP,
                )
            )
        return records

    def _create_test_activity_records(
        self, end_date: date, days: int
    ) -> list[ActivityRecord]:
        """Create test activity records."""
        records = []
        for i in range(days):
            day = end_date - timedelta(days=days - 1 - i)
            # Create hourly activity
            for hour in range(24):
                value = 100 if 8 <= hour <= 20 else 10  # Active during day
                records.append(
                    ActivityRecord(
                        source_name="test",
                        start_date=datetime.combine(day, datetime.min.time()).replace(
                            hour=hour
                        ),
                        end_date=datetime.combine(day, datetime.min.time()).replace(
                            hour=hour, minute=59
                        ),
                        activity_type=ActivityType.STEP_COUNT,
                        value=value,
                        unit="steps",
                    )
                )
        return records

    def _create_test_heart_records(
        self, end_date: date, days: int
    ) -> list[HeartRateRecord]:
        """Create test heart rate records."""
        records = []
        for i in range(days):
            day = end_date - timedelta(days=days - 1 - i)
            # Create records every 4 hours
            for hour in [0, 4, 8, 12, 16, 20]:
                hr = 55 if 0 <= hour < 8 else 70
                records.append(
                    HeartRateRecord(
                        source_name="test",
                        timestamp=datetime.combine(day, datetime.min.time()).replace(
                            hour=hour
                        ),
                        metric_type=HeartMetricType.HEART_RATE,
                        value=hr,
                        unit="bpm",
                    )
                )
        return records
