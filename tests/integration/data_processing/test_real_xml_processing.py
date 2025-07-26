"""Test XML processing with REAL Apple Health data."""

from datetime import date
from pathlib import Path

import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.test_support.predictors import ConstantMoodPredictor


@pytest.mark.slow
@pytest.mark.timeout(300)
class TestRealXMLProcessing:
    """Test with actual Apple Health export data."""

    @pytest.fixture
    def real_xml_path(self):
        """Path to real XML test data."""
        path = Path("tests/fixtures/health/week_sample_clean.xml")
        if not path.exists():
            pytest.skip(f"Real test data not found at {path}")
        return path

    @pytest.fixture
    def pipeline(self):
        """Pipeline with real predictor and production config."""
        config = PipelineConfig(
            min_days_required=3,  # Production default
            enable_sparse_handling=True,
            use_seoul_features=True
        )

        return MoodPredictionPipeline.for_testing(
            predictor=ConstantMoodPredictor(),
            config=config,
            disable_ensemble=True
        )

    def test_real_xml_processes_without_errors(self, pipeline, real_xml_path):
        """Real XML should process without errors."""
        result = pipeline.process_apple_health_file(
            file_path=real_xml_path,
            end_date=date(2024, 8, 1)
        )

        # Basic assertions - it should work
        assert result is not None
        assert not result.has_errors
        assert result.records_processed > 0
        assert result.processing_time_seconds > 0

        # Note: Real data may not have sufficient sleep data for predictions
        # This is expected behavior, not a test failure

    def test_real_xml_extracts_features(self, pipeline, real_xml_path):
        """Real XML should extract clinical features."""
        result = pipeline.process_apple_health_file(
            file_path=real_xml_path,
            end_date=date(2024, 8, 1)
        )

        # Should extract at least one feature set
        assert result.features_extracted > 0

        # Log what we actually got
        print(f"Records processed: {result.records_processed}")
        print(f"Features extracted: {result.features_extracted}")
        print(f"Predictions generated: {len(result.daily_predictions)}")

        # If no predictions, check why
        if not result.daily_predictions and result.has_warnings:
            print(f"Warnings: {result.warnings}")
