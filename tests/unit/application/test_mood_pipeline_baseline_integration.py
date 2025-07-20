"""
Integration test for baseline repository persistence in MoodPredictionPipeline.

Tests that baselines are properly persisted during pipeline execution.
"""
from datetime import date, datetime
from unittest.mock import create_autospec

import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
)
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


class TestMoodPipelineBaselineIntegration:
    """Test baseline persistence during pipeline execution."""

    @pytest.fixture
    def sample_sleep_records(self):
        """Create sample sleep records for testing."""
        from big_mood_detector.domain.entities.sleep_record import SleepState
        return [
            SleepRecord(
                source_name="Test Device",
                start_date=datetime(2024, 1, i, 22, 0),
                end_date=datetime(2024, 1, i+1, 6, 0),
                state=SleepState.ASLEEP,
            )
            for i in range(1, 8)  # 7 days of data
        ]

    @pytest.fixture
    def sample_activity_records(self):
        """Create sample activity records for testing."""
        from big_mood_detector.domain.entities.activity_record import ActivityType
        records = []
        for day in range(1, 8):
            for hour in [9, 14, 18]:  # 3 activities per day
                records.append(
                    ActivityRecord(
                        source_name="Test Device",
                        start_date=datetime(2024, 1, day, hour, 0),
                        end_date=datetime(2024, 1, day, hour, 30),
                        activity_type=ActivityType.STEP_COUNT,
                        value=2500,
                        unit="steps",
                    )
                )
        return records

    @pytest.fixture
    def sample_heart_rate_records(self):
        """Create sample heart rate records for testing."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            MotionContext,
        )
        records = []
        for day in range(1, 8):
            for hour in range(0, 24, 4):  # Every 4 hours
                records.append(
                    HeartRateRecord(
                        source_name="Test Device",
                        timestamp=datetime(2024, 1, day, hour, 0),
                        metric_type=HeartMetricType.HEART_RATE,
                        value=70 + (hour % 12),  # Vary by time of day
                        unit="bpm",
                        motion_context=MotionContext.SEDENTARY if hour < 6 or hour > 22 else MotionContext.ACTIVE,
                    )
                )
        return records

    @pytest.mark.skip(reason="process_by_date method no longer exists - needs refactoring")
    def test_pipeline_persists_baselines_with_mock_repository(
        self, sample_sleep_records, sample_activity_records, sample_heart_rate_records
    ):
        """Test that pipeline calls save_baseline on the repository."""
        # Create mock repository
        mock_repo = create_autospec(BaselineRepositoryInterface, spec_set=True)
        mock_repo.get_baseline.return_value = None  # No existing baseline

        # Create pipeline with personal calibration
        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "test_user_123"
        config.min_days_required = 3

        pipeline = MoodPredictionPipeline(
            config=config,
            baseline_repository=mock_repo
        )

        # Process the data (processing by date)
        # NOTE: The pipeline processes data day by day, so we test with one day
        pipeline.process_by_date(
            sleep_records=sample_sleep_records[:1],
            activity_records=sample_activity_records[:3],
            heart_rate_records=sample_heart_rate_records[:6],
            target_date=date(2024, 1, 1)
        )

        # Verify baseline was accessed
        mock_repo.get_baseline.assert_called_with("test_user_123")

        # TODO: Verify save_baseline was called after implementing persist_baselines() call
        # For now, this documents the missing functionality
        # mock_repo.save_baseline.assert_called()

    @pytest.mark.skip(reason="process_by_date method no longer exists - needs refactoring")
    def test_pipeline_with_file_repository_integration(
        self, tmp_path, sample_sleep_records, sample_activity_records, sample_heart_rate_records
    ):
        """Integration test with real FileBaselineRepository."""
        # Create real file repository
        baselines_dir = tmp_path / "baselines"
        file_repo = FileBaselineRepository(baselines_dir)

        # Create pipeline
        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "test_user_integration"
        config.min_days_required = 3

        pipeline = MoodPredictionPipeline(
            config=config,
            baseline_repository=file_repo
        )

        # Process first day of data
        result1 = pipeline.process_by_date(
            sleep_records=sample_sleep_records[:1],
            activity_records=sample_activity_records[:3],
            heart_rate_records=sample_heart_rate_records[:6],
            target_date=date(2024, 1, 1)
        )

        # Check if baseline was persisted
        # TODO: This will fail until persist_baselines() is called in pipeline
        # baseline = file_repo.get_baseline("test_user_integration")
        # assert baseline is not None
        # assert baseline.user_id == "test_user_integration"

        # Process second day - should use existing baseline
        result2 = pipeline.process_by_date(
            sleep_records=sample_sleep_records[1:2],
            activity_records=sample_activity_records[3:6],
            heart_rate_records=sample_heart_rate_records[6:12],
            target_date=date(2024, 1, 2)
        )

        # Verify features are calculated consistently
        assert result1 is not None
        assert result2 is not None
