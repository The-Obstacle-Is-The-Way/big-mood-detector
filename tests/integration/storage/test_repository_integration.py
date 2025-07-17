"""
Repository Integration Tests

Test multi-repository workflows and data consistency.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.value_objects.time_period import TimePeriod
from big_mood_detector.infrastructure.repositories.file_activity_repository import (
    FileActivityRepository,
)
from big_mood_detector.infrastructure.repositories.file_heart_rate_repository import (
    FileHeartRateRepository,
)
from big_mood_detector.infrastructure.repositories.file_sleep_repository import (
    FileSleepRepository,
)


class TestRepositoryIntegration:
    """Integration tests for multiple repositories working together."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def repositories(self, temp_dir):
        """Create all repository instances."""
        return {
            "sleep": FileSleepRepository(data_dir=temp_dir),
            "activity": FileActivityRepository(data_dir=temp_dir),
            "heart_rate": FileHeartRateRepository(data_dir=temp_dir),
        }

    async def test_repository_initialization_creates_separate_directories(
        self, repositories, temp_dir
    ):
        """Test that each repository creates its own subdirectory."""
        # Verify directories were created
        assert (temp_dir / "sleep_records").exists()
        assert (temp_dir / "activity_records").exists()
        assert (temp_dir / "heart_rate_records").exists()

    async def test_cross_repository_data_persistence(self, repositories):
        """Test that data persists correctly across different repositories."""
        sleep_repo = repositories["sleep"]
        activity_repo = repositories["activity"]
        hr_repo = repositories["heart_rate"]

        # Create related records for the same time period
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        sleep_record = SleepRecord(
            source_name="apple-watch",
            start_date=timestamp - timedelta(hours=8),
            end_date=timestamp,
            state=SleepState.ASLEEP,
        )

        activity_record = ActivityRecord(
            source_name="apple-watch",
            start_date=timestamp,
            end_date=timestamp + timedelta(hours=1),
            activity_type=ActivityType.STEP_COUNT,
            value=2500.0,
            unit="count",
        )

        hr_record = HeartRateRecord(
            source_name="apple-watch",
            timestamp=timestamp + timedelta(minutes=30),
            metric_type=HeartMetricType.HEART_RATE,
            value=75.0,
            unit="count/min",
            motion_context=MotionContext.ACTIVE,
        )

        # Save records to different repositories
        await sleep_repo.save(sleep_record)
        await activity_repo.save(activity_record)
        await hr_repo.save(hr_record)

        # Verify records can be retrieved
        period = TimePeriod(
            start=timestamp - timedelta(hours=12), end=timestamp + timedelta(hours=2)
        )

        sleep_records = await sleep_repo.get_by_period(period)
        activity_records = await activity_repo.get_by_period(period)
        hr_records = await hr_repo.get_by_period(period)

        assert len(sleep_records) == 1
        assert len(activity_records) == 1
        assert len(hr_records) == 1

    async def test_batch_operations_across_repositories(self, repositories):
        """Test batch operations work consistently across repositories."""
        sleep_repo = repositories["sleep"]
        activity_repo = repositories["activity"]

        # Create multiple related records
        base_time = datetime(2024, 1, 1, 22, 0, 0)
        sleep_records = []
        activity_records = []

        for i in range(3):
            day_offset = timedelta(days=i)

            sleep_record = SleepRecord(
                source_name=f"device-{i}",
                start_date=base_time + day_offset,
                end_date=base_time + day_offset + timedelta(hours=8),
                state=SleepState.ASLEEP,
            )
            sleep_records.append(sleep_record)

            activity_record = ActivityRecord(
                source_name=f"device-{i}",
                start_date=base_time + day_offset + timedelta(hours=10),
                end_date=base_time + day_offset + timedelta(hours=11),
                activity_type=ActivityType.STEP_COUNT,
                value=1000.0 * (i + 1),
                unit="count",
            )
            activity_records.append(activity_record)

        # Save batches
        await sleep_repo.save_batch(sleep_records)
        await activity_repo.save_batch(activity_records)

        # Verify all records saved
        all_sleep = await sleep_repo.get_latest(limit=10)
        all_activity = await activity_repo.get_latest(limit=10)

        assert len(all_sleep) == 3
        assert len(all_activity) == 3
