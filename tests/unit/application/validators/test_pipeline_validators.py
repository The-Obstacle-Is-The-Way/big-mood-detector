"""
Unit tests for pipeline validators.

These tests verify that PAT and XGBoost validators correctly
assess data sufficiency for their respective models.
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from big_mood_detector.application.validators.pipeline_validators import (
    PATValidator,
    ValidationResult,
    XGBoostValidator,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestPATValidator:
    """Test cases for PAT model validator."""

    @pytest.fixture
    def validator(self) -> PATValidator:
        """Create PAT validator instance."""
        return PATValidator()

    def test_pat_requires_exactly_7_consecutive_days(
        self, validator: PATValidator
    ) -> None:
        """Test that PAT requires exactly 7 consecutive days of data."""
        # Create exactly 7 consecutive days of activity data
        base_date = date(2025, 7, 20)
        records = []
        
        for day_offset in range(7):
            activity_date = datetime(
                base_date.year,
                base_date.month,
                base_date.day + day_offset,
                12, 0, 0,
                tzinfo=UTC
            )
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Validate
        result = validator.validate(
            activity_records=records,
            start_date=base_date,
            end_date=base_date + timedelta(days=6),
        )
        
        assert result.is_valid is True
        assert result.can_run is True
        assert result.consecutive_days == 7
        assert result.days_available == 7
        assert len(result.missing_data) == 0
        assert "7 consecutive days" in result.message

    def test_pat_fails_with_gaps_in_data(
        self, validator: PATValidator
    ) -> None:
        """Test that PAT fails when there are gaps in the data."""
        base_date = date(2025, 7, 20)
        records = []
        
        # Create data with a gap on day 3
        for day_offset in [0, 1, 2, 4, 5, 6, 7]:  # Missing day 3
            activity_date = datetime(
                base_date.year,
                base_date.month,
                base_date.day + day_offset,
                12, 0, 0,
                tzinfo=UTC
            )
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        result = validator.validate(
            activity_records=records,
            start_date=base_date,
            end_date=base_date + timedelta(days=7),
        )
        
        assert result.is_valid is False
        assert result.can_run is False
        assert result.consecutive_days == 4  # Longest consecutive run (days 4-7)
        assert result.days_available == 7  # Total days with data
        assert len(result.missing_data) > 0
        assert "Need 3 more consecutive days" in result.missing_data[0]

    def test_pat_finds_best_consecutive_window(
        self, validator: PATValidator
    ) -> None:
        """Test that PAT can find a valid 7-day window within sparse data."""
        base_date = date(2025, 7, 1)
        records = []
        
        # Create sparse data with a 7-day consecutive window in the middle
        # Days 1-3: sparse
        for day in [1, 3]:
            activity_date = datetime(2025, 7, day, 12, 0, 0, tzinfo=UTC)
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Days 10-16: consecutive (7 days)
        for day in range(10, 17):
            activity_date = datetime(2025, 7, day, 12, 0, 0, tzinfo=UTC)
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Days 20, 22: sparse
        for day in [20, 22]:
            activity_date = datetime(2025, 7, day, 12, 0, 0, tzinfo=UTC)
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        result = validator.validate(
            activity_records=records,
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 31),
        )
        
        assert result.is_valid is True
        assert result.can_run is True
        assert result.consecutive_days == 7
        assert result.days_available == 11  # Total days with data

    def test_pat_insufficient_consecutive_days(
        self, validator: PATValidator
    ) -> None:
        """Test PAT validation with only 5 consecutive days."""
        base_date = date(2025, 7, 20)
        records = []
        
        # Only 5 consecutive days
        for day_offset in range(5):
            activity_date = datetime(
                base_date.year,
                base_date.month,
                base_date.day + day_offset,
                12, 0, 0,
                tzinfo=UTC
            )
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        result = validator.validate(
            activity_records=records,
            start_date=base_date,
            end_date=base_date + timedelta(days=4),
        )
        
        assert result.is_valid is False
        assert result.can_run is False
        assert result.consecutive_days == 5
        assert "found 5" in result.message


class TestXGBoostValidator:
    """Test cases for XGBoost model validator."""

    @pytest.fixture
    def validator(self) -> XGBoostValidator:
        """Create XGBoost validator instance."""
        return XGBoostValidator()

    def test_xgboost_accepts_sparse_30_days(
        self, validator: XGBoostValidator
    ) -> None:
        """Test that XGBoost works with 30+ sparse days."""
        # Create 35 days of sparse data (every other day)
        sleep_records = []
        activity_records = []
        
        for day in range(0, 70, 2):  # Every other day for 35 days
            # Sleep record
            sleep_date = datetime(2025, 7, 1, 22, 0, 0, tzinfo=UTC) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="iPhone",
                    start_date=sleep_date,
                    end_date=sleep_date + timedelta(hours=8),
                    state=SleepState.ASLEEP,
                )
            )
            
            # Activity record
            activity_date = datetime(2025, 7, 1, 12, 0, 0, tzinfo=UTC) + timedelta(days=day)
            activity_records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        result = validator.validate(
            sleep_records=sleep_records,
            activity_records=activity_records,
            start_date=date(2025, 7, 1),
            end_date=date(2025, 9, 8),
        )
        
        assert result.is_valid is True
        assert result.can_run is True
        assert result.days_available == 35
        assert result.consecutive_days == 0  # Not required for XGBoost
        assert len(result.missing_data) == 0
        assert "found 35" in result.message

    def test_xgboost_fails_with_insufficient_days(
        self, validator: XGBoostValidator
    ) -> None:
        """Test that XGBoost fails with less than 30 days."""
        # Create only 20 days of data
        sleep_records = []
        
        for day in range(20):
            sleep_date = datetime(2025, 7, 1, 22, 0, 0, tzinfo=UTC) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="iPhone",
                    start_date=sleep_date,
                    end_date=sleep_date + timedelta(hours=8),
                    state=SleepState.ASLEEP,
                )
            )
        
        result = validator.validate(
            sleep_records=sleep_records,
            activity_records=[],
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 20),
        )
        
        assert result.is_valid is False
        assert result.can_run is False
        assert result.days_available == 20
        assert len(result.missing_data) > 0
        assert "Need 10 more days" in result.missing_data[0]

    def test_xgboost_optimal_with_60_days(
        self, validator: XGBoostValidator
    ) -> None:
        """Test that XGBoost recognizes optimal 60+ days."""
        # Create 65 days of data
        activity_records = []
        
        for day in range(65):
            activity_date = datetime(2025, 5, 1, 12, 0, 0, tzinfo=UTC) + timedelta(days=day)
            activity_records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        result = validator.validate(
            sleep_records=[],
            activity_records=activity_records,
            start_date=date(2025, 5, 1),
            end_date=date(2025, 7, 4),
        )
        
        assert result.is_valid is True
        assert result.can_run is True
        assert result.days_available == 65
        assert "found 65" in result.message

    def test_xgboost_accepts_mixed_data_sources(
        self, validator: XGBoostValidator
    ) -> None:
        """Test that XGBoost works with some days having only sleep or activity."""
        sleep_records = []
        activity_records = []
        
        # Days 1-15: Only sleep data
        for day in range(15):
            sleep_date = datetime(2025, 7, 1, 22, 0, 0, tzinfo=UTC) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="iPhone",
                    start_date=sleep_date,
                    end_date=sleep_date + timedelta(hours=8),
                    state=SleepState.ASLEEP,
                )
            )
        
        # Days 10-25: Only activity data (overlap with sleep on days 10-15)
        for day in range(10, 25):
            activity_date = datetime(2025, 7, 1, 12, 0, 0, tzinfo=UTC) + timedelta(days=day)
            activity_records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Days 20-35: Both sleep and activity
        for day in range(20, 35):
            sleep_date = datetime(2025, 7, 1, 22, 0, 0, tzinfo=UTC) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="iPhone",
                    start_date=sleep_date,
                    end_date=sleep_date + timedelta(hours=8),
                    state=SleepState.ASLEEP,
                )
            )
        
        result = validator.validate(
            sleep_records=sleep_records,
            activity_records=activity_records,
            start_date=date(2025, 7, 1),
            end_date=date(2025, 8, 4),
        )
        
        # Total unique days: 1-35 = 35 days
        assert result.is_valid is True
        assert result.can_run is True
        assert result.days_available == 35