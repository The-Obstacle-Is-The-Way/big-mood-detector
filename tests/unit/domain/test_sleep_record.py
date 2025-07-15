"""
Tests for Sleep Record Domain Entity

Following TDD and ensuring business rules are enforced.
"""

import pytest
from datetime import datetime, timezone, timedelta

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestSleepState:
    """Test suite for SleepState enum."""
    
    def test_sleep_state_from_healthkit_value(self):
        """Test conversion from HealthKit string values."""
        # ARRANGE & ACT & ASSERT
        assert SleepState.from_healthkit_value("HKCategoryValueSleepAnalysisInBed") == SleepState.IN_BED
        assert SleepState.from_healthkit_value("HKCategoryValueSleepAnalysisAsleep") == SleepState.ASLEEP
        assert SleepState.from_healthkit_value("HKCategoryValueSleepAnalysisREM") == SleepState.REM
        assert SleepState.from_healthkit_value("HKCategoryValueSleepAnalysisDeep") == SleepState.DEEP
    
    def test_invalid_healthkit_value_raises_error(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError, match="Unknown sleep state"):
            SleepState.from_healthkit_value("InvalidValue")


class TestSleepRecord:
    """Test suite for SleepRecord entity."""
    
    def test_create_valid_sleep_record(self):
        """Test creating a valid sleep record."""
        # ARRANGE
        start = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc)
        
        # ACT
        record = SleepRecord(
            source_name="Apple Watch",
            start_date=start,
            end_date=end,
            state=SleepState.ASLEEP
        )
        
        # ASSERT
        assert record.source_name == "Apple Watch"
        assert record.start_date == start
        assert record.end_date == end
        assert record.state == SleepState.ASLEEP
    
    def test_sleep_record_is_immutable(self):
        """Test that sleep record cannot be modified after creation."""
        # ARRANGE
        record = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc),
            state=SleepState.ASLEEP
        )
        
        # ACT & ASSERT
        with pytest.raises(AttributeError):
            record.source_name = "iPhone"
    
    def test_invalid_date_range_raises_error(self):
        """Test that end date must be after start date."""
        # ARRANGE
        start = datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="End date must be after start date"):
            SleepRecord(
                source_name="Apple Watch",
                start_date=start,
                end_date=end,
                state=SleepState.ASLEEP
            )
    
    def test_empty_source_name_raises_error(self):
        """Test that empty source name is not allowed."""
        # ARRANGE
        start = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc)
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Source name is required"):
            SleepRecord(
                source_name="",
                start_date=start,
                end_date=end,
                state=SleepState.ASLEEP
            )
    
    def test_duration_hours_calculation(self):
        """Test duration calculation in hours."""
        # ARRANGE
        start = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 7, 30, tzinfo=timezone.utc)
        record = SleepRecord(
            source_name="Apple Watch",
            start_date=start,
            end_date=end,
            state=SleepState.ASLEEP
        )
        
        # ACT & ASSERT
        assert record.duration_hours == 8.5
    
    def test_is_actual_sleep_property(self):
        """Test identification of actual sleep vs in bed."""
        # ARRANGE
        base_start = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        base_end = datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc)
        
        # ACT & ASSERT
        asleep = SleepRecord("Watch", base_start, base_end, SleepState.ASLEEP)
        assert asleep.is_actual_sleep is True
        
        rem = SleepRecord("Watch", base_start, base_end, SleepState.REM)
        assert rem.is_actual_sleep is True
        
        deep = SleepRecord("Watch", base_start, base_end, SleepState.DEEP)
        assert deep.is_actual_sleep is True
        
        in_bed = SleepRecord("Watch", base_start, base_end, SleepState.IN_BED)
        assert in_bed.is_actual_sleep is False
        
        awake = SleepRecord("Watch", base_start, base_end, SleepState.AWAKE)
        assert awake.is_actual_sleep is False
    
    def test_sleep_quality_indicator(self):
        """Test sleep quality categorization."""
        # ARRANGE
        base_start = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        base_end = datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc)
        
        # ACT & ASSERT
        deep = SleepRecord("Watch", base_start, base_end, SleepState.DEEP)
        assert deep.sleep_quality_indicator == "restorative"
        
        rem = SleepRecord("Watch", base_start, base_end, SleepState.REM)
        assert rem.sleep_quality_indicator == "rem"
        
        awake = SleepRecord("Watch", base_start, base_end, SleepState.AWAKE)
        assert awake.sleep_quality_indicator == "disrupted"