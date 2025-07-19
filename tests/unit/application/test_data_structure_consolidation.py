"""Test data structure consolidation from DailyFeatures to ClinicalFeatureSet."""

from datetime import date, datetime, timedelta

import pytest

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureSet,
)


@pytest.fixture
def sample_sleep_records():
    """Create sample sleep records."""
    records = []
    base_date = datetime(2024, 1, 1, 23, 0)

    for i in range(30):  # 30 days to ensure sufficient history
        start = base_date + timedelta(days=i)
        end = start + timedelta(hours=8)
        record = SleepRecord(
            source_name="test",
            start_date=start,
            end_date=end,
            state=SleepState.ASLEEP,
        )
        records.append(record)

    return records


@pytest.fixture
def sample_activity_records():
    """Create sample activity records."""
    records = []
    base_date = datetime(2024, 1, 1, 8, 0)

    for i in range(30):
        for hour in range(24):  # Full day of hourly data
            record_date = base_date + timedelta(days=i, hours=hour)
            steps = 500 if 8 <= hour <= 20 else 50  # Active during day
            record = ActivityRecord(
                source_name="test",
                start_date=record_date,
                end_date=record_date + timedelta(hours=1),
                activity_type=ActivityType.STEP_COUNT,
                value=float(steps),
                unit="count",
            )
            records.append(record)

    return records


@pytest.fixture
def sample_heart_records():
    """Create sample heart rate records."""
    return []  # Empty for now, as other tests do


def test_pipeline_returns_clinical_feature_set(sample_sleep_records, sample_activity_records, sample_heart_records):
    """Test that aggregation pipeline returns ClinicalFeatureSet instead of DailyFeatures."""
    # Use a range to ensure we get results
    start_date = date(2024, 1, 10)
    end_date = date(2024, 1, 20)

    pipeline = AggregationPipeline()
    result_list = pipeline.aggregate_daily_features(
        sleep_records=sample_sleep_records,
        activity_records=sample_activity_records,
        heart_records=sample_heart_records,
        start_date=start_date,
        end_date=end_date,
    )

    # --- RED assertions (fail until we merge data models) ---
    assert result_list, "Pipeline returned no daily features"
    daily = result_list[0]

    # First assertion: Should return ClinicalFeatureSet, not DailyFeatures
    assert isinstance(
        daily, ClinicalFeatureSet
    ), f"Expected ClinicalFeatureSet, got {type(daily)}"

    # Second assertion: Activity stats must be top-level, not nested
    # Currently they exist on DailyFeatures but we want them flattened on ClinicalFeatureSet
    for field in ("total_steps", "activity_variance", "sedentary_hours"):
        assert hasattr(daily, field), f"{field} should be a direct attribute"
