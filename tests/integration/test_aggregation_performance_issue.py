"""
Test that demonstrates the O(n*m) performance issue in aggregation.

This is the test that should FAIL with the current implementation
and PASS after we implement pre-indexing.
"""

import time
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


@pytest.mark.integration
@pytest.mark.slow
def test_aggregation_should_be_linear_not_quadratic():
    """
    Test that aggregation time grows linearly with data size.

    Current implementation has O(n*m) complexity where:
    - n = number of days
    - m = number of records

    This test should FAIL with current implementation.
    """
    # Test configurations: (days, records_per_day)
    test_cases = [
        (30, 100),   # 3,000 records
        (60, 100),   # 6,000 records
        (120, 100),  # 12,000 records
    ]

    timings = []

    for num_days, records_per_day in test_cases:
        # Generate test data
        sleep_records = []
        activity_records = []

        base_date = date.today() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = base_date + timedelta(days=day)

            # 3 sleep records per day
            for i in range(3):
                start_time = datetime.combine(current_date, datetime.min.time()).replace(hour=22) + timedelta(hours=i*2)
                end_time = start_time + timedelta(hours=1, minutes=30)

                sleep_record = SleepRecord(
                    source_name="Apple Watch",
                    start_date=start_time,
                    end_date=end_time,
                    state=SleepState.ASLEEP,
                )
                sleep_records.append(sleep_record)

            # Many activity records per day (this causes O(n*m))
            for i in range(records_per_day):
                activity_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=i % 24, minutes=(i * 5) % 60)

                activity_record = ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_time,
                    end_date=activity_time,
                    activity_type=ActivityType.STEP_COUNT,
                    value=100.0 + i,
                    unit="count",
                )
                activity_records.append(activity_record)

        # Time the aggregation with DLMO disabled for performance testing
        from big_mood_detector.application.services.aggregation_pipeline import (
            AggregationConfig,
        )

        config = AggregationConfig(
            enable_dlmo_calculation=False,  # Disable expensive DLMO
            enable_circadian_analysis=False,  # Disable expensive circadian
        )
        pipeline = AggregationPipeline(config=config)

        start_time = time.time()
        features = pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            start_date=base_date,
            end_date=base_date + timedelta(days=num_days-1),
        )
        duration = time.time() - start_time

        total_records = len(sleep_records) + len(activity_records)
        timings.append((num_days, total_records, duration))

        print(f"\n{num_days} days, {total_records:,} records:")
        print(f"  Aggregation time: {duration:.2f}s")
        print(f"  Time per day: {duration/num_days:.3f}s")
        print(f"  Features generated: {len(features)}")

    # Analyze scaling
    if len(timings) >= 2:
        # Compare first and last
        days1, records1, time1 = timings[0]
        days2, records2, time2 = timings[-1]

        time_ratio = time2 / time1
        days_ratio = days2 / days1
        records_ratio = records2 / records1

        print("\nScaling analysis:")
        print(f"  Days increased: {days_ratio:.1f}x")
        print(f"  Records increased: {records_ratio:.1f}x")
        print(f"  Time increased: {time_ratio:.1f}x")

        # For O(n) we expect time_ratio ≈ records_ratio
        # For O(n*m) we expect time_ratio ≈ days_ratio * records_ratio
        expected_quadratic = days_ratio * records_ratio

        print("\nExpected time increase:")
        print(f"  If O(n): {records_ratio:.1f}x")
        print(f"  If O(n*m): {expected_quadratic:.1f}x")
        print(f"  Actual: {time_ratio:.1f}x")

        # This assertion should FAIL with current implementation
        # We allow 50% tolerance for timing variations
        assert time_ratio < records_ratio * 1.5, (
            f"Aggregation appears to be O(n*m)! "
            f"Time increased {time_ratio:.1f}x but records only increased {records_ratio:.1f}x"
        )


@pytest.mark.integration
@pytest.mark.slow
def test_aggregation_performance_target():
    """Test that 365 days of data can be aggregated in reasonable time."""
    # Generate 1 year of realistic data
    num_days = 365
    records_per_day = 1000  # Realistic for active Apple Health user

    sleep_records = []
    activity_records = []

    base_date = date.today() - timedelta(days=num_days)

    print(f"\nGenerating {num_days} days of test data...")
    for day in range(num_days):
        current_date = base_date + timedelta(days=day)

        # 3 sleep records per day
        for i in range(3):
            start_time = datetime.combine(current_date, datetime.min.time()).replace(hour=22) + timedelta(hours=i*2)
            end_time = start_time + timedelta(hours=1, minutes=30)

            sleep_record = SleepRecord(
                source_name="Apple Watch",
                start_date=start_time,
                end_date=end_time,
                state=SleepState.ASLEEP,
            )
            sleep_records.append(sleep_record)

        # Activity records throughout the day
        for i in range(records_per_day):
            activity_time = datetime.combine(current_date, datetime.min.time()) + timedelta(
                hours=i % 24,
                minutes=(i * 5) % 60
            )

            activity_record = ActivityRecord(
                source_name="iPhone",
                start_date=activity_time,
                end_date=activity_time,
                activity_type=ActivityType.STEP_COUNT,
                value=100.0 + i,
                unit="count",
            )
            activity_records.append(activity_record)

    total_records = len(sleep_records) + len(activity_records)
    print(f"Generated {total_records:,} records")

    # Time the aggregation with DLMO disabled
    from big_mood_detector.application.services.aggregation_pipeline import (
        AggregationConfig,
    )

    config = AggregationConfig(
        enable_dlmo_calculation=False,  # Disable expensive DLMO
        enable_circadian_analysis=False,  # Disable expensive circadian
    )
    pipeline = AggregationPipeline(config=config)

    print("\nRunning aggregation...")
    start_time = time.time()

    features = pipeline.aggregate_daily_features(
        sleep_records=sleep_records,
        activity_records=activity_records,
        heart_records=[],
        start_date=base_date,
        end_date=base_date + timedelta(days=num_days-1),
    )

    duration = time.time() - start_time

    print("\nResults:")
    print(f"  Total records: {total_records:,}")
    print(f"  Days processed: {num_days}")
    print(f"  Features generated: {len(features)}")
    print(f"  Aggregation time: {duration:.2f}s")
    print(f"  Time per day: {duration/num_days:.3f}s")
    print(f"  Records per second: {total_records/duration:,.0f}")

    # Target: Process 1 year in under 60 seconds
    assert duration < 60, (
        f"Aggregation took {duration:.1f}s for 1 year of data, "
        f"target is < 60s"
    )
