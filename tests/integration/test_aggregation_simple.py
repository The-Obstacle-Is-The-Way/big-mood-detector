"""Simple test to demonstrate aggregation performance issue."""

import time
from datetime import date, datetime, timedelta

import pytest

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.xfail(strict=False, reason="Known O(n²) performance issue - tracked in issue #38")
def test_aggregation_scales_poorly():
    """Demonstrate that aggregation has O(n*m) complexity."""
    # Create simple test data
    sleep_records = []

    # Generate sleep records for different numbers of days
    for num_days in [30, 60, 90]:
        records = []
        base_date = date.today() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = base_date + timedelta(days=day)
            # Create 3 sleep records per day
            for i in range(3):
                # Convert date to datetime
                start_time = datetime.combine(current_date, datetime.min.time()).replace(hour=22) + timedelta(hours=i*2)
                end_time = start_time + timedelta(hours=1, minutes=30)

                record = SleepRecord(
                    source_name="Apple Watch",
                    start_date=start_time,
                    end_date=end_time,
                    state=SleepState.ASLEEP,
                )
                records.append(record)

        # Time the aggregation
        pipeline = AggregationPipeline()
        start_time = time.time()

        features = pipeline.aggregate_daily_features(
            sleep_records=records,
            activity_records=[],
            heart_records=[],
            start_date=base_date,
            end_date=base_date + timedelta(days=num_days-1),
        )

        duration = time.time() - start_time

        print(f"\n{num_days} days with {len(records)} records:")
        print(f"  Aggregation time: {duration:.2f}s")
        print(f"  Time per day: {duration/num_days:.3f}s")
        print(f"  Features generated: {len(features)}")

        # Store for comparison
        sleep_records.append((num_days, duration))

    # Check if time grows quadratically
    if len(sleep_records) >= 2:
        days1, time1 = sleep_records[0]
        days2, time2 = sleep_records[1]

        # If O(n), time should double when days double
        # If O(n²), time should quadruple when days double
        time_ratio = time2 / time1
        days_ratio = days2 / days1

        print("\nScaling analysis:")
        print(f"  Days increased by {days_ratio:.1f}x")
        print(f"  Time increased by {time_ratio:.1f}x")

        if time_ratio > days_ratio * 1.5:
            print("  ⚠️  This looks like O(n²) complexity!")
        else:
            print("  ✓  This looks like O(n) complexity")


if __name__ == "__main__":
    test_aggregation_scales_poorly()
