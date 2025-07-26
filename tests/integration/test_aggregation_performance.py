"""
Test aggregation performance to identify the real bottleneck.

The XML parsing is fast (33MB/s), but the aggregation might be slow.
"""

import time

# Import generator
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)


class XMLDataGenerator:
    """Generate realistic Apple Health XML test data."""

    def __init__(self, base_path: Path = Path("tests/_data/generated")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_export(
        self,
        num_days: int = 30,
        records_per_day: int = 1000,
        size_mb: int | None = None,
    ) -> Path:
        """Create a test Apple Health export XML file."""
        # Create root element
        root = Element("HealthData")
        root.set("locale", "en_US")

        # Calculate records needed for target size
        if size_mb:
            # Approximate: each record is ~500 bytes
            total_records = (size_mb * 1024 * 1024) // 500
            records_per_day = total_records // max(1, num_days)

        # Generate records
        start_date = datetime.now() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)

            # Sleep records (3-4 per day)
            for i in range(3):
                record = SubElement(root, "Record")
                record.set("type", "HKCategoryTypeIdentifierSleepAnalysis")
                record.set("value", "HKCategoryValueSleepAnalysisAsleep")
                record.set("sourceName", "Apple Watch")

                sleep_start = current_date.replace(hour=22) + timedelta(hours=i*2)
                sleep_end = sleep_start + timedelta(hours=1, minutes=30)

                record.set("startDate", sleep_start.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("endDate", sleep_end.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("creationDate", sleep_end.strftime("%Y-%m-%d %H:%M:%S +0000"))

            # Activity records (most of the data)
            for i in range(records_per_day - 3):
                record = SubElement(root, "Record")
                record.set("type", "HKQuantityTypeIdentifierStepCount")
                record.set("value", str(100 + i % 500))
                record.set("unit", "count")
                record.set("sourceName", "iPhone")

                activity_time = current_date + timedelta(hours=i % 24, minutes=(i * 5) % 60)
                record.set("startDate", activity_time.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("endDate", activity_time.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("creationDate", activity_time.strftime("%Y-%m-%d %H:%M:%S +0000"))

        # Write to file
        filename = f"test_export_{size_mb or num_days}mb.xml"
        filepath = self.base_path / filename

        with open(filepath, "wb") as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(tostring(root, encoding="utf-8"))

        return filepath


import pytest


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.xfail(strict=False, reason="Known O(n²) performance issue - tracked in issue #38")
def test_aggregation_bottleneck():
    """Test to identify where the bottleneck is."""
    # Test with different numbers of days
    test_configs = [
        (30, 1000),   # 30 days, 1000 records/day = 30k records
        (90, 1000),   # 90 days, 1000 records/day = 90k records
        (180, 1000),  # 180 days, 1000 records/day = 180k records
        (365, 1000),  # 365 days, 1000 records/day = 365k records
    ]

    generator = XMLDataGenerator()
    parser = StreamingXMLParser()
    pipeline = AggregationPipeline()

    print("\nTesting aggregation performance scaling:")
    print("Days | Records | Parse Time | Aggregate Time | Total | Aggregate/Parse Ratio")
    print("-" * 80)

    for num_days, records_per_day in test_configs:
        # Generate test file
        test_file = generator.create_export(
            num_days=num_days,
            records_per_day=records_per_day
        )

        try:
            # Parse the file
            parse_start = time.time()

            sleep_records = []
            activity_records = []
            heart_records = []

            from big_mood_detector.domain.entities.activity_record import ActivityRecord
            from big_mood_detector.domain.entities.heart_rate_record import (
                HeartRateRecord,
            )
            from big_mood_detector.domain.entities.sleep_record import SleepRecord

            for entity in parser.parse_file(str(test_file)):
                if isinstance(entity, SleepRecord):
                    sleep_records.append(entity)
                elif isinstance(entity, ActivityRecord):
                    activity_records.append(entity)
                elif isinstance(entity, HeartRateRecord):
                    heart_records.append(entity)

            parse_time = time.time() - parse_start
            total_records = len(sleep_records) + len(activity_records) + len(heart_records)

            # Aggregate features
            aggregate_start = time.time()

            end_date = date.today()
            start_date = end_date - timedelta(days=num_days-1)

            pipeline.aggregate_daily_features(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                start_date=start_date,
                end_date=end_date,
            )

            aggregate_time = time.time() - aggregate_start
            total_time = parse_time + aggregate_time

            # Print results
            ratio = aggregate_time / parse_time if parse_time > 0 else 0
            print(f"{num_days:4d} | {total_records:7d} | {parse_time:10.2f}s | {aggregate_time:14.2f}s | {total_time:5.2f}s | {ratio:5.1f}x")

            # Check for O(n*m) behavior
            # If aggregation time grows quadratically, this is the problem

        finally:
            if test_file.exists():
                test_file.unlink()

    print("\nIf aggregate/parse ratio increases with more days, we have O(n*m) complexity!")


def test_aggregation_with_date_filter():
    """Test aggregation with different date ranges to confirm O(n*m) issue."""
    # Generate a large dataset (365 days)
    generator = XMLDataGenerator()
    test_file = generator.create_export(num_days=365, records_per_day=1000)

    try:
        # Parse once
        parser = StreamingXMLParser()
        sleep_records = []
        activity_records = []
        heart_records = []

        from big_mood_detector.domain.entities.activity_record import ActivityRecord
        from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
        from big_mood_detector.domain.entities.sleep_record import SleepRecord

        for entity in parser.parse_file(str(test_file)):
            if isinstance(entity, SleepRecord):
                sleep_records.append(entity)
            elif isinstance(entity, ActivityRecord):
                activity_records.append(entity)
            elif isinstance(entity, HeartRateRecord):
                heart_records.append(entity)

        print(f"\nTotal records parsed: {len(sleep_records) + len(activity_records)}")

        # Test aggregation with different date ranges
        pipeline = AggregationPipeline()
        end_date = date.today()

        print("\nTesting aggregation with different date ranges:")
        print("Days | Aggregate Time | Time per Day")
        print("-" * 40)

        for days_back in [30, 60, 90, 180, 365]:
            start_date = end_date - timedelta(days=days_back-1)

            start_time = time.time()
            pipeline.aggregate_daily_features(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                start_date=start_date,
                end_date=end_date,
            )
            aggregate_time = time.time() - start_time

            time_per_day = aggregate_time / days_back
            print(f"{days_back:4d} | {aggregate_time:14.2f}s | {time_per_day:12.4f}s")

        print("\nIf time per day increases significantly, we have O(n*m) complexity!")

    finally:
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    test_aggregation_bottleneck()
    test_aggregation_with_date_filter()
