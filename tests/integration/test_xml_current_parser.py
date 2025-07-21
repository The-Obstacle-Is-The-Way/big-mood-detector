"""Test current parser performance with larger files."""

import time

# Import directly
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

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


def test_100mb_file():
    """Test 100MB file performance."""
    generator = XMLDataGenerator()
    large_file = generator.create_export(size_mb=100)

    try:
        parser = StreamingXMLParser()
        start_time = time.time()

        records = list(parser.parse_file(str(large_file)))

        duration = time.time() - start_time

        print(f"100MB file: {len(records)} records in {duration:.2f}s")
        print(f"Records per second: {len(records) / duration:.0f}")
        print(f"MB per second: {100 / duration:.1f}")

        # Extrapolate to 500MB
        estimated_500mb = duration * 5
        print(f"\nEstimated time for 500MB: {estimated_500mb:.1f}s ({estimated_500mb/60:.1f} minutes)")

        assert duration < 60, f"100MB took {duration:.2f}s, expected < 60s"

    finally:
        if large_file.exists():
            large_file.unlink()


def test_200mb_file():
    """Test 200MB file performance."""
    generator = XMLDataGenerator()
    large_file = generator.create_export(size_mb=200)

    try:
        parser = StreamingXMLParser()
        start_time = time.time()

        # Don't collect all records to save memory
        record_count = 0
        for _record in parser.parse_file(str(large_file)):
            record_count += 1

        duration = time.time() - start_time

        print(f"\n200MB file: {record_count} records in {duration:.2f}s")
        print(f"Records per second: {record_count / duration:.0f}")
        print(f"MB per second: {200 / duration:.1f}")

        # Extrapolate to 500MB
        estimated_500mb = duration * 2.5
        print(f"\nEstimated time for 500MB: {estimated_500mb:.1f}s ({estimated_500mb/60:.1f} minutes)")

        assert duration < 120, f"200MB took {duration:.2f}s, expected < 120s"

    finally:
        if large_file.exists():
            large_file.unlink()


if __name__ == "__main__":
    test_100mb_file()
    test_200mb_file()
