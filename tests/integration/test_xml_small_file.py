"""Test streaming parser with small files first."""

import time

# Import the generator directly
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


def test_small_file_parses_correctly():
    """Test that small files parse correctly with streaming parser."""
    # Given: A small test file
    generator = XMLDataGenerator()
    small_file = generator.create_export(num_days=7, records_per_day=100)

    try:
        # When: Parsing the file
        parser = StreamingXMLParser()
        records = list(parser.parse_file(str(small_file)))

        # Then: Should have some records
        assert len(records) > 0, "No records parsed"
        print(f"Parsed {len(records)} records from small file")

        # Check record types
        from big_mood_detector.domain.entities.activity_record import ActivityRecord
        from big_mood_detector.domain.entities.sleep_record import SleepRecord

        sleep_count = sum(1 for r in records if isinstance(r, SleepRecord))
        activity_count = sum(1 for r in records if isinstance(r, ActivityRecord))

        print(f"Sleep records: {sleep_count}")
        print(f"Activity records: {activity_count}")

        assert sleep_count > 0, "No sleep records found"
        assert activity_count > 0, "No activity records found"

    finally:
        if small_file.exists():
            small_file.unlink()


def test_medium_file_performance():
    """Test performance with medium-sized file."""
    # Given: A 50MB test file
    generator = XMLDataGenerator()
    medium_file = generator.create_export(size_mb=50)

    try:
        # When: Parsing the file
        parser = StreamingXMLParser()
        start_time = time.time()

        records = list(parser.parse_file(str(medium_file)))

        duration = time.time() - start_time

        # Then: Should complete quickly
        print(f"Parsed {len(records)} records in {duration:.2f} seconds")
        print(f"Records per second: {len(records) / duration:.0f}")

        # 50MB should parse in under 30 seconds
        assert duration < 30, f"Parsing took {duration:.2f}s, expected < 30s"

    finally:
        if medium_file.exists():
            medium_file.unlink()


if __name__ == "__main__":
    test_small_file_parses_correctly()
    test_medium_file_performance()
