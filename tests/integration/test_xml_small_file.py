"""Test streaming parser with small files first."""

import time
from pathlib import Path

from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)
from tests.integration.test_xml_streaming_performance import XMLDataGenerator


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
        from big_mood_detector.domain.entities.sleep_record import SleepRecord
        from big_mood_detector.domain.entities.activity_record import ActivityRecord
        
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