"""
Tests for Streaming XML Parser

Test-driven development for handling large XML files using iterparse.
Following Clean Architecture and streaming patterns.
"""

import tempfile
from pathlib import Path

import pytest

from big_mood_detector.infrastructure.parsers.xml.streaming_parser import (
    StreamingXMLParser,
)


class TestStreamingXMLParser:
    """Test suite for StreamingXMLParser - handles large XML files efficiently."""

    def test_streaming_parser_exists(self):
        """Test that StreamingXMLParser class can be instantiated."""
        # ARRANGE & ACT
        parser = StreamingXMLParser()

        # ASSERT
        assert parser is not None
        assert isinstance(parser, StreamingXMLParser)

    def test_parse_small_file_streaming(self):
        """Test parsing a small XML file using streaming."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    value="1000"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 11:00:00 -0800"
                    unit="count"/>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    value="72"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    unit="count/min"/>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        records = list(parser.parse_file(temp_path))

        # ASSERT
        assert len(records) == 2
        assert records[0]['type'] == 'HKQuantityTypeIdentifierStepCount'
        assert records[1]['type'] == 'HKQuantityTypeIdentifierHeartRate'

        # Cleanup
        Path(temp_path).unlink()

    def test_parse_with_record_filter(self):
        """Test parsing with a filter to only get specific record types."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKQuantityTypeIdentifierStepCount" value="1000"/>
            <Record type="HKQuantityTypeIdentifierHeartRate" value="72"/>
            <Record type="HKQuantityTypeIdentifierStepCount" value="2000"/>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        step_records = list(
            parser.parse_file(
                temp_path,
                record_types=['HKQuantityTypeIdentifierStepCount']
            )
        )

        # ASSERT
        assert len(step_records) == 2
        assert all(r['type'] == 'HKQuantityTypeIdentifierStepCount' for r in step_records)

        # Cleanup
        Path(temp_path).unlink()

    def test_memory_efficient_large_file(self):
        """Test that streaming parser handles large files memory-efficiently."""
        # ARRANGE - Create a large XML file with many records
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<HealthData locale="en_US">\n')
            
            # Write 10,000 records
            for i in range(10000):
                f.write(f'  <Record type="HKQuantityTypeIdentifierStepCount" '
                       f'value="{i}" '
                       f'startDate="2024-01-01 {i%24:02d}:00:00 -0800" '
                       f'endDate="2024-01-01 {i%24:02d}:01:00 -0800"/>\n')
            
            f.write('</HealthData>')
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT - Process records in batches
        batch_size = 100
        total_records = 0
        
        for batch in parser.parse_file_in_batches(temp_path, batch_size=batch_size):
            assert len(batch) <= batch_size
            total_records += len(batch)

        # ASSERT
        assert total_records == 10000

        # Cleanup
        Path(temp_path).unlink()

    def test_parse_with_progress_callback(self):
        """Test parsing with progress callback for large files."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKQuantityTypeIdentifierStepCount" value="1000"/>
            <Record type="HKQuantityTypeIdentifierStepCount" value="2000"/>
            <Record type="HKQuantityTypeIdentifierStepCount" value="3000"/>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()
        progress_calls = []

        def progress_callback(records_processed: int, bytes_processed: int):
            progress_calls.append((records_processed, bytes_processed))

        # ACT
        records = list(parser.parse_file(temp_path, progress_callback=progress_callback))

        # ASSERT
        assert len(records) == 3
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 3  # Final count should be 3 records

        # Cleanup
        Path(temp_path).unlink()

    def test_parse_specific_parsers(self):
        """Test using streaming with specific parser types."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    value="HKCategoryValueSleepAnalysisAsleepCore"
                    startDate="2024-01-01 23:00:00 -0800"
                    endDate="2024-01-02 07:00:00 -0800"/>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    value="5000"
                    startDate="2024-01-02 08:00:00 -0800"
                    endDate="2024-01-02 12:00:00 -0800"/>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    value="65"
                    startDate="2024-01-02 06:00:00 -0800"
                    endDate="2024-01-02 06:01:00 -0800"/>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        sleep_records = list(parser.parse_sleep_records(temp_path))
        activity_records = list(parser.parse_activity_records(temp_path))
        heart_records = list(parser.parse_heart_records(temp_path))

        # ASSERT
        assert len(sleep_records) == 1
        assert len(activity_records) == 1
        assert len(heart_records) == 1

        # Cleanup
        Path(temp_path).unlink()

    def test_handle_malformed_xml(self):
        """Test handling of malformed XML gracefully."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKQuantityTypeIdentifierStepCount" value="1000"
            <!-- Missing closing tag -->
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT & ASSERT
        with pytest.raises(ValueError, match="XML parsing error"):
            list(parser.parse_file(temp_path))

        # Cleanup
        Path(temp_path).unlink()

    def test_parse_to_entities_streaming(self):
        """Test converting streaming records to domain entities."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    value="72"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    unit="count/min"/>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        entities = list(parser.parse_heart_entities(temp_path))

        # ASSERT
        assert len(entities) == 1
        entity = entities[0]
        assert entity.source_name == "Apple Watch"
        assert entity.value == 72.0

        # Cleanup
        Path(temp_path).unlink()