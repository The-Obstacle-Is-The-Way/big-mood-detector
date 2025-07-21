"""
Tests for Streaming XML Adapter

Test-driven development for handling large XML files using iterparse.
"""

import tempfile
import tracemalloc
from pathlib import Path

import pytest

class TestStreamingAdapter:
    """Test suite for StreamingXMLParser - memory-efficient XML parsing."""

    def test_streaming_parser_exists(self):
        """Test that StreamingXMLParser can be instantiated."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

        # ARRANGE & ACT
        parser = StreamingXMLParser()

        # ASSERT
        assert parser is not None
        assert isinstance(parser, StreamingXMLParser)

    def test_parse_small_file_streaming(self):
        """Test parsing a small XML file using streaming."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        records = list(parser.iter_records(temp_path))

        # ASSERT
        assert len(records) == 2
        assert records[0]["type"] == "HKQuantityTypeIdentifierStepCount"
        assert records[1]["type"] == "HKQuantityTypeIdentifierHeartRate"

        # Cleanup
        Path(temp_path).unlink()

    def test_parse_to_entities(self):
        """Test parsing to domain entities."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    sourceName="Apple Watch"
                    value="HKCategoryValueSleepAnalysisAsleepCore"
                    startDate="2024-01-01 23:00:00 -0800"
                    endDate="2024-01-02 07:00:00 -0800"/>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    value="5000"
                    startDate="2024-01-02 08:00:00 -0800"
                    endDate="2024-01-02 12:00:00 -0800"
                    unit="count"/>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    value="65"
                    startDate="2024-01-02 06:00:00 -0800"
                    endDate="2024-01-02 06:01:00 -0800"
                    unit="count/min"/>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        entities = list(parser.parse_file(temp_path))

        # ASSERT
        assert len(entities) == 3
        # Verify entity types
        entity_types = [type(e).__name__ for e in entities]
        assert "SleepRecord" in entity_types
        assert "ActivityRecord" in entity_types
        assert "HeartRateRecord" in entity_types

        # Cleanup
        Path(temp_path).unlink()

    def test_memory_efficient_large_file(self):
        """Test that streaming parser is memory-efficient with large files."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

        # ARRANGE - Create a large XML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<HealthData locale="en_US">\n')

            # Write 10,000 records
            for i in range(10000):
                f.write(
                    f'  <Record type="HKQuantityTypeIdentifierStepCount" '
                    f'value="{i}" '
                    f'startDate="2024-01-01 {i%24:02d}:00:00 -0800" '
                    f'endDate="2024-01-01 {i%24:02d}:01:00 -0800" '
                    f'unit="count"/>\n'
                )

            f.write("</HealthData>")
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT - Process with memory tracking
        tracemalloc.start()
        record_count = 0

        for _record in parser.iter_records(temp_path):
            record_count += 1
            # Check memory usage periodically
            if record_count % 1000 == 0:
                current, peak = tracemalloc.get_traced_memory()
                # Memory should stay low even after many records
                assert peak < 50 * 1024 * 1024  # Less than 50MB

        tracemalloc.stop()

        # ASSERT
        assert record_count == 10000

        # Cleanup
        Path(temp_path).unlink()

    def test_parse_specific_entity_types(self):
        """Test parsing only specific entity types."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    sourceName="Apple Watch"
                    value="HKCategoryValueSleepAnalysisAsleepCore"
                    startDate="2024-01-01 23:00:00 -0800"
                    endDate="2024-01-02 07:00:00 -0800"/>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    value="5000"
                    startDate="2024-01-02 08:00:00 -0800"
                    endDate="2024-01-02 12:00:00 -0800"
                    unit="count"/>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    value="65"
                    startDate="2024-01-02 06:00:00 -0800"
                    endDate="2024-01-02 06:01:00 -0800"
                    unit="count/min"/>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        sleep_entities = list(parser.parse_file(temp_path, entity_type="sleep"))
        activity_entities = list(parser.parse_file(temp_path, entity_type="activity"))
        heart_entities = list(parser.parse_file(temp_path, entity_type="heart"))

        # ASSERT
        assert len(sleep_entities) == 1
        assert len(activity_entities) == 1
        assert len(heart_entities) == 1

        # Cleanup
        Path(temp_path).unlink()

    def test_batch_processing(self):
        """Test processing entities in batches."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">"""

        # Add 25 records
        for i in range(25):
            xml_content += f"""
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    value="{i * 100}"
                    startDate="2024-01-01 {i%24:02d}:00:00 -0800"
                    endDate="2024-01-01 {i%24:02d}:01:00 -0800"
                    unit="count"/>"""

        xml_content += """
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        batches = list(parser.parse_file_in_batches(temp_path, batch_size=10))

        # ASSERT
        assert len(batches) == 3  # 10 + 10 + 5
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

        # Cleanup
        Path(temp_path).unlink()

    def test_handle_missing_file(self):
        """Test handling of missing file."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

        # ARRANGE
        parser = StreamingXMLParser()

        # ACT & ASSERT
        with pytest.raises(ValueError, match="File not found"):
            list(parser.iter_records("nonexistent.xml"))

    def test_heart_rate_motion_context(self):
        """Test extraction of heart rate motion context metadata."""
        from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData locale="en_US">
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    value="120"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    unit="count/min">
                <MetadataEntry key="HKMetadataKeyHeartRateMotionContext"
                               value="HKHeartRateMotionContextActive"/>
            </Record>
        </HealthData>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = f.name

        parser = StreamingXMLParser()

        # ACT
        records = list(parser.iter_records(temp_path))

        # ASSERT
        assert len(records) == 1
        assert records[0]["motionContext"] == "HKHeartRateMotionContextActive"

        # Cleanup
        Path(temp_path).unlink()
