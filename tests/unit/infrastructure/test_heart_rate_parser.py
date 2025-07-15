"""
Tests for Apple HealthKit Heart Rate Parser

Test-driven development for heart rate and HRV data extraction.
Following Clean Architecture and SOLID principles.
"""

import pytest

from big_mood_detector.infrastructure.parsers.heart_rate_parser import (
    HeartRateParser,
)


class TestHeartRateParser:
    """Test suite for HeartRateParser - Apple HealthKit heart data extraction."""

    def test_heart_rate_parser_exists(self):
        """Test that HeartRateParser class can be instantiated."""
        # ARRANGE & ACT
        parser = HeartRateParser()

        # ASSERT
        assert parser is not None
        assert isinstance(parser, HeartRateParser)

    def test_parse_accepts_xml_string(self):
        """Test that parse method accepts XML string input."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = "<HealthData></HealthData>"

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert result is not None
        assert isinstance(result, list)

    def test_parse_empty_xml_returns_empty_list(self):
        """Test parsing empty HealthData returns empty list."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData></HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert result == []

    def test_extract_heart_rate_record(self):
        """Test extraction of heart rate records."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    unit="count/min"
                    creationDate="2024-01-01 10:00:00 -0800"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    value="72"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["type"] == "HKQuantityTypeIdentifierHeartRate"
        assert result[0]["sourceName"] == "Apple Watch"
        assert result[0]["unit"] == "count/min"
        assert result[0]["value"] == "72"
        assert result[0]["startDate"] == "2024-01-01 10:00:00 -0800"

    def test_extract_heart_rate_variability_record(self):
        """Test extraction of HRV (SDNN) records."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
                    sourceName="Apple Watch"
                    unit="ms"
                    creationDate="2024-01-01 07:00:00 -0800"
                    startDate="2024-01-01 07:00:00 -0800"
                    endDate="2024-01-01 07:00:00 -0800"
                    value="45.5"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["type"] == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
        assert result[0]["value"] == "45.5"
        assert result[0]["unit"] == "ms"

    def test_extract_resting_heart_rate_record(self):
        """Test extraction of resting heart rate records."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierRestingHeartRate"
                    sourceName="Apple Watch"
                    unit="count/min"
                    creationDate="2024-01-01 08:00:00 -0800"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="58"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["type"] == "HKQuantityTypeIdentifierRestingHeartRate"
        assert result[0]["value"] == "58"

    def test_extract_multiple_heart_metrics(self):
        """Test extraction of different heart metric types."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    unit="count/min"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    value="72"/>
            <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
                    sourceName="Apple Watch"
                    unit="ms"
                    startDate="2024-01-01 07:00:00 -0800"
                    endDate="2024-01-01 07:00:00 -0800"
                    value="45.5"/>
            <Record type="HKQuantityTypeIdentifierRestingHeartRate"
                    sourceName="Apple Watch"
                    unit="count/min"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="58"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 3
        types = [r["type"] for r in result]
        assert "HKQuantityTypeIdentifierHeartRate" in types
        assert "HKQuantityTypeIdentifierHeartRateVariabilitySDNN" in types
        assert "HKQuantityTypeIdentifierRestingHeartRate" in types

    def test_parse_invalid_xml_raises_exception(self):
        """Test that invalid XML raises appropriate exception."""
        # ARRANGE
        parser = HeartRateParser()
        invalid_xml = "This is not valid XML"

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Invalid XML"):
            parser.parse(invalid_xml)

    def test_filters_non_heart_records(self):
        """Test that non-heart records are filtered out."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    value="1000"/>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    unit="count/min"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    value="72"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["type"] == "HKQuantityTypeIdentifierHeartRate"

    def test_parse_to_entities(self):
        """Test parsing to domain entities."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    unit="count/min"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    value="72"/>
        </HealthData>"""

        # ACT
        entities = parser.parse_to_entities(xml_data)

        # ASSERT
        assert len(entities) == 1
        entity = entities[0]
        assert entity.source_name == "Apple Watch"
        assert entity.value == 72.0
        assert entity.unit == "count/min"
        assert entity.is_instantaneous

    def test_supported_heart_types_property(self):
        """Test that parser exposes supported heart metric types."""
        # ARRANGE
        parser = HeartRateParser()

        # ACT
        supported_types = parser.supported_heart_types

        # ASSERT
        assert isinstance(supported_types, list)
        assert len(supported_types) >= 3
        assert "HKQuantityTypeIdentifierHeartRate" in supported_types
        assert "HKQuantityTypeIdentifierHeartRateVariabilitySDNN" in supported_types
        assert "HKQuantityTypeIdentifierRestingHeartRate" in supported_types

    def test_heart_rate_context_extraction(self):
        """Test extraction of heart rate context (motion, workout)."""
        # ARRANGE
        parser = HeartRateParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    sourceName="Apple Watch"
                    unit="count/min"
                    startDate="2024-01-01 10:00:00 -0800"
                    endDate="2024-01-01 10:00:00 -0800"
                    value="120">
                <HeartRateMotionContext>active</HeartRateMotionContext>
            </Record>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["value"] == "120"
        assert result[0].get("motionContext") == "active"

