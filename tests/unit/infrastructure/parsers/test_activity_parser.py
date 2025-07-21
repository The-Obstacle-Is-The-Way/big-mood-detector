"""
Tests for Apple HealthKit Activity Data Parser

Test-driven development for activity data extraction.
Following Clean Architecture and Uncle Bob's principles.
"""

import pytest

from big_mood_detector.infrastructure.parsers.xml import ActivityParser


class TestActivityParser:
    """Test suite for ActivityParser - Apple HealthKit activity data extraction."""

    def test_activity_parser_exists(self):
        """Test that ActivityParser class can be instantiated."""
        # ARRANGE & ACT
        parser = ActivityParser()

        # ASSERT
        assert parser is not None
        assert isinstance(parser, ActivityParser)

    def test_parse_accepts_xml_string(self):
        """Test that parse method accepts XML string input."""
        # ARRANGE
        parser = ActivityParser()
        xml_data = "<HealthData></HealthData>"

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert result is not None
        assert isinstance(result, list)

    def test_parse_empty_xml_returns_empty_list(self):
        """Test parsing empty HealthData returns empty list."""
        # ARRANGE
        parser = ActivityParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData></HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert result == []

    def test_extract_step_count_record(self):
        """Test extraction of step count records."""
        # ARRANGE
        parser = ActivityParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    unit="count"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="10000"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["type"] == "HKQuantityTypeIdentifierStepCount"
        assert result[0]["sourceName"] == "iPhone"
        assert result[0]["unit"] == "count"
        assert result[0]["value"] == "10000"
        assert result[0]["startDate"] == "2024-01-01 00:00:00 -0800"
        assert result[0]["endDate"] == "2024-01-01 23:59:59 -0800"

    def test_extract_multiple_activity_types(self):
        """Test extraction of different activity types."""
        # ARRANGE
        parser = ActivityParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    unit="count"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="10000"/>
            <Record type="HKQuantityTypeIdentifierActiveEnergyBurned"
                    sourceName="Apple Watch"
                    unit="Cal"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="450"/>
            <Record type="HKQuantityTypeIdentifierDistanceWalkingRunning"
                    sourceName="iPhone"
                    unit="km"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="7.5"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 3
        types = [r["type"] for r in result]
        assert "HKQuantityTypeIdentifierStepCount" in types
        assert "HKQuantityTypeIdentifierActiveEnergyBurned" in types
        assert "HKQuantityTypeIdentifierDistanceWalkingRunning" in types

    def test_parse_invalid_xml_raises_exception(self):
        """Test that invalid XML raises appropriate exception."""
        # ARRANGE
        parser = ActivityParser()
        invalid_xml = "This is not valid XML"

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Invalid XML"):
            parser.parse(invalid_xml)

    def test_filters_non_activity_records(self):
        """Test that non-activity records are filtered out."""
        # ARRANGE
        parser = ActivityParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    value="HKCategoryValueSleepAnalysisAsleep"/>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    unit="count"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="10000"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["type"] == "HKQuantityTypeIdentifierStepCount"

    def test_parse_to_entities(self):
        """Test parsing to domain entities."""
        # ARRANGE
        parser = ActivityParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    unit="count"
                    startDate="2024-01-01 00:00:00 -0800"
                    endDate="2024-01-01 23:59:59 -0800"
                    value="10000"/>
        </HealthData>"""

        # ACT
        entities = parser.parse_to_entities(xml_data)

        # ASSERT
        assert len(entities) == 1
        entity = entities[0]
        assert entity.source_name == "iPhone"
        assert entity.value == 10000.0
        assert entity.unit == "count"

    def test_supported_activity_types_property(self):
        """Test that parser exposes supported activity types."""
        # ARRANGE
        parser = ActivityParser()

        # ACT
        supported_types = parser.supported_activity_types

        # ASSERT
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert "HKQuantityTypeIdentifierStepCount" in supported_types
        assert "HKQuantityTypeIdentifierActiveEnergyBurned" in supported_types

    def test_instantaneous_vs_duration_records(self):
        """Test handling of instantaneous vs duration-based records."""
        # ARRANGE
        parser = ActivityParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    sourceName="iPhone"
                    unit="count"
                    startDate="2024-01-01 12:00:00 -0800"
                    endDate="2024-01-01 12:00:00 -0800"
                    value="100"/>
            <Record type="HKQuantityTypeIdentifierActiveEnergyBurned"
                    sourceName="Apple Watch"
                    unit="Cal"
                    startDate="2024-01-01 12:00:00 -0800"
                    endDate="2024-01-01 13:00:00 -0800"
                    value="50"/>
        </HealthData>"""

        # ACT
        entities = parser.parse_to_entities(xml_data)

        # ASSERT
        assert len(entities) == 2
        assert entities[0].is_instantaneous  # Same start/end time
        assert not entities[1].is_instantaneous  # Different times
