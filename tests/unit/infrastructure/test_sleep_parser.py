"""
Tests for Apple HealthKit Sleep Data Parser

Test-driven development for clinical-grade sleep data extraction.
Following Clean Architecture principles.
"""

import pytest

class TestSleepParser:
    """Test suite for SleepParser - Apple HealthKit sleep data extraction."""

    def test_sleep_parser_exists(self):
        """Test that SleepParser class can be instantiated."""
        from big_mood_detector.infrastructure.parsers.xml import SleepParser

        # ARRANGE & ACT
        parser = SleepParser()

        # ASSERT
        assert parser is not None
        assert isinstance(parser, SleepParser)

    def test_parse_accepts_xml_string(self):
        """Test that parse method accepts XML string input."""
        from big_mood_detector.infrastructure.parsers.xml import SleepParser

        # ARRANGE
        parser = SleepParser()
        xml_data = "<HealthData></HealthData>"

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert result is not None
        assert isinstance(result, list)

    def test_parse_empty_xml_returns_empty_list(self):
        """Test parsing empty HealthData returns empty list."""
        from big_mood_detector.infrastructure.parsers.xml import SleepParser

        # ARRANGE
        parser = SleepParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData></HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert result == []

    def test_extract_single_sleep_record(self, sample_sleep_xml):
        """Test extraction of sleep records from real XML structure."""
        from big_mood_detector.infrastructure.parsers.xml import SleepParser

        # ARRANGE
        parser = SleepParser()
        single_record_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    sourceName="Apple Watch"
                    startDate="2024-01-01 23:30:00 -0800"
                    endDate="2024-01-02 07:30:00 -0800"
                    value="HKCategoryValueSleepAnalysisAsleep"/>
        </HealthData>"""

        # ACT
        result = parser.parse(single_record_xml)

        # ASSERT
        assert len(result) == 1
        assert result[0]["sourceName"] == "Apple Watch"
        assert result[0]["startDate"] == "2024-01-01 23:30:00 -0800"
        assert result[0]["endDate"] == "2024-01-02 07:30:00 -0800"
        assert result[0]["value"] == "HKCategoryValueSleepAnalysisAsleep"

    def test_handles_inbed_vs_asleep_states(self):
        """Test distinction between InBed and Asleep states."""
        from big_mood_detector.infrastructure.parsers.xml import SleepParser

        # ARRANGE
        parser = SleepParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    sourceName="Apple Watch"
                    startDate="2024-01-01 23:00:00 -0800"
                    endDate="2024-01-01 23:30:00 -0800"
                    value="HKCategoryValueSleepAnalysisInBed"/>
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    sourceName="Apple Watch"
                    startDate="2024-01-01 23:30:00 -0800"
                    endDate="2024-01-02 07:00:00 -0800"
                    value="HKCategoryValueSleepAnalysisAsleep"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 2
        assert result[0]["value"] == "HKCategoryValueSleepAnalysisInBed"
        assert result[1]["value"] == "HKCategoryValueSleepAnalysisAsleep"

    def test_parse_invalid_xml_raises_exception(self):
        """Test that invalid XML raises appropriate exception."""
        from big_mood_detector.infrastructure.parsers.xml import SleepParser

        # ARRANGE
        parser = SleepParser()
        invalid_xml = "This is not valid XML"

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Invalid XML"):
            parser.parse(invalid_xml)

    def test_parse_non_sleep_records_are_filtered(self):
        """Test that non-sleep records are filtered out."""
        from big_mood_detector.infrastructure.parsers.xml import SleepParser

        # ARRANGE
        parser = SleepParser()
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    value="1000"
                    startDate="2024-01-01 10:00:00 -0800"/>
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    sourceName="Apple Watch"
                    startDate="2024-01-01 23:30:00 -0800"
                    endDate="2024-01-02 07:30:00 -0800"
                    value="HKCategoryValueSleepAnalysisAsleep"/>
        </HealthData>"""

        # ACT
        result = parser.parse(xml_data)

        # ASSERT
        assert len(result) == 1
        assert result[0]["value"] == "HKCategoryValueSleepAnalysisAsleep"
