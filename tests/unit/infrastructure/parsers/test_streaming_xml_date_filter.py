"""Test date filtering in streaming XML parser."""

import tempfile
from pathlib import Path

import pytest

from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)


@pytest.fixture
def sample_xml_with_dates() -> str:
    """Create sample XML with various dates."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" 
            startDate="2025-01-01 23:00:00 -0700" 
            endDate="2025-01-02 07:00:00 -0700" 
            value="HKCategoryValueSleepAnalysisInBed"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" 
            startDate="2025-05-15 22:30:00 -0700" 
            endDate="2025-05-16 06:30:00 -0700" 
            value="HKCategoryValueSleepAnalysisAsleep"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" 
            startDate="2025-06-01 23:15:00 -0700" 
            endDate="2025-06-02 07:45:00 -0700" 
            value="HKCategoryValueSleepAnalysisInBed"/>
    <Record type="HKQuantityTypeIdentifierStepCount" 
            startDate="2025-05-20 10:00:00 -0700" 
            endDate="2025-05-20 10:05:00 -0700" 
            value="250"/>
</HealthData>"""


class TestStreamingXMLDateFilter:
    """Test date filtering functionality."""

    def test_date_filter_includes_in_range_records(self, sample_xml_with_dates: str):
        """Test that records within date range are included."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(sample_xml_with_dates)
            temp_path = Path(f.name)

        try:
            parser = StreamingXMLParser()

            # Parse with date range
            records = list(
                parser.iter_records(
                    temp_path,
                    start_date="2025-05-01",
                    end_date="2025-05-31"
                )
            )

            # Should include May records only
            assert len(records) == 2

            # Check dates
            dates = [r["startDate"] for r in records]
            assert any("2025-05-15" in d for d in dates)
            assert any("2025-05-20" in d for d in dates)
            assert not any("2025-01-01" in d for d in dates)
            assert not any("2025-06-01" in d for d in dates)

        finally:
            temp_path.unlink()

    def test_date_filter_excludes_out_of_range_records(self, sample_xml_with_dates: str):
        """Test that records outside date range are excluded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(sample_xml_with_dates)
            temp_path = Path(f.name)

        try:
            parser = StreamingXMLParser()

            # Parse with narrow date range
            records = list(
                parser.iter_records(
                    temp_path,
                    start_date="2025-03-01",
                    end_date="2025-03-31"
                )
            )

            # Should have no records in March
            assert len(records) == 0

        finally:
            temp_path.unlink()

    def test_date_filter_handles_timezone_correctly(self, sample_xml_with_dates: str):
        """Test that timezone parsing works correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(sample_xml_with_dates)
            temp_path = Path(f.name)

        try:
            parser = StreamingXMLParser()

            # Edge case: record starts May 15 22:30 -0700
            # which is May 16 05:30 UTC
            records = list(
                parser.iter_records(
                    temp_path,
                    start_date="2025-05-15",
                    end_date="2025-05-15"  # Single day
                )
            )

            # Should include the sleep record that starts on May 15
            assert len(records) == 1
            assert "2025-05-15" in records[0]["startDate"]

        finally:
            temp_path.unlink()

    def test_no_date_filter_returns_all_records(self, sample_xml_with_dates: str):
        """Test that no date filter returns all records."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(sample_xml_with_dates)
            temp_path = Path(f.name)

        try:
            parser = StreamingXMLParser()

            # Parse without date filter
            records = list(parser.iter_records(temp_path))

            # Should include all 4 records
            assert len(records) == 4

        finally:
            temp_path.unlink()

    def test_invalid_date_format_skips_record(self):
        """Test that records with invalid dates are skipped."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" 
            startDate="invalid-date-format" 
            endDate="2025-01-02 07:00:00 -0700" 
            value="HKCategoryValueSleepAnalysisInBed"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" 
            startDate="2025-05-15 22:30:00 -0700" 
            endDate="2025-05-16 06:30:00 -0700" 
            value="HKCategoryValueSleepAnalysisAsleep"/>
</HealthData>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            parser = StreamingXMLParser()

            # Parse with date filter
            records = list(
                parser.iter_records(
                    temp_path,
                    start_date="2025-01-01",
                    end_date="2025-12-31"
                )
            )

            # Should only include the valid record
            assert len(records) == 1
            assert "2025-05-15" in records[0]["startDate"]

        finally:
            temp_path.unlink()
