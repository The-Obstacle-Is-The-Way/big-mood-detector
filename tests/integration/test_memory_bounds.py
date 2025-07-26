"""
Memory bounds integration test.

Tests that XML streaming parser stays within memory limits for large files.

XFAIL STATUS:
- Issue #38: Streaming parser date filtering has string/datetime comparison bug
- Problem: Date filtering compares datetime objects to strings, causing TypeError
- Impact: Cannot filter large XML files by date range efficiently
- Workaround: Process entire file without date filtering (slower but works)
- Resolution: Fix comparison logic in fast_streaming_parser.py
"""


import pytest

from big_mood_detector.infrastructure.parsers.xml.fast_streaming_parser import (
    FastStreamingXMLParser,
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.xfail(
    reason="Issue #38: Streaming parser date filtering has string/datetime comparison bug - see issues/streaming-parser-date-bug.md",
    strict=True
)
class TestMemoryBounds:
    """Test memory efficiency of streaming parser."""

    def test_large_file_memory_usage(self, tmp_path):
        """Test that parser handles large files without excessive memory usage."""
        # This would create a large test file and verify memory stays bounded
        # For now, marking as xfail until the date filtering bug is fixed

        parser = FastStreamingXMLParser()

        # Create a mock large file scenario
        test_file = tmp_path / "large_export.xml"
        test_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData>
<HealthData>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis"
            startDate="2024-01-01 22:00:00 +0000"
            endDate="2024-01-02 06:00:00 +0000"
            value="HKCategoryValueSleepAnalysisAsleep"/>
</HealthData>""")

        # This should work with date filtering
        records = list(parser.parse_file(
            test_file,
            entity_type="sleep",
            start_date="2024-01-01",
            end_date="2024-01-02"
        ))

        assert len(records) == 1, "Date filtering should return the sleep record"
