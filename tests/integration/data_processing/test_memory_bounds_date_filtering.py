"""
Integration tests for memory-bounded date filtering.

Tests that date range filtering prevents out-of-memory errors on large files.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import psutil
import pytest

from big_mood_detector.application.services.data_parsing_service import (
    DataParsingService,
)
from big_mood_detector.infrastructure.parsers.xml.fast_streaming_parser import (
    FastStreamingXMLParser,
)
from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)


class TestMemoryBoundsDateFiltering:
    """Test that date filtering keeps memory usage bounded."""

    @pytest.fixture
    def temp_xml_file(self):
        """Create a temporary XML file with many records."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            # Write XML header
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<HealthData locale="en_US">\n')
            
            # Generate records across multiple years
            base_date = datetime(2020, 1, 1)
            for i in range(100000):  # 100k records to simulate large file
                # Spread across 4 years (2020-2024)
                date = base_date + timedelta(days=i // 69)  # ~69 records per day
                f.write(f'  <Record type="HKCategoryTypeIdentifierSleepAnalysis" '
                       f'sourceName="Apple Watch" '
                       f'sourceVersion="10.0" '
                       f'device="&lt;&lt;HKDevice&gt;&gt;" '
                       f'creationDate="{date.isoformat()}" '
                       f'startDate="{date.isoformat()}" '
                       f'endDate="{(date + timedelta(hours=8)).isoformat()}" '
                       f'value="HKCategoryValueSleepAnalysisInBed"/>\n')
            
            f.write('</HealthData>')
            
            file_path = Path(f.name)
        
        yield file_path
        
        # Cleanup
        file_path.unlink()

    def test_memory_usage_with_date_filtering(self, temp_xml_file):
        """Test that date filtering reduces memory usage."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create parser
        parser = DataParsingService()
        
        # Parse only last 30 days (from our test data range)
        end_date = datetime(2023, 6, 1).date()
        start_date = end_date - timedelta(days=30)
        
        # Parse with date filter
        result = parser.parse_xml_export(
            temp_xml_file,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check memory usage after parsing
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Verify we got only 30 days of data (approximately)
        assert len(result.sleep_records) < 100  # Should be ~30 records, not 100k
        
        # Memory increase should be minimal (< 100MB for filtered data)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f} MB"
        
        print(f"\nMemory usage with date filtering:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        print(f"  Records parsed: {len(result.sleep_records)}")

    def test_streaming_parser_date_filtering(self, temp_xml_file):
        """Test streaming parser with date filtering."""
        # First, let's check what dates are in our test file
        print(f"\nTest file: {temp_xml_file}")
        
        # Read first few lines to see date format
        with open(temp_xml_file, 'r') as f:
            for i, line in enumerate(f):
                if i > 10 and 'Record' in line:
                    print(f"Sample record: {line.strip()}")
                    break
        
        # Use fast streaming parser if available
        try:
            parser = FastStreamingXMLParser()
        except ImportError:
            parser = StreamingXMLParser()
        
        # Count all records first to verify file
        total_count = 0
        for entity in parser.parse_file(temp_xml_file, entity_type="sleep"):
            total_count += 1
            if total_count == 1:
                print(f"First entity: {entity}")
        print(f"Total records in file: {total_count}")
        
        # Now test with date filtering
        # Our test data spans ~1450 days starting from 2020-01-01
        # 100000 records / 69 per day = ~1450 days
        # So records should go up to around May 2023
        end_date = "2023-05-31"
        start_date = "2023-05-01"
        
        record_count = 0
        try:
            for entity in parser.parse_file(
                temp_xml_file,
                entity_type="sleep",
                start_date=start_date,
                end_date=end_date
            ):
                record_count += 1
        except Exception as e:
            print(f"Error during parsing: {e}")
            raise
        
        # Should get approximately 31 days of records
        # With ~69 records per day, should get ~2100 records for May 2023
        assert record_count > 1000, f"Expected at least 1000 records, got {record_count}"
        assert record_count < 3000, f"Expected less than 3000 records, got {record_count}"

    def test_large_file_warning_threshold(self):
        """Test that large file size triggers appropriate warnings."""
        from big_mood_detector.interfaces.cli.commands import validate_input_path
        
        # Create a mock large file
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            # Write minimal content
            f.write(b"<HealthData/>")
            file_path = Path(f.name)
        
        try:
            # Mock file size to appear large
            with patch.object(Path, 'stat') as mock_stat:
                mock_stat.return_value.st_size = 600 * 1024 * 1024  # 600MB
                
                # Capture output
                import io
                from contextlib import redirect_stdout
                
                f = io.StringIO()
                with redirect_stdout(f):
                    validate_input_path(file_path)
                
                output = f.getvalue()
                
                # Should show warning about large file
                assert "Very large file" in output
                assert "--days-back" in output or "--date-range" in output
        
        finally:
            file_path.unlink()

    @pytest.mark.skipif(
        not os.path.exists("data/large_export.xml"),
        reason="Large test file not available"
    )
    def test_real_large_file_memory_bounds(self):
        """Test with real large export file if available."""
        file_path = Path("data/large_export.xml")
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        print(f"\nTesting with real file: {file_size_mb:.1f} MB")
        
        # Get baseline memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Parse with date filtering (last 90 days)
        parser = DataParsingService()
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        
        result = parser.parse_xml_export(
            file_path,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check memory didn't explode
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Records parsed: {len(result.sleep_records)} sleep records")
        
        # Memory should stay under control even for large files
        # With date filtering, should use < 500MB even for multi-GB files
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"

    def test_progress_callback_with_date_filtering(self, temp_xml_file):
        """Test that progress callbacks work with date filtering."""
        parser = DataParsingService()
        
        # Track progress calls
        progress_calls = []
        
        def progress_callback(message: str, progress: float):
            progress_calls.append((message, progress))
        
        # Parse with date filter and progress
        end_date = datetime(2023, 12, 31).date()
        start_date = end_date - timedelta(days=30)
        
        result = parser.parse_xml_export(
            temp_xml_file,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback
        )
        
        # Should have progress calls
        assert len(progress_calls) > 0
        assert any("Parsing" in msg for msg, _ in progress_calls)
        assert any(progress == 1.0 for _, progress in progress_calls)
        
        # Should have filtered results
        assert len(result.sleep_records) < 100