"""
Test Data Parsing Service

TDD approach for extracting data parsing responsibilities from MoodPredictionPipeline.
The DataParsingService will handle all file I/O and parsing operations.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestDataParsingService:
    """Test the data parsing service extraction."""

    @pytest.fixture
    def parsing_service(self):
        """Create DataParsingService instance."""
        from big_mood_detector.application.services.data_parsing_service import (
            DataParsingService,
        )

        return DataParsingService()

    @pytest.fixture
    def sample_sleep_records(self):
        """Create sample sleep records."""
        records = []
        base_date = datetime(2024, 1, 1, 23, 0)

        for i in range(7):
            start = base_date + timedelta(days=i)
            end = start + timedelta(hours=8)
            record = SleepRecord(
                source_name="test",
                start_date=start,
                end_date=end,
                state=SleepState.ASLEEP,
            )
            records.append(record)

        return records

    @pytest.fixture
    def sample_activity_records(self):
        """Create sample activity records."""
        records = []
        base_date = datetime(2024, 1, 1, 8, 0)

        for i in range(7):
            record_date = base_date + timedelta(days=i)
            record = ActivityRecord(
                source_name="test",
                start_date=record_date,
                end_date=record_date + timedelta(hours=1),
                activity_type=ActivityType.STEP_COUNT,
                value=8000.0 + (i * 500),
                unit="count",
            )
            records.append(record)

        return records

    def test_parse_health_data_from_path(self, parsing_service):
        """Test parsing health data from file path."""
        # Test with both XML and JSON paths
        Path("export.xml")  # XML path
        Path("health_export")  # JSON path

        # Should determine parser type from path
        assert hasattr(parsing_service, "parse_health_data")

    def test_parse_xml_export(self, parsing_service):
        """Test parsing XML export file."""
        xml_path = Path("test_export.xml")

        # Mock the XML parser
        with patch.object(parsing_service, "_xml_parser") as mock_parser:
            mock_parser.parse_file.return_value = []

            result = parsing_service.parse_xml_export(xml_path)

            # Result should be ParsedHealthData object
            assert hasattr(result, "sleep_records")
            assert hasattr(result, "activity_records")
            assert hasattr(result, "heart_rate_records")
            assert isinstance(result.sleep_records, list)
            assert isinstance(result.activity_records, list)
            assert isinstance(result.heart_rate_records, list)

    def test_parse_json_export(self, parsing_service):
        """Test parsing JSON export directory."""
        json_dir = Path("health_export")

        # Mock the JSON parsers and glob
        with patch.object(parsing_service, "_sleep_parser") as mock_sleep_parser:
            with patch.object(
                parsing_service, "_activity_parser"
            ) as mock_activity_parser:
                with patch.object(Path, "glob") as mock_glob:
                    # Mock finding JSON files
                    mock_glob.side_effect = [
                        [Path("Sleep.json")],  # Sleep files for *[Ss]leep*.json
                        [Path("Sleep.json")],  # Sleep files for [Ss]leep*.json
                        [Path("Step_Count.json")],  # Activity files for *[Ss]tep*.json
                        [Path("Step_Count.json")],  # Activity files for [Ss]tep*.json
                    ]
                    mock_sleep_parser.parse_file.return_value = []
                    mock_activity_parser.parse_file.return_value = []

                    result = parsing_service.parse_json_export(json_dir)

                    # Result should be ParsedHealthData object
                    assert hasattr(result, "sleep_records")
                    assert hasattr(result, "activity_records")
                    assert hasattr(result, "heart_rate_records")
                    assert isinstance(result.sleep_records, list)
                    assert isinstance(result.activity_records, list)

    def test_filter_records_by_date_range(self, parsing_service, sample_sleep_records):
        """Test filtering records by date range."""
        start_date = date(2024, 1, 3)
        end_date = date(2024, 1, 5)

        filtered = parsing_service.filter_records_by_date_range(
            records=sample_sleep_records,
            start_date=start_date,
            end_date=end_date,
            date_extractor=lambda r: r.start_date.date(),
        )

        assert len(filtered) == 3  # Days 3, 4, 5
        assert all(start_date <= r.start_date.date() <= end_date for r in filtered)

    def test_validate_parsed_data(self, parsing_service, sample_sleep_records):
        """Test validation of parsed health data."""
        data = {
            "sleep_records": sample_sleep_records,
            "activity_records": [],
            "heart_rate_records": [],
        }

        validation_result = parsing_service.validate_parsed_data(data)

        assert hasattr(validation_result, "is_valid")
        assert hasattr(validation_result, "sleep_record_count")
        assert hasattr(validation_result, "activity_record_count")
        assert hasattr(validation_result, "heart_record_count")
        assert hasattr(validation_result, "date_range")
        assert hasattr(validation_result, "warnings")

    def test_get_data_summary(
        self, parsing_service, sample_sleep_records, sample_activity_records
    ):
        """Test getting summary of parsed data."""
        data = {
            "sleep_records": sample_sleep_records,
            "activity_records": sample_activity_records,
            "heart_rate_records": [],
        }

        summary = parsing_service.get_data_summary(data)

        assert "total_records" in summary
        assert "sleep_days" in summary
        assert "activity_days" in summary
        assert "date_range" in summary
        assert "data_density" in summary

    def test_parser_selection_strategy(self, parsing_service):
        """Test automatic parser selection based on file type."""
        # XML file
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.suffix", ".xml"):
                xml_parser = parsing_service.get_parser_for_path(Path("export.xml"))
                assert xml_parser is not None

        # JSON directory
        with patch("pathlib.Path.is_file", return_value=False):
            with patch("pathlib.Path.is_dir", return_value=True):
                json_parser = parsing_service.get_parser_for_path(
                    Path("health_export/")
                )
                assert json_parser is not None

        # Invalid path
        with patch("pathlib.Path.is_file", return_value=False):
            with patch("pathlib.Path.is_dir", return_value=False):
                with pytest.raises(ValueError):
                    parsing_service.get_parser_for_path(Path("invalid.txt"))

    def test_parse_with_progress_callback(self, parsing_service):
        """Test parsing with progress reporting."""
        progress_updates = []

        def progress_callback(stage: str, percent: float):
            progress_updates.append((stage, percent))

        xml_path = Path("test_export.xml")

        with patch.object(parsing_service, "_xml_parser") as mock_parser:
            mock_parser.parse_file.return_value = []

            with patch("pathlib.Path.is_file", return_value=True):
                with patch(
                    "pathlib.Path.suffix",
                    new_callable=lambda: property(lambda self: ".xml"),
                ):
                    parsing_service.parse_health_data(
                        file_path=xml_path, progress_callback=progress_callback
                    )

            assert len(progress_updates) > 0
            # Should have parsing stages
            stages = [update[0] for update in progress_updates]
            assert any("parsing" in stage.lower() for stage in stages)

    def test_handle_parsing_errors(self, parsing_service):
        """Test error handling during parsing."""
        xml_path = Path("corrupt_export.xml")

        with patch.object(parsing_service, "_xml_parser") as mock_parser:
            mock_parser.parse_file.side_effect = Exception("Parse error")

            result = parsing_service.parse_health_data(
                file_path=xml_path, continue_on_error=True
            )

            assert result is not None
            assert "errors" in result
            assert len(result["errors"]) > 0

    def test_memory_efficient_parsing(self, parsing_service):
        """Test memory-efficient parsing for large files."""
        large_xml = Path("large_export.xml")

        # Should use streaming parser for large files
        assert hasattr(parsing_service, "parse_large_file")

        # Mock file size check
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 500 * 1024 * 1024  # 500MB

            parser_type = parsing_service._select_parser_type(large_xml)
            assert parser_type == "streaming"

    def test_data_source_abstraction(self, parsing_service):
        """Test abstraction over different data sources."""
        # Should support multiple data sources through common interface
        sources = parsing_service.get_supported_sources()

        assert "apple_health_xml" in sources
        assert "health_auto_export_json" in sources

        # Each source should have a parser
        for source in sources:
            parser = parsing_service.get_parser_for_source(source)
            assert parser is not None

    def test_parsed_data_caching(self, parsing_service):
        """Test caching of parsed data for performance."""
        xml_path = Path("test_export.xml")

        with patch.object(parsing_service, "_xml_parser") as mock_parser:
            mock_parser.parse_file.return_value = []

            with patch("pathlib.Path.is_file", return_value=True):
                with patch(
                    "pathlib.Path.suffix",
                    new_callable=lambda: property(lambda self: ".xml"),
                ):
                    # First parse
                    result1 = parsing_service.parse_health_data(
                        xml_path, use_cache=True
                    )

                    # Second parse should use cache
                    result2 = parsing_service.parse_health_data(
                        xml_path, use_cache=True
                    )

            # Parser should only be called once (3 times for sleep, activity, heart)
            assert mock_parser.parse_file.call_count == 3
            assert result1 == result2
