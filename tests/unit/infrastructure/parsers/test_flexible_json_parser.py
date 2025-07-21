"""
Test Flexible JSON Parser

TDD for creating a JSON parser that can handle multiple formats:
1. Health Auto Export format (nested metrics)
2. Simple format (flat data array)
"""

import json
import tempfile
from pathlib import Path

class TestFlexibleJSONParser:
    """Test flexible JSON parsing for different formats."""

    def test_parse_simple_sleep_format(self):
        """Test parsing simple sleep JSON format."""
        from big_mood_detector.domain.entities.sleep_record import SleepState
        from big_mood_detector.infrastructure.parsers.json.json_parsers import SleepJSONParser

        parser = SleepJSONParser()

        # Simple format from our integration test
        data = {
            "data": [
                {
                    "sourceName": "AutoSleep",
                    "startDate": "2024-01-01T23:00:00Z",
                    "endDate": "2024-01-02T07:00:00Z",
                    "value": "ASLEEP",
                }
            ]
        }

        records = parser.parse(data)

        assert len(records) == 1
        assert records[0].source_name == "AutoSleep"
        assert records[0].state == SleepState.ASLEEP
        assert records[0].start_date.hour == 23
        assert records[0].end_date.hour == 7

    def test_parse_health_auto_export_format(self):
        """Test parsing Health Auto Export nested format."""
        from big_mood_detector.infrastructure.parsers.json.json_parsers import SleepJSONParser

        parser = SleepJSONParser()

        # Health Auto Export format
        data = {
            "data": {
                "metrics": [
                    {
                        "name": "sleep_analysis",
                        "data": [
                            {
                                "source": "AutoSleep",
                                "start": "2024-01-01T23:00:00Z",
                                "finish": "2024-01-02T07:00:00Z",
                                "asleep": True,
                            }
                        ],
                    }
                ]
            }
        }

        records = parser.parse(data)

        assert len(records) == 1
        assert records[0].source_name == "AutoSleep"

    def test_parse_simple_activity_format(self):
        """Test parsing simple activity JSON format."""
        from big_mood_detector.infrastructure.parsers.json.json_parsers import ActivityJSONParser
        from big_mood_detector.domain.entities.activity_record import ActivityType

        parser = ActivityJSONParser()

        # Simple format
        data = {
            "data": [
                {
                    "sourceName": "iPhone",
                    "startDate": "2024-01-01T08:00:00Z",
                    "endDate": "2024-01-01T09:00:00Z",
                    "value": 1000,
                    "unit": "count",
                }
            ]
        }

        records = parser.parse(data)

        assert len(records) == 1
        assert records[0].source_name == "iPhone"
        assert records[0].value == 1000
        assert records[0].unit == "count"
        assert records[0].activity_type == ActivityType.STEP_COUNT

    def test_handle_empty_data(self):
        """Test handling empty data gracefully."""
        from big_mood_detector.infrastructure.parsers.json.json_parsers import SleepJSONParser

        parser = SleepJSONParser()

        # Empty formats
        assert parser.parse({}) == []
        assert parser.parse({"data": []}) == []
        assert parser.parse({"data": {"metrics": []}}) == []

    def test_handle_malformed_data(self):
        """Test handling malformed data gracefully."""
        from big_mood_detector.infrastructure.parsers.json.json_parsers import SleepJSONParser

        parser = SleepJSONParser()

        # Missing required fields
        data = {
            "data": [
                {
                    "sourceName": "AutoSleep",
                    # Missing dates
                    "value": "ASLEEP",
                }
            ]
        }

        records = parser.parse(data)
        assert len(records) == 0  # Should skip malformed entries

    def test_parse_file_integration(self):
        """Test parsing from actual file."""
        from big_mood_detector.infrastructure.parsers.json.json_parsers import SleepJSONParser

        parser = SleepJSONParser()

        # Create temporary file
        data = {
            "data": [
                {
                    "sourceName": "AutoSleep",
                    "startDate": "2024-01-01T23:00:00Z",
                    "endDate": "2024-01-02T07:00:00Z",
                    "value": "ASLEEP",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            file_path = Path(f.name)

        try:
            records = parser.parse_file(file_path)
            assert len(records) == 1
        finally:
            file_path.unlink()

    def test_timezone_handling(self):
        """Test proper timezone handling in dates."""
        from big_mood_detector.infrastructure.parsers.json.json_parsers import SleepJSONParser

        parser = SleepJSONParser()

        data = {
            "data": [
                {
                    "sourceName": "AutoSleep",
                    "startDate": "2024-01-01T23:00:00+00:00",  # With timezone
                    "endDate": "2024-01-02T07:00:00Z",  # Z notation
                    "value": "ASLEEP",
                }
            ]
        }

        records = parser.parse(data)
        assert len(records) == 1
        # Should handle both timezone formats
