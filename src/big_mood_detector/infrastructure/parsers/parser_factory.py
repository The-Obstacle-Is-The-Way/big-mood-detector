"""
Smart Parser Factory for Dual Pipeline Architecture
Automatically detects and routes to appropriate parser based on data format.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord

# Import both parser types
from .json import ActivityJSONParser, HeartRateJSONParser, SleepJSONParser
from .xml import ActivityParser, HeartRateParser, SleepParser


class DataParser(Protocol):
    """Protocol for all data parsers."""

    def parse(self, data: Any) -> list[Any]:
        """Parse data and return records."""
        ...


class ParserFactory:
    """
    Smart factory for creating appropriate parsers based on data format.

    Supports:
    - JSON files from Health Auto Export app
    - XML files from native Apple Health export
    """

    @staticmethod
    def detect_format(file_path: str | Path) -> str:
        """
        Detect the format of a health data file.

        Args:
            file_path: Path to the file

        Returns:
            'json' or 'xml' based on content

        Raises:
            ValueError: If format cannot be determined
        """
        file_path = Path(file_path)

        # First try by extension
        if file_path.suffix.lower() == ".json":
            return "json"
        elif file_path.suffix.lower() == ".xml":
            return "xml"

        # Try to parse content
        try:
            with open(file_path) as f:
                first_line = f.readline().strip()
                if first_line.startswith("{") or first_line.startswith("["):
                    return "json"
                elif first_line.startswith("<?xml"):
                    return "xml"
        except Exception:
            pass

        raise ValueError(f"Cannot determine format of {file_path}")

    @staticmethod
    def create_parser(file_path: str | Path) -> Any:
        """
        Create a unified parser that can handle all data types.

        For XML files, returns the StreamingXMLParser.
        For JSON directories, returns a composite parser.
        """
        from .xml.streaming_adapter import StreamingXMLParser

        file_path = Path(file_path)

        if file_path.is_file() and file_path.suffix == ".xml":
            return StreamingXMLParser()
        elif file_path.is_dir():
            # For JSON directories, return a composite parser
            class CompositeJSONParser:
                def __init__(self) -> None:
                    self.sleep_parser = SleepJSONParser()
                    self.activity_parser = ActivityJSONParser()
                    self.heart_parser = HeartRateJSONParser()

                def parse(self, directory: Path) -> dict:
                    """Parse all JSON files in directory."""
                    sleep_records = []
                    activity_records = []
                    heart_records = []

                    # Parse sleep data
                    sleep_file = directory / "Sleep Analysis.json"
                    if sleep_file.exists():
                        with open(sleep_file) as f:
                            data = json.load(f)
                        sleep_records = self.sleep_parser.parse(data)

                    # Parse activity data
                    step_file = directory / "Step Count.json"
                    if step_file.exists():
                        with open(step_file) as f:
                            data = json.load(f)
                        activity_records = self.activity_parser.parse(data)

                    # Parse heart rate data
                    heart_file = directory / "Heart Rate.json"
                    if heart_file.exists():
                        with open(heart_file) as f:
                            data = json.load(f)
                        heart_records = self.heart_parser.parse(data)

                    return {
                        "sleep_records": sleep_records,
                        "activity_records": activity_records,
                        "heart_rate_records": heart_records,
                    }

            return CompositeJSONParser()
        else:
            raise ValueError(f"Unsupported file type or structure: {file_path}")

    @staticmethod
    def create_sleep_parser(format_type: str) -> DataParser:
        """Create appropriate sleep parser based on format."""
        if format_type == "json":
            return SleepJSONParser()
        elif format_type == "xml":
            return SleepParser()
        else:
            raise ValueError(f"Unknown format: {format_type}")

    @staticmethod
    def create_activity_parser(format_type: str) -> DataParser:
        """Create appropriate activity parser based on format."""
        if format_type == "json":
            return ActivityJSONParser()
        elif format_type == "xml":
            return ActivityParser()
        else:
            raise ValueError(f"Unknown format: {format_type}")

    @staticmethod
    def create_heart_rate_parser(format_type: str) -> DataParser:
        """Create appropriate heart rate parser based on format."""
        if format_type == "json":
            return HeartRateJSONParser()
        elif format_type == "xml":
            return HeartRateParser()
        else:
            raise ValueError(f"Unknown format: {format_type}")

    @classmethod
    def parse_file(cls, file_path: str | Path, data_type: str) -> list[Any]:
        """
        Parse a file automatically detecting its format.

        Args:
            file_path: Path to the file
            data_type: Type of data ('sleep', 'activity', 'heart_rate')

        Returns:
            List of parsed records
        """
        file_path = Path(file_path)
        format_type = cls.detect_format(file_path)

        # Create appropriate parser
        if data_type == "sleep":
            parser = cls.create_sleep_parser(format_type)
        elif data_type == "activity":
            parser = cls.create_activity_parser(format_type)
        elif data_type == "heart_rate":
            parser = cls.create_heart_rate_parser(format_type)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Parse based on format
        if format_type == "json":
            with open(file_path) as f:
                data = json.load(f)
            return parser.parse(data)
        elif format_type == "xml":
            with open(file_path) as f:
                xml_data = f.read()
            return parser.parse(xml_data)

        raise ValueError(f"Cannot parse format: {format_type}")


class UnifiedHealthDataParser:
    """
    High-level parser that can handle mixed data sources.

    Example:
        parser = UnifiedHealthDataParser()

        # Parse JSON exports
        parser.add_json_source("data/Sleep Analysis.json", "sleep")
        parser.add_json_source("data/Step Count.json", "activity")

        # Parse XML export
        parser.add_xml_export("export.xml")

        # Get all parsed data
        all_data = parser.get_all_records()
    """

    def __init__(self) -> None:
        """Initialize with empty record collections."""
        self.sleep_records: list[SleepRecord] = []
        self.activity_records: list[ActivityRecord] = []
        self.heart_records: list[HeartRateRecord] = []

    def add_json_source(self, file_path: str | Path, data_type: str) -> None:
        """Add records from a JSON file."""
        records = ParserFactory.parse_file(file_path, data_type)

        if data_type == "sleep":
            self.sleep_records.extend(records)
        elif data_type == "activity":
            self.activity_records.extend(records)
        elif data_type == "heart_rate":
            self.heart_records.extend(records)

    def add_xml_export(self, file_path: str | Path) -> None:
        """Add records from an XML export (parses all data types)."""
        with open(file_path) as f:
            xml_data = f.read()

        # Parse all data types from XML
        sleep_parser = SleepParser()
        activity_parser = ActivityParser()
        heart_parser = HeartRateParser()

        self.sleep_records.extend(sleep_parser.parse_to_entities(xml_data))
        self.activity_records.extend(activity_parser.parse_to_entities(xml_data))
        self.heart_records.extend(heart_parser.parse_to_entities(xml_data))

    def get_all_records(self) -> dict[str, list[Any]]:
        """Get all parsed records organized by type."""
        return {
            "sleep": self.sleep_records,
            "activity": self.activity_records,
            "heart_rate": self.heart_records,
        }
