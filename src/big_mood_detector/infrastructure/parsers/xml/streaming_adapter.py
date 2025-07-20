"""
Streaming XML Adapter for Large Apple Health Exports

Uses iterparse for memory-efficient streaming of large XML files.
Based on Simple-Apple-Health-XML-to-CSV approach.
"""

import xml.etree.ElementTree as ET
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.infrastructure.parsers.xml import (
    ActivityParser,
    HeartRateParser,
    SleepParser,
)


class StreamingXMLParser:
    """
    Memory-efficient XML parser using iterparse.

    Streams through large Apple Health export files without loading
    the entire document into memory.
    """

    def __init__(self) -> None:
        """Initialize with existing parsers for entity conversion."""
        self.sleep_parser = SleepParser()
        self.activity_parser = ActivityParser()
        self.heart_parser = HeartRateParser()

    def iter_records(
        self,
        file_path: str | Path,
        record_types: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream records from XML file using iterparse.

        Args:
            file_path: Path to the XML file
            record_types: Optional list of record types to filter
            start_date: Optional start date (YYYY-MM-DD format) to filter records
            end_date: Optional end date (YYYY-MM-DD format) to filter records

        Yields:
            Dictionary of record attributes

        Raises:
            ValueError: If file not found or XML parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        try:
            # Use iterparse for memory efficiency
            for event, elem in ET.iterparse(str(file_path), events=("end",)):
                if event == "end" and elem.tag == "Record":
                    record_type = elem.get("type")

                    # Filter by record types if specified
                    if record_types and record_type not in record_types:
                        elem.clear()
                        continue

                    # Extract all attributes
                    record_data = dict(elem.attrib)

                    # Check date filtering if enabled
                    if start_date or end_date:
                        # Apple Health uses startDate attribute
                        record_date_str = record_data.get("startDate", "")
                        if record_date_str:
                            # Parse date with timezone awareness
                            try:
                                # Parse the full datetime string with timezone
                                record_datetime = date_parser.parse(record_date_str)
                                # Convert to UTC for consistent comparison
                                record_date_utc = record_datetime.date()

                                # Apply date filters
                                if start_date:
                                    start = datetime.strptime(
                                        start_date, "%Y-%m-%d"
                                    ).date()
                                    if record_date_utc < start:
                                        elem.clear()
                                        continue

                                if end_date:
                                    end = datetime.strptime(end_date, "%Y-%m-%d").date()
                                    if record_date_utc > end:
                                        elem.clear()
                                        continue
                            except (ValueError, TypeError):
                                # Skip records with invalid dates
                                elem.clear()
                                continue

                    # Extract metadata entries (e.g., heart rate motion context)
                    for metadata in elem.findall("MetadataEntry"):
                        key = metadata.get("key")
                        value = metadata.get("value")
                        if (
                            key == "HKMetadataKeyHeartRateMotionContext"
                            and value is not None
                        ):
                            record_data["motionContext"] = value

                    yield record_data

                    # Clear the element to free memory
                    elem.clear()

        except ET.ParseError as e:
            raise ValueError(f"XML parsing error: {str(e)}") from e

    def parse_file(
        self,
        file_path: str | Path,
        entity_type: str = "all",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Generator[SleepRecord | ActivityRecord | HeartRateRecord, None, None]:
        """
        Parse file and yield domain entities.

        Args:
            file_path: Path to the XML file
            entity_type: Type of entities to parse ('sleep', 'activity', 'heart', 'all')
            start_date: Optional start date (YYYY-MM-DD format) to filter records
            end_date: Optional end date (YYYY-MM-DD format) to filter records

        Yields:
            Domain entities
        """
        # Define record types for each entity
        sleep_types = ["HKCategoryTypeIdentifierSleepAnalysis"]
        activity_types = self.activity_parser.supported_activity_types
        heart_types = self.heart_parser.supported_heart_types

        # Determine which types to parse
        if entity_type == "sleep":
            types_to_parse = sleep_types
        elif entity_type == "activity":
            types_to_parse = activity_types
        elif entity_type == "heart":
            types_to_parse = heart_types
        else:  # 'all'
            types_to_parse = sleep_types + activity_types + heart_types

        # Stream through records
        for record_dict in self.iter_records(
            file_path, types_to_parse, start_date, end_date
        ):
            record_type = record_dict.get("type")

            try:
                # Convert to appropriate entity based on type
                if record_type in sleep_types:
                    # Create minimal XML for parser compatibility
                    elem = self._dict_to_element(record_dict)
                    sleep_entities = self.sleep_parser.parse_to_entities(elem)
                    yield from sleep_entities

                elif record_type in activity_types:
                    elem = self._dict_to_element(record_dict)
                    activity_entities = self.activity_parser.parse_to_entities(elem)
                    yield from activity_entities

                elif record_type in heart_types:
                    elem = self._dict_to_element(record_dict)
                    heart_entities = self.heart_parser.parse_to_entities(elem)
                    yield from heart_entities

            except (ValueError, KeyError):
                # Skip records that can't be converted
                continue

    def parse_file_in_batches(
        self,
        file_path: str | Path,
        batch_size: int = 1000,
        entity_type: str = "all",
    ) -> Generator[list[Any], None, None]:
        """
        Parse file and yield entities in batches.

        Args:
            file_path: Path to the XML file
            batch_size: Number of entities per batch
            entity_type: Type of entities to parse

        Yields:
            List of domain entities
        """
        batch = []

        for entity in self.parse_file(file_path, entity_type):
            batch.append(entity)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining entities
        if batch:
            yield batch

    def _dict_to_element(self, record_dict: dict[str, Any]) -> ET.Element:
        """Convert record dictionary to minimal XML element for parser compatibility."""
        # Create a minimal HealthData root with one Record
        root = ET.Element("HealthData")
        record = ET.SubElement(root, "Record")

        # Set attributes
        for key, value in record_dict.items():
            if key == "motionContext":
                # Recreate metadata entry for heart rate motion context
                meta_elem = ET.SubElement(record, "MetadataEntry")
                meta_elem.set("key", "HKMetadataKeyHeartRateMotionContext")
                meta_elem.set("value", value)
            else:
                record.set(key, str(value))

        return root
