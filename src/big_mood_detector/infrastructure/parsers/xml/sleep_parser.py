"""
Apple HealthKit Sleep Data Parser

Clinical-grade sleep data extraction from Apple Health exports.
Based on proven reference implementation.

Following SOLID principles:
- Single Responsibility: Only parses sleep data from XML
- Open/Closed: Extensible for new sleep record types
- Liskov Substitution: Can be substituted with other parsers
- Interface Segregation: Focused interface for sleep parsing
- Dependency Inversion: Returns domain entities, not raw data
"""

from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class SleepParser:
    """Parses Apple HealthKit sleep data from XML exports."""

    # Constants following DRY principle
    SLEEP_RECORD_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
    RECORD_TAG = "Record"

    # Date format from HealthKit exports
    HEALTHKIT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"

    def __init__(self) -> None:
        """Initialize sleep parser."""
        pass

    def parse(self, xml_data: str) -> list[dict[str, Any]]:
        """
        Parse Apple HealthKit XML and extract sleep records.

        Args:
            xml_data: XML string containing HealthKit export data

        Returns:
            List of sleep record dictionaries with keys:
            - sourceName: Device that recorded the data
            - startDate: Sleep start timestamp
            - endDate: Sleep end timestamp
            - value: Sleep state (InBed or Asleep)

        Raises:
            ValueError: If XML is invalid or malformed
        """
        try:
            root = ET.fromstring(xml_data)
        except ParseError as e:
            raise ValueError(f"Invalid XML: {str(e)}") from e

        sleep_records = []

        # Extract only sleep analysis records
        for record in root.findall(f"./{self.RECORD_TAG}"):
            if record.get("type") == self.SLEEP_RECORD_TYPE:
                sleep_records.append(self._extract_sleep_data(record))

        return sleep_records

    def parse_to_entities(self, xml_data: str) -> list[SleepRecord]:
        """
        Parse XML and return domain entities.

        This method follows Open/Closed principle - extends functionality
        without modifying existing parse() method.
        """
        raw_records = self.parse(xml_data)
        return [self._to_domain_entity(record) for record in raw_records]

    def _extract_sleep_data(self, element: ET.Element) -> dict[str, Any]:
        """
        Extract sleep data from a single XML element.

        Single Responsibility: This method only extracts data from one element.
        """
        return {
            "sourceName": element.get("sourceName"),
            "startDate": element.get("startDate"),
            "endDate": element.get("endDate"),
            "value": element.get("value"),
        }

    def _to_domain_entity(self, raw_record: dict[str, Any]) -> SleepRecord:
        """
        Convert raw dictionary to domain entity.

        Factory Method pattern - encapsulates object creation.
        """
        return SleepRecord(
            source_name=raw_record["sourceName"],
            start_date=self._parse_date(raw_record["startDate"]),
            end_date=self._parse_date(raw_record["endDate"]),
            state=SleepState.from_healthkit_value(raw_record["value"]),
        )

    def _parse_date(self, date_string: str) -> datetime:
        """Parse HealthKit date format to datetime object."""
        return datetime.strptime(date_string, self.HEALTHKIT_DATE_FORMAT)
