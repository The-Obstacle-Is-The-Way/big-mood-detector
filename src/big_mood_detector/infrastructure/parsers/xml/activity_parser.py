"""
Activity Parser for Apple HealthKit Data

Extracts physical activity records from Apple Health XML exports.
Following Clean Architecture infrastructure patterns.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from xml.etree.ElementTree import ParseError

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)


class ActivityParser:
    """Parser for Apple HealthKit activity data."""

    RECORD_TAG = "Record"

    # Supported activity types from HealthKit
    SUPPORTED_TYPES = [
        "HKQuantityTypeIdentifierStepCount",
        "HKQuantityTypeIdentifierActiveEnergyBurned",
        "HKQuantityTypeIdentifierDistanceWalkingRunning",
        "HKQuantityTypeIdentifierFlightsClimbed",
        "HKQuantityTypeIdentifierBasalEnergyBurned",
        "HKQuantityTypeIdentifierAppleExerciseTime",
        "HKQuantityTypeIdentifierAppleStandTime",
    ]

    def parse(self, xml_data: str | ET.Element) -> list[dict[str, Any]]:
        """
        Parse XML data and extract activity records.

        Args:
            xml_data: Raw XML string or ElementTree.Element from Apple Health export

        Returns:
            List of activity record dictionaries

        Raises:
            ValueError: If XML is invalid
        """
        if isinstance(xml_data, str):
            try:
                root = ET.fromstring(xml_data)
            except ParseError as e:
                raise ValueError(f"Invalid XML: {str(e)}") from e
        elif isinstance(xml_data, ET.Element):
            root = xml_data
        else:
            raise ValueError(
                f"Expected string or ElementTree.Element, got {type(xml_data)}"
            )

        activity_records = []

        for record in root.findall(f"./{self.RECORD_TAG}"):
            record_type = record.get("type")
            if record_type in self.SUPPORTED_TYPES:
                activity_records.append(self._extract_activity_data(record))

        return activity_records

    def parse_to_entities(self, xml_data: str | ET.Element) -> list[ActivityRecord]:
        """
        Parse XML data to domain entities.

        Args:
            xml_data: Raw XML string or ElementTree.Element from Apple Health export

        Returns:
            List of ActivityRecord domain entities
        """
        raw_records = self.parse(xml_data)
        entities = []

        for record in raw_records:
            try:
                entity = self._create_activity_entity(record)
                entities.append(entity)
            except (ValueError, KeyError):
                # Skip records that can't be converted to entities
                continue

        return entities

    @property
    def supported_activity_types(self) -> list[str]:
        """Get list of supported HealthKit activity types."""
        return self.SUPPORTED_TYPES.copy()

    def _extract_activity_data(self, record_element: ET.Element) -> dict[str, Any]:
        """Extract activity data from XML element."""
        return {
            "type": record_element.get("type"),
            "sourceName": record_element.get("sourceName"),
            "unit": record_element.get("unit"),
            "startDate": record_element.get("startDate"),
            "endDate": record_element.get("endDate"),
            "value": record_element.get("value"),
        }

    def _create_activity_entity(self, record_dict: dict[str, Any]) -> ActivityRecord:
        """Create ActivityRecord entity from parsed data."""
        # Parse dates
        start_date = self._parse_date(record_dict["startDate"])
        end_date = self._parse_date(record_dict["endDate"])

        # Convert activity type
        activity_type = ActivityType.from_healthkit_identifier(record_dict["type"])

        # Convert value to float
        value = float(record_dict["value"])

        return ActivityRecord(
            source_name=record_dict["sourceName"],
            start_date=start_date,
            end_date=end_date,
            activity_type=activity_type,
            value=value,
            unit=record_dict["unit"],
        )

    def _parse_date(self, date_string: str) -> datetime:
        """Parse Apple Health date format."""
        # Apple Health format: "2024-01-01 00:00:00 -0800"
        return datetime.strptime(date_string[:19], "%Y-%m-%d %H:%M:%S")
