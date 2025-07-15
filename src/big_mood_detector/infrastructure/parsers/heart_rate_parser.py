"""
Heart Rate Parser for Apple HealthKit Data

Extracts heart rate and HRV records from Apple Health XML exports.
Following Clean Architecture infrastructure patterns.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from xml.etree.ElementTree import ParseError

from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)


class HeartRateParser:
    """Parser for Apple HealthKit heart rate data."""

    RECORD_TAG = "Record"

    # Supported heart metric types from HealthKit
    SUPPORTED_TYPES = [
        "HKQuantityTypeIdentifierHeartRate",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        "HKQuantityTypeIdentifierRestingHeartRate",
        "HKQuantityTypeIdentifierWalkingHeartRateAverage",
        "HKQuantityTypeIdentifierHeartRateRecoveryOneMinute",
    ]

    def parse(self, xml_data: str) -> list[dict[str, Any]]:
        """
        Parse XML data and extract heart rate records.

        Args:
            xml_data: Raw XML string from Apple Health export

        Returns:
            List of heart rate record dictionaries

        Raises:
            ValueError: If XML is invalid
        """
        try:
            root = ET.fromstring(xml_data)
        except ParseError as e:
            raise ValueError(f"Invalid XML: {str(e)}") from e

        heart_records = []

        for record in root.findall(f"./{self.RECORD_TAG}"):
            record_type = record.get("type")
            if record_type in self.SUPPORTED_TYPES:
                heart_records.append(self._extract_heart_data(record))

        return heart_records

    def parse_to_entities(self, xml_data: str) -> list[HeartRateRecord]:
        """
        Parse XML data to domain entities.

        Args:
            xml_data: Raw XML string from Apple Health export

        Returns:
            List of HeartRateRecord domain entities
        """
        raw_records = self.parse(xml_data)
        entities = []

        for record in raw_records:
            try:
                entity = self._create_heart_entity(record)
                entities.append(entity)
            except (ValueError, KeyError):
                # Skip records that can't be converted to entities
                continue

        return entities

    @property
    def supported_heart_types(self) -> list[str]:
        """Get list of supported HealthKit heart metric types."""
        return self.SUPPORTED_TYPES.copy()

    def _extract_heart_data(self, record_element: ET.Element) -> dict[str, Any]:
        """Extract heart data from XML element."""
        data = {
            "type": record_element.get("type"),
            "sourceName": record_element.get("sourceName"),
            "unit": record_element.get("unit"),
            "startDate": record_element.get("startDate"),
            "endDate": record_element.get("endDate"),
            "value": record_element.get("value"),
            "creationDate": record_element.get("creationDate"),
        }

        # Extract motion context if present
        motion_context_elem = record_element.find("HeartRateMotionContext")
        if motion_context_elem is not None:
            data["motionContext"] = motion_context_elem.text

        return data

    def _create_heart_entity(self, record_dict: dict[str, Any]) -> HeartRateRecord:
        """Create HeartRateRecord entity from parsed data."""
        # Parse timestamp (use start date)
        timestamp = self._parse_date(record_dict["startDate"])

        # Convert metric type
        metric_type = HeartMetricType.from_healthkit_identifier(record_dict["type"])

        # Convert value to float
        value = float(record_dict["value"])

        # Parse motion context
        motion_context = MotionContext.from_string(record_dict.get("motionContext"))

        return HeartRateRecord(
            source_name=record_dict["sourceName"],
            timestamp=timestamp,
            metric_type=metric_type,
            value=value,
            unit=record_dict["unit"],
            motion_context=motion_context,
        )

    def _parse_date(self, date_string: str) -> datetime:
        """Parse Apple Health date format."""
        # Apple Health format: "2024-01-01 00:00:00 -0800"
        return datetime.strptime(date_string[:19], "%Y-%m-%d %H:%M:%S")

