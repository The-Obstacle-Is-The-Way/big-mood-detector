"""
JSON-based Health Data Parsers for Apple Health Export
Parses aggregated daily data from Apple Health JSON exports.
Following Clean Architecture and SOLID principles.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)


class SleepJSONParser:
    """Parser for Apple Health Sleep Analysis JSON data."""

    def parse(self, data: dict[str, Any]) -> list[SleepRecord]:
        """Parse sleep data from JSON dictionary.

        Args:
            data: Dictionary containing sleep analysis data

        Returns:
            List of SleepRecord entities
        """
        records = []

        # Check if it's simple format (direct data array)
        if "data" in data and isinstance(data["data"], list):
            # Simple format: {"data": [...]}
            for entry in data["data"]:
                record = self._parse_simple_sleep_entry(entry)
                if record:
                    records.append(record)
        else:
            # Health Auto Export format: {"data": {"metrics": [...]}}
            metrics = data.get("data", {}).get("metrics", [])
            for metric in metrics:
                if metric.get("name") == "sleep_analysis":
                    for entry in metric.get("data", []):
                        record = self._parse_sleep_entry(entry)
                        if record:
                            records.append(record)

        return records

    def parse_file(self, file_path: str | Path) -> list[SleepRecord]:
        """Parse sleep data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of SleepRecord entities
        """
        with open(file_path) as f:
            data = json.load(f)
        return self.parse(data)

    def _parse_sleep_entry(self, entry: dict[str, Any]) -> SleepRecord | None:
        """Parse a single sleep entry.

        Args:
            entry: Dictionary containing sleep data for one night

        Returns:
            SleepRecord entity or None if parsing fails
        """
        try:
            # Parse dates - support multiple field names
            start_str = entry.get("sleepStart") or entry.get("start")
            end_str = entry.get("sleepEnd") or entry.get("finish")

            if not start_str or not end_str:
                return None

            # Handle different date formats
            if "T" in start_str:  # ISO format
                start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            else:
                # Parse datetime strings (format: "2025-01-14 23:29:06 -0500")
                start_date = datetime.strptime(start_str[:19], "%Y-%m-%d %H:%M:%S")
                end_date = datetime.strptime(end_str[:19], "%Y-%m-%d %H:%M:%S")

            # For aggregated sleep data, we'll create a single ASLEEP record
            # The individual sleep stages are stored in our test expectations
            # but not in the actual entity

            return SleepRecord(
                source_name=entry.get("source", "Unknown"),
                start_date=start_date,
                end_date=end_date,
                state=SleepState.ASLEEP,
            )

        except (ValueError, KeyError) as e:
            logger.debug(
                "sleep_entry_parse_error", error=str(e), error_type=type(e).__name__
            )
            return None

    def _parse_simple_sleep_entry(self, entry: dict[str, Any]) -> SleepRecord | None:
        """Parse a simple format sleep entry.

        Args:
            entry: Dictionary with simple sleep data format

        Returns:
            SleepRecord entity or None if parsing fails
        """
        try:
            # Parse dates - handle ISO format
            start_str = entry.get("startDate")
            end_str = entry.get("endDate")

            if not start_str or not end_str:
                return None

            # Parse dates - handle both ISO and simple formats
            try:
                # Try ISO format first
                start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                # Fall back to simple format "YYYY-MM-DD HH:MM:SS"
                try:
                    start_date = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                    end_date = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # Last resort - use dateutil if available
                    try:
                        from dateutil import parser

                        start_date = parser.parse(start_str)
                        end_date = parser.parse(end_str)
                    except ImportError:
                        # dateutil not available, re-raise original error
                        raise ValueError(f"Cannot parse date: {start_str}") from None

            # Map value to state
            value = entry.get("value", "").upper()
            state = SleepState.ASLEEP if "ASLEEP" in value else SleepState.IN_BED

            return SleepRecord(
                source_name=entry.get("sourceName", "Unknown"),
                start_date=start_date,
                end_date=end_date,
                state=state,
            )

        except (ValueError, KeyError, AttributeError) as e:
            logger.debug(
                "simple_sleep_entry_parse_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None


class HeartRateJSONParser:
    """Parser for Apple Health Heart Rate JSON data."""

    def parse(self, data: dict[str, Any]) -> list[HeartRateRecord]:
        """Parse heart rate data from JSON dictionary.

        Args:
            data: Dictionary containing heart rate data

        Returns:
            List of HeartRateRecord entities
        """
        records = []

        metrics = data.get("data", {}).get("metrics", [])

        for metric in metrics:
            if metric.get("name") == "heart_rate":
                for entry in metric.get("data", []):
                    record = self._parse_heart_rate_entry(entry)
                    if record:
                        records.append(record)

        return records

    def parse_resting_heart_rate(self, data: dict[str, Any]) -> list[HeartRateRecord]:
        """Parse resting heart rate data.

        Args:
            data: Dictionary containing resting heart rate data

        Returns:
            List of HeartRateRecord entities marked as resting
        """
        records = []

        metrics = data.get("data", {}).get("metrics", [])

        for metric in metrics:
            if metric.get("name") == "resting_heart_rate":
                for entry in metric.get("data", []):
                    record = self._parse_resting_hr_entry(entry)
                    if record:
                        records.append(record)

        return records

    def parse_hrv(self, data: dict[str, Any]) -> list[HeartRateRecord]:
        """Parse heart rate variability data.

        Args:
            data: Dictionary containing HRV data

        Returns:
            List of HeartRateRecord entities with HRV metric type
        """
        records = []

        metrics = data.get("data", {}).get("metrics", [])

        for metric in metrics:
            if metric.get("name") == "heart_rate_variability":
                for entry in metric.get("data", []):
                    record = self._parse_hrv_entry(entry)
                    if record:
                        records.append(record)

        return records

    def parse_file(self, file_path: str | Path) -> list[HeartRateRecord]:
        """Parse heart rate data from JSON file."""
        with open(file_path) as f:
            data = json.load(f)
        return self.parse(data)

    def _parse_heart_rate_entry(self, entry: dict[str, Any]) -> HeartRateRecord | None:
        """Parse a single heart rate entry."""
        try:
            date_str = entry.get("date")
            if not date_str:
                return None

            # Parse date - use noon as the timestamp for daily aggregates
            record_date = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")

            # Use average as the main value
            avg_hr = entry.get("Avg", entry.get("avg", 0))

            return HeartRateRecord(
                source_name=entry.get("source", "Unknown"),
                timestamp=record_date,
                value=avg_hr,
                metric_type=HeartMetricType.HEART_RATE,
                unit="bpm",
                motion_context=MotionContext.UNKNOWN,  # Daily aggregates are mixed
            )

        except (ValueError, KeyError) as e:
            logger.debug(
                "heart_rate_entry_parse_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _parse_resting_hr_entry(self, entry: dict[str, Any]) -> HeartRateRecord | None:
        """Parse a single resting heart rate entry."""
        try:
            date_str = entry.get("date")
            if not date_str:
                return None

            record_date = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")

            return HeartRateRecord(
                source_name=entry.get("source", "Apple Watch"),  # Default source
                timestamp=record_date,
                value=entry.get(
                    "qty", entry.get("value", 0)
                ),  # Handle both 'qty' and 'value'
                metric_type=HeartMetricType.HEART_RATE,
                unit="bpm",
                motion_context=MotionContext.SEDENTARY,  # Resting HR
            )

        except (ValueError, KeyError) as e:
            logger.debug(
                "resting_hr_entry_parse_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _parse_hrv_entry(self, entry: dict[str, Any]) -> HeartRateRecord | None:
        """Parse a single HRV entry."""
        try:
            date_str = entry.get("date")
            if not date_str:
                return None

            record_date = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")

            # HRV data uses 'qty' field in Apple Health export
            value = entry.get("qty", entry.get("value", 0))

            return HeartRateRecord(
                source_name=entry.get("source", "Apple Watch"),  # Default source
                timestamp=record_date,
                value=value,
                metric_type=HeartMetricType.HRV_SDNN,
                unit="ms",
                motion_context=MotionContext.UNKNOWN,
            )

        except (ValueError, KeyError) as e:
            logger.debug(
                "hrv_entry_parse_error", error=str(e), error_type=type(e).__name__
            )
            return None


class ActivityJSONParser:
    """Parser for Apple Health Activity JSON data."""

    def parse(self, data: dict[str, Any]) -> list[ActivityRecord]:
        """Parse step count data from JSON dictionary.

        Args:
            data: Dictionary containing step count data

        Returns:
            List of ActivityRecord entities
        """
        records = []

        # Check if it's simple format (direct data array)
        if "data" in data and isinstance(data["data"], list):
            # Simple format: {"data": [...]}
            for entry in data["data"]:
                record = self._parse_simple_activity_entry(entry)
                if record:
                    records.append(record)
        else:
            # Health Auto Export format: {"data": {"metrics": [...]}}
            metrics = data.get("data", {}).get("metrics", [])
            for metric in metrics:
                metric_name = metric.get("name", "")
                # Handle both named metrics and unnamed ones (default to step count)
                if "step" in metric_name.lower() or metric_name == "":
                    for entry in metric.get("data", []):
                        record = self._parse_step_entry(entry)
                        if record:
                            records.append(record)

        return records

    def parse_distance(self, data: dict[str, Any]) -> list[ActivityRecord]:
        """Parse walking/running distance data.

        Args:
            data: Dictionary containing distance data

        Returns:
            List of ActivityRecord entities
        """
        records = []

        metrics = data.get("data", {}).get("metrics", [])

        for metric in metrics:
            metric_name = metric.get("name", "")
            if "distance" in metric_name.lower():
                for entry in metric.get("data", []):
                    record = self._parse_distance_entry(entry)
                    if record:
                        records.append(record)

        return records

    def parse_file(self, file_path: str | Path) -> list[ActivityRecord]:
        """Parse activity data from JSON file."""
        with open(file_path) as f:
            data = json.load(f)
        return self.parse(data)

    def _parse_step_entry(self, entry: dict[str, Any]) -> ActivityRecord | None:
        """Parse a single step count entry."""
        try:
            date_str = entry.get("date")
            if not date_str:
                return None

            # Handle multiple sources
            source = entry.get("source", "Unknown")
            if "|" in source:
                # Take the first source if multiple
                source = source.split("|")[0]

            # For daily aggregates, use the whole day as the period
            start_date = datetime.strptime(
                date_str[:10] + " 00:00:00", "%Y-%m-%d %H:%M:%S"
            )
            end_date = datetime.strptime(
                date_str[:10] + " 23:59:59", "%Y-%m-%d %H:%M:%S"
            )

            return ActivityRecord(
                source_name=source,
                start_date=start_date,
                end_date=end_date,
                activity_type=ActivityType.STEP_COUNT,
                value=entry.get("qty", 0),
                unit="count",
            )

        except (ValueError, KeyError) as e:
            logger.debug(
                "step_entry_parse_error", error=str(e), error_type=type(e).__name__
            )
            return None

    def _parse_simple_activity_entry(
        self, entry: dict[str, Any]
    ) -> ActivityRecord | None:
        """Parse a simple format activity entry.

        Args:
            entry: Dictionary with simple activity data format

        Returns:
            ActivityRecord entity or None if parsing fails
        """
        try:
            # Parse dates - handle ISO format
            start_str = entry.get("startDate")
            end_str = entry.get("endDate")

            if not start_str or not end_str:
                return None

            # Parse dates - handle both ISO and simple formats
            try:
                # Try ISO format first
                start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                # Fall back to simple format "YYYY-MM-DD HH:MM:SS"
                try:
                    start_date = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                    end_date = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # Last resort - use dateutil if available
                    try:
                        from dateutil import parser

                        start_date = parser.parse(start_str)
                        end_date = parser.parse(end_str)
                    except ImportError:
                        # dateutil not available, re-raise original error
                        raise ValueError(f"Cannot parse date: {start_str}") from None

            # Determine activity type (default to step count)
            unit = entry.get("unit", "count").lower()
            if "count" in unit:
                activity_type = ActivityType.STEP_COUNT
            elif "km" in unit or "meter" in unit:
                activity_type = ActivityType.DISTANCE_WALKING
            else:
                activity_type = ActivityType.STEP_COUNT  # Default

            return ActivityRecord(
                source_name=entry.get("sourceName", "Unknown"),
                start_date=start_date,
                end_date=end_date,
                activity_type=activity_type,
                value=float(entry.get("value", 0)),
                unit=entry.get("unit", "count"),
            )

        except (ValueError, KeyError, AttributeError) as e:
            logger.debug(
                "simple_activity_entry_parse_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _parse_distance_entry(self, entry: dict[str, Any]) -> ActivityRecord | None:
        """Parse a single distance entry."""
        try:
            date_str = entry.get("date")
            if not date_str:
                return None

            source = entry.get("source", "Unknown")
            if "|" in source:
                source = source.split("|")[0]

            # For daily aggregates, use the whole day as the period
            start_date = datetime.strptime(
                date_str[:10] + " 00:00:00", "%Y-%m-%d %H:%M:%S"
            )
            end_date = datetime.strptime(
                date_str[:10] + " 23:59:59", "%Y-%m-%d %H:%M:%S"
            )

            return ActivityRecord(
                source_name=source,
                start_date=start_date,
                end_date=end_date,
                activity_type=ActivityType.DISTANCE_WALKING,
                value=entry.get("qty", 0),
                unit="km",
            )

        except (ValueError, KeyError) as e:
            logger.debug(
                "distance_entry_parse_error", error=str(e), error_type=type(e).__name__
            )
            return None
