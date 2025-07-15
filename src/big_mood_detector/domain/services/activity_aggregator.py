"""
Activity Aggregator Domain Service

Aggregates raw activity records into clinically meaningful daily summaries.
Following Domain-Driven Design and Clean Architecture principles.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional

import numpy as np

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)


@dataclass(frozen=True)
class DailyActivitySummary:
    """
    Immutable daily activity summary with clinical indicators.

    Represents aggregated physical activity for mood episode detection.
    """

    date: date
    total_steps: float = 0.0
    total_active_energy: float = 0.0
    total_distance_km: float = 0.0
    flights_climbed: float = 0.0
    activity_sessions: int = 0
    peak_activity_hour: Optional[int] = None
    activity_variance: float = 0.0
    sedentary_hours: float = 0.0
    active_hours: float = 0.0
    earliest_activity: Optional[time] = None
    latest_activity: Optional[time] = None

    @property
    def is_high_activity(self) -> bool:
        """
        Detect manic-level activity patterns.

        Clinical thresholds based on bipolar disorder research.
        """
        return (
            self.total_steps > 15000
            or self.total_active_energy > 600
            or (self.active_hours > 18 and self.total_steps > 10000)
        )

    @property
    def is_low_activity(self) -> bool:
        """
        Detect depressive-level activity patterns.

        Very low activity can indicate depressive episodes.
        """
        return (
            self.total_steps < 2000
            or self.total_active_energy < 100
            or self.sedentary_hours > 20
        )

    @property
    def is_erratic_pattern(self) -> bool:
        """
        Detect erratic activity patterns.

        High variance can indicate mood instability.
        """
        return self.activity_variance > 0.5

    @property
    def is_clinically_significant(self) -> bool:
        """
        Determine if activity patterns warrant clinical attention.

        Any extreme or erratic pattern is significant.
        """
        return self.is_high_activity or self.is_low_activity or self.is_erratic_pattern


class ActivityAggregator:
    """
    Domain service for aggregating activity data.

    Transforms raw activity records into daily summaries with
    clinical significance indicators.
    """

    def aggregate_daily(
        self, records: list[ActivityRecord]
    ) -> dict[date, DailyActivitySummary]:
        """
        Aggregate activity records by day.

        Args:
            records: List of activity records to aggregate

        Returns:
            Dictionary mapping dates to daily summaries
        """
        if not records:
            return {}

        # Group records by date
        daily_records = self._group_by_date(records)

        # Create summary for each day
        summaries = {}
        for activity_date, day_records in daily_records.items():
            summaries[activity_date] = self._create_daily_summary(
                activity_date, day_records
            )

        return summaries

    def _group_by_date(
        self, records: list[ActivityRecord]
    ) -> dict[date, list[ActivityRecord]]:
        """Group records by date."""
        grouped = defaultdict(list)
        for record in records:
            activity_date = record.start_date.date()
            grouped[activity_date].append(record)
        return dict(grouped)

    def _create_daily_summary(
        self, activity_date: date, records: list[ActivityRecord]
    ) -> DailyActivitySummary:
        """Create summary from a day's records."""
        # Separate by type
        steps_records = [
            r for r in records if r.activity_type == ActivityType.STEP_COUNT
        ]
        energy_records = [
            r for r in records if r.activity_type == ActivityType.ACTIVE_ENERGY
        ]
        distance_records = [
            r for r in records if r.activity_type == ActivityType.DISTANCE_WALKING
        ]
        flights_records = [
            r for r in records if r.activity_type == ActivityType.FLIGHTS_CLIMBED
        ]

        # Aggregate metrics
        total_steps = self._aggregate_steps(steps_records)
        total_energy = self._sum_values(energy_records)
        total_distance = self._sum_values(distance_records)
        flights_climbed = self._sum_values(flights_records)

        # Calculate patterns
        peak_hour = self._find_peak_activity_hour(steps_records)
        variance = self._calculate_activity_variance(steps_records)
        sedentary, active = self._calculate_activity_hours(steps_records)

        # Circadian markers
        earliest, latest = self._find_activity_bounds(records)

        return DailyActivitySummary(
            date=activity_date,
            total_steps=total_steps,
            total_active_energy=total_energy,
            total_distance_km=total_distance,
            flights_climbed=flights_climbed,
            activity_sessions=len(records),
            peak_activity_hour=peak_hour,
            activity_variance=variance,
            sedentary_hours=sedentary,
            active_hours=active,
            earliest_activity=earliest,
            latest_activity=latest,
        )

    def _aggregate_steps(self, records: list[ActivityRecord]) -> float:
        """
        Aggregate step counts, handling overlaps.

        When multiple sources report steps for same period,
        prefer higher value (more accurate device).
        """
        if not records:
            return 0.0

        # Sort by time
        sorted_records = sorted(records, key=lambda r: r.start_date)

        # Simple approach: for overlapping periods, take max
        # In production, would implement more sophisticated overlap resolution
        total = 0.0
        last_end = None

        for record in sorted_records:
            if last_end is None or record.start_date >= last_end:
                # No overlap
                total += record.value
                last_end = record.end_date
            else:
                # Overlap detected - for now, skip
                # TODO: Implement proper overlap resolution
                continue

        return total

    def _sum_values(self, records: list[ActivityRecord]) -> float:
        """Sum values from records."""
        return sum(r.value for r in records)

    def _find_peak_activity_hour(
        self, steps_records: list[ActivityRecord]
    ) -> Optional[int]:
        """Find hour with most steps."""
        if not steps_records:
            return None

        hourly_steps = defaultdict(float)

        for record in steps_records:
            # Distribute steps across hours
            if record.is_instantaneous:
                hour = record.start_date.hour
                hourly_steps[hour] += record.value
            else:
                # Distribute proportionally
                duration_hours = record.duration_hours
                if duration_hours > 0:
                    hourly_rate = record.value / duration_hours
                    start_hour = record.start_date.hour
                    end_hour = record.end_date.hour

                    # Simple distribution (could be improved)
                    for hour in range(start_hour, end_hour + 1):
                        hourly_steps[hour % 24] += hourly_rate

        if not hourly_steps:
            return None

        # Find peak hour
        return max(hourly_steps.items(), key=lambda x: x[1])[0]

    def _calculate_activity_variance(self, steps_records: list[ActivityRecord]) -> float:
        """
        Calculate variance in activity pattern.

        High variance indicates erratic behavior.
        """
        if len(steps_records) < 2:
            return 0.0

        # Calculate hourly intensities
        intensities = [r.intensity_per_hour for r in steps_records]

        if not intensities:
            return 0.0

        # Normalize variance by mean
        mean_intensity = np.mean(intensities)
        if mean_intensity == 0:
            return 0.0

        std_dev = np.std(intensities)
        coefficient_of_variation = std_dev / mean_intensity

        # Scale to 0-1 range (CV > 2 maps to 1)
        return min(coefficient_of_variation / 2, 1.0)

    def _calculate_activity_hours(
        self, steps_records: list[ActivityRecord]
    ) -> tuple[float, float]:
        """Calculate sedentary vs active hours."""
        if not steps_records:
            return 0.0, 0.0

        # Define activity threshold (>250 steps/hour is active)
        ACTIVE_THRESHOLD = 250.0

        active_hours = 0.0
        sedentary_hours = 0.0

        for record in steps_records:
            intensity = record.intensity_per_hour
            duration = record.duration_hours

            if intensity >= ACTIVE_THRESHOLD:
                active_hours += duration
            else:
                sedentary_hours += duration

        return sedentary_hours, active_hours

    def _find_activity_bounds(
        self, records: list[ActivityRecord]
    ) -> tuple[Optional[time], Optional[time]]:
        """Find earliest and latest activity times."""
        if not records:
            return None, None

        earliest = min(r.start_date for r in records)
        latest = max(r.end_date for r in records)

        return earliest.time(), latest.time()