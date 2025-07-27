"""
Sleep Aggregator Domain Service

Aggregates sleep records into daily summaries for clinical analysis.
Following Domain-Driven Design and Clean Code principles.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from big_mood_detector.domain.entities.sleep_record import SleepRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailySleepSummary:
    """
    Daily sleep summary with clinical metrics.

    Value Object representing aggregated sleep data for one day.
    """

    date: date
    total_time_in_bed_hours: float
    total_sleep_hours: float
    sleep_efficiency: float  # Percentage of time in bed actually sleeping
    sleep_sessions: int
    longest_sleep_hours: float
    sleep_fragmentation_index: float  # Higher = more fragmented

    # Circadian rhythm indicators
    earliest_bedtime: time | None = None
    latest_wake_time: time | None = None
    mid_sleep_time: datetime | None = None  # Midpoint of main sleep

    @property
    def is_clinically_significant(self) -> bool:
        """Check if sleep pattern indicates potential mood episode risk."""
        return (
            self.total_sleep_hours < 4.0  # Too little sleep
            or self.total_sleep_hours > 10.0  # Too much sleep
            or self.sleep_efficiency < 0.70  # Poor quality
            or self.sleep_fragmentation_index > 0.3  # Highly fragmented
        )

    @property
    def sleep_onset(self) -> datetime:
        """Estimated sleep onset time from earliest bedtime."""
        if self.earliest_bedtime and self.date:
            return datetime.combine(self.date, self.earliest_bedtime)
        # Default to 11 PM if no bedtime data
        return datetime.combine(self.date, time(23, 0))

    @property
    def wake_time(self) -> datetime:
        """Estimated wake time from latest wake time."""
        if self.latest_wake_time and self.date:
            # If wake time is early morning, it's next day
            if self.latest_wake_time.hour < 12:
                return datetime.combine(
                    self.date + timedelta(days=1), self.latest_wake_time
                )
            return datetime.combine(self.date, self.latest_wake_time)
        # Default to 7 AM next day if no wake time data
        return datetime.combine(self.date + timedelta(days=1), time(7, 0))


class SleepAggregator:
    """
    Domain service for aggregating sleep data.

    Follows Single Responsibility: Only aggregates sleep records.
    """

    def aggregate_daily(
        self, sleep_records: list[SleepRecord]
    ) -> dict[date, DailySleepSummary]:
        """
        Aggregate sleep records into daily summaries.

        Args:
            sleep_records: List of sleep records to aggregate

        Returns:
            Dictionary mapping dates to daily sleep summaries
        """
        if not sleep_records:
            logger.debug("No sleep records to aggregate")
            return {}

        logger.debug(f"Aggregating {len(sleep_records)} sleep records")

        # Group records by date
        daily_records = self._group_by_date(sleep_records)
        logger.debug(f"Grouped into {len(daily_records)} days")

        # Create summary for each day
        summaries = {}
        for day, records in daily_records.items():
            summaries[day] = self._create_daily_summary(day, records)
            logger.debug(
                f"Day {day}: {len(records)} records, "
                f"total sleep {summaries[day].total_sleep_hours:.1f}h"
            )

        return summaries

    def _group_by_date(
        self, records: list[SleepRecord]
    ) -> dict[date, list[SleepRecord]]:
        """Group sleep records by the date they primarily belong to."""
        grouped = defaultdict(list)

        for record in records:
            # Assign to date based on when most of sleep occurred
            sleep_date = self._determine_sleep_date(record)
            grouped[sleep_date].append(record)

        return dict(grouped)

    def _determine_sleep_date(self, record: SleepRecord) -> date:
        """
        Determine which date a sleep record belongs to.

        Uses Apple Health convention:
        - Sleep is assigned to the date you wake up
        - If you wake up before 3pm, it's assigned to that day
        - If you wake up after 3pm, it's assigned to the next day

        This matches how Apple Health exports data and ensures consistency.
        """
        # Get the wake time (end of sleep)
        wake_time = record.end_date

        # If wake time is at or before 3pm (15:00), assign to wake date
        # Using <= 15 to include 3:00-3:59pm as same day
        if wake_time.hour < 15 or (wake_time.hour == 15 and wake_time.minute == 0):
            assigned_date = wake_time.date()
        else:  # After 3pm
            # Assign to next day
            assigned_date = (wake_time + timedelta(days=1)).date()

        logger.debug(
            f"Sleep {record.start_date.strftime('%Y-%m-%d %H:%M')} to "
            f"{record.end_date.strftime('%Y-%m-%d %H:%M')} "
            f"(duration {record.duration_hours:.1f}h) -> assigned to {assigned_date}"
        )

        return assigned_date

    def _create_daily_summary(
        self, day: date, records: list[SleepRecord]
    ) -> DailySleepSummary:
        """Create a daily summary from sleep records."""
        # Separate sleep records from in-bed records
        sleep_records = [r for r in records if r.is_actual_sleep]

        # Calculate merged durations to handle overlapping records
        # This fixes the bug where multiple devices recording simultaneously
        # would result in inflated sleep durations
        total_bed_time = self._calculate_merged_duration(records)
        total_sleep_time = self._calculate_merged_duration(sleep_records)

        # Log if we detected significant overlap
        raw_bed_time = sum(r.duration_hours for r in records)
        if raw_bed_time > total_bed_time * 1.1:  # More than 10% overlap
            overlap_hours = raw_bed_time - total_bed_time
            logger.warning(
                f"Detected {overlap_hours:.1f}h of overlapping sleep records for {day}. "
                f"Raw total: {raw_bed_time:.1f}h, merged: {total_bed_time:.1f}h. "
                f"This typically occurs when multiple devices record sleep simultaneously."
            )

        # Sanity check - cap at 24 hours
        if total_bed_time > 24.0:
            logger.error(
                f"Total bed time {total_bed_time:.1f}h still exceeds 24h after merging for {day}. "
                f"This indicates a data quality issue."
            )
            total_bed_time = 24.0
            total_sleep_time = min(total_sleep_time, total_bed_time)

        # Sleep efficiency
        efficiency = total_sleep_time / total_bed_time if total_bed_time > 0 else 0.0

        # Find longest continuous sleep
        longest_sleep = max(
            (r.duration_hours for r in records if r.is_actual_sleep), default=0.0
        )

        # Calculate fragmentation
        sleep_sessions = [r for r in records if r.is_actual_sleep]
        fragmentation = self._calculate_fragmentation(sleep_sessions)

        # Circadian rhythm indicators
        earliest_bed, latest_wake, mid_sleep = self._calculate_circadian_markers(
            records
        )

        logger.debug(
            f"Summary for {day}: "
            f"sleep={total_sleep_time:.1f}h, "
            f"bed={total_bed_time:.1f}h, "
            f"efficiency={efficiency:.1%}, "
            f"sessions={len(sleep_sessions)}, "
            f"fragmentation={fragmentation:.2f}"
        )

        return DailySleepSummary(
            date=day,
            total_time_in_bed_hours=total_bed_time,
            total_sleep_hours=total_sleep_time,
            sleep_efficiency=efficiency,
            sleep_sessions=len(sleep_sessions),
            longest_sleep_hours=longest_sleep,
            sleep_fragmentation_index=fragmentation,
            earliest_bedtime=earliest_bed,
            latest_wake_time=latest_wake,
            mid_sleep_time=mid_sleep,
        )

    def _calculate_fragmentation(self, sleep_sessions: list[SleepRecord]) -> float:
        """
        Calculate sleep fragmentation index.

        Higher values indicate more fragmented sleep (bad for mood stability).
        """
        if len(sleep_sessions) <= 1:
            return 0.0

        # Sort by start time
        sorted_sessions = sorted(sleep_sessions, key=lambda r: r.start_date)

        # Calculate gaps between sleep sessions
        total_gap_time = 0.0
        for i in range(1, len(sorted_sessions)):
            gap = (
                sorted_sessions[i].start_date - sorted_sessions[i - 1].end_date
            ).total_seconds() / 3600
            total_gap_time += gap

        # Total sleep period (first sleep start to last sleep end)
        total_period = (
            sorted_sessions[-1].end_date - sorted_sessions[0].start_date
        ).total_seconds() / 3600

        # Fragmentation index = gap time / total period
        return total_gap_time / total_period if total_period > 0 else 0.0

    def _calculate_circadian_markers(
        self, records: list[SleepRecord]
    ) -> tuple[time | None, time | None, datetime | None]:
        """Calculate circadian rhythm markers from sleep records."""
        if not records:
            return None, None, None

        # Sort by start time
        sorted_records = sorted(records, key=lambda r: r.start_date)

        # Earliest bedtime
        earliest_bed = min(r.start_date.time() for r in sorted_records)

        # Latest wake time
        latest_wake = max(r.end_date.time() for r in sorted_records)

        # Find main sleep period (longest continuous sleep)
        main_sleep = max(sorted_records, key=lambda r: r.duration_hours)
        mid_sleep = (
            main_sleep.start_date + (main_sleep.end_date - main_sleep.start_date) / 2
        )

        return earliest_bed, latest_wake, mid_sleep

    def _merge_overlapping_intervals(
        self, intervals: list[tuple[datetime, datetime]]
    ) -> list[tuple[datetime, datetime]]:
        """
        Merge overlapping time intervals.

        This is the core algorithm for handling overlapping sleep records
        from multiple devices (e.g., iPhone and Apple Watch recording simultaneously).

        Args:
            intervals: List of (start, end) datetime tuples

        Returns:
            List of merged non-overlapping intervals
        """
        if not intervals:
            return []

        # Sort intervals by start time
        sorted_intervals = sorted(intervals, key=lambda x: x[0])

        # Initialize with first interval
        merged = [sorted_intervals[0]]

        for start, end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]

            # Check if current interval overlaps with the last merged interval
            if start <= last_end:
                # Overlapping - extend the last interval
                merged[-1] = (last_start, max(last_end, end))
            else:
                # Non-overlapping - add as new interval
                merged.append((start, end))

        return merged

    def _calculate_merged_duration(
        self, records: list[SleepRecord]
    ) -> float:
        """
        Calculate total duration from sleep records, merging overlaps.
        
        Args:
            records: List of sleep records to merge
            
        Returns:
            Total hours after merging overlapping periods
        """
        if not records:
            return 0.0

        # Convert records to intervals
        intervals = [(r.start_date, r.end_date) for r in records]

        # Merge overlapping intervals
        merged = self._merge_overlapping_intervals(intervals)

        # Calculate total duration from merged intervals
        total_seconds = sum(
            (end - start).total_seconds() for start, end in merged
        )

        return total_seconds / 3600  # Convert to hours
