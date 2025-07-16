"""
Sleep Aggregator Domain Service

Aggregates sleep records into daily summaries for clinical analysis.
Following Domain-Driven Design and Clean Code principles.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from big_mood_detector.domain.entities.sleep_record import SleepRecord


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
                return datetime.combine(self.date + timedelta(days=1), self.latest_wake_time)
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
            return {}

        # Group records by date
        daily_records = self._group_by_date(sleep_records)

        # Create summary for each day
        summaries = {}
        for day, records in daily_records.items():
            summaries[day] = self._create_daily_summary(day, records)

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

        Uses the "sleep day" concept - sleep after 6 PM belongs to that night,
        sleep before 6 PM belongs to previous night.
        """
        midpoint = record.start_date + (record.end_date - record.start_date) / 2

        # If sleep starts after 6 PM, it's that day's sleep
        if record.start_date.hour >= 18:
            return record.start_date.date()
        # If sleep starts before 6 AM, it's previous day's sleep
        elif record.start_date.hour < 6:
            return (record.start_date - timedelta(days=1)).date()
        # Otherwise use midpoint
        else:
            return (
                midpoint.date()
                if midpoint.hour >= 12
                else (midpoint - timedelta(days=1)).date()
            )

    def _create_daily_summary(
        self, day: date, records: list[SleepRecord]
    ) -> DailySleepSummary:
        """Create a daily summary from sleep records."""
        # Calculate basic metrics
        total_bed_time = sum(r.duration_hours for r in records)
        total_sleep_time = sum(r.duration_hours for r in records if r.is_actual_sleep)

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
