"""
Heart Rate Aggregator Domain Service

Aggregates raw heart rate records into clinically meaningful daily summaries.
Following Domain-Driven Design and Clean Architecture principles.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)


@dataclass(frozen=True)
class DailyHeartSummary:
    """
    Immutable daily heart rate summary with clinical indicators.

    Represents aggregated heart metrics for mood episode detection.
    """

    date: date
    avg_resting_hr: float = 0.0
    min_hr: float = 0.0
    max_hr: float = 0.0
    avg_hrv_sdnn: float = 0.0
    min_hrv_sdnn: float = 0.0  # Track minimum HRV for clinical significance
    hr_measurements: int = 0
    hrv_measurements: int = 0
    high_hr_episodes: int = 0
    low_hr_episodes: int = 0
    circadian_hr_range: float = 0.0
    morning_hr: float = 0.0
    evening_hr: float = 0.0

    @property
    def has_high_resting_hr(self) -> bool:
        """
        Detect elevated resting heart rate.

        Clinical threshold: >90 bpm at rest
        May indicate: anxiety, mania, hyperthyroidism
        """
        return self.avg_resting_hr > 90

    @property
    def has_low_hrv(self) -> bool:
        """
        Detect low heart rate variability.

        Clinical threshold: <20 ms SDNN
        Indicates: poor autonomic function, stress, poor recovery
        """
        # Check both average and minimum HRV - a single low reading is clinically significant
        return (self.avg_hrv_sdnn > 0 and self.avg_hrv_sdnn < 20) or (self.min_hrv_sdnn > 0 and self.min_hrv_sdnn < 20)

    @property
    def has_abnormal_circadian_rhythm(self) -> bool:
        """
        Detect abnormal circadian heart rate pattern.

        Normal: 10-30 bpm difference between morning/evening
        Abnormal: <10 bpm (flat) or >30 bpm (excessive)
        """
        if self.circadian_hr_range == 0:
            return False
        return self.circadian_hr_range < 10 or self.circadian_hr_range > 30

    @property
    def is_clinically_significant(self) -> bool:
        """
        Determine if heart patterns warrant clinical attention.

        Significant if:
        - High resting HR (stress/mania indicator)
        - Low HRV (autonomic dysfunction)
        - Abnormal circadian rhythm
        - Frequent episodes of abnormal HR
        """
        return (
            self.has_high_resting_hr
            or self.has_low_hrv
            or self.has_abnormal_circadian_rhythm
            or self.high_hr_episodes > 10
            or self.low_hr_episodes > 5
        )


class HeartRateAggregator:
    """
    Domain service for aggregating heart rate data.

    Transforms raw heart rate records into daily summaries with
    clinical significance indicators for mood disorder detection.
    """

    def aggregate_daily(
        self, records: list[HeartRateRecord]
    ) -> dict[date, DailyHeartSummary]:
        """
        Aggregate heart rate records by day.

        Args:
            records: List of heart rate records to aggregate

        Returns:
            Dictionary mapping dates to daily summaries
        """
        if not records:
            return {}

        # Group records by date
        daily_records = self._group_by_date(records)

        # Create summary for each day
        summaries = {}
        for record_date, day_records in daily_records.items():
            summaries[record_date] = self._create_daily_summary(
                record_date, day_records
            )

        return summaries

    def _group_by_date(
        self, records: list[HeartRateRecord]
    ) -> dict[date, list[HeartRateRecord]]:
        """Group records by date."""
        grouped = defaultdict(list)
        for record in records:
            record_date = record.timestamp.date()
            grouped[record_date].append(record)
        return dict(grouped)

    def _create_daily_summary(
        self, summary_date: date, records: list[HeartRateRecord]
    ) -> DailyHeartSummary:
        """Create summary from a day's records."""
        # Separate by type
        hr_records = [r for r in records if r.metric_type == HeartMetricType.HEART_RATE]
        hrv_records = [r for r in records if r.metric_type == HeartMetricType.HRV_SDNN]

        # Calculate metrics
        avg_resting = self._calculate_resting_hr(hr_records)
        min_hr, max_hr = self._find_hr_range(hr_records)
        avg_hrv = self._calculate_avg_hrv(hrv_records)
        min_hrv = self._calculate_min_hrv(hrv_records)
        high_episodes = self._count_high_hr_episodes(hr_records)
        low_episodes = self._count_low_hr_episodes(hr_records)
        morning_hr, evening_hr = self._calculate_circadian_markers(hr_records)
        circadian_range = (
            abs(evening_hr - morning_hr) if morning_hr and evening_hr else 0.0
        )

        return DailyHeartSummary(
            date=summary_date,
            avg_resting_hr=avg_resting,
            min_hr=min_hr,
            max_hr=max_hr,
            avg_hrv_sdnn=avg_hrv,
            min_hrv_sdnn=min_hrv,
            hr_measurements=len(hr_records),
            hrv_measurements=len(hrv_records),
            high_hr_episodes=high_episodes,
            low_hr_episodes=low_episodes,
            circadian_hr_range=circadian_range,
            morning_hr=morning_hr or 0.0,
            evening_hr=evening_hr or 0.0,
        )

    def _calculate_resting_hr(self, hr_records: list[HeartRateRecord]) -> float:
        """Calculate average resting heart rate (sedentary only)."""
        resting_records = [
            r
            for r in hr_records
            if r.motion_context in [MotionContext.SEDENTARY, MotionContext.UNKNOWN]
        ]

        if not resting_records:
            return 0.0

        return sum(r.value for r in resting_records) / len(resting_records)

    def _find_hr_range(self, hr_records: list[HeartRateRecord]) -> tuple[float, float]:
        """Find min and max heart rate."""
        if not hr_records:
            return 0.0, 0.0

        values = [r.value for r in hr_records]
        return min(values), max(values)

    def _calculate_avg_hrv(self, hrv_records: list[HeartRateRecord]) -> float:
        """Calculate average HRV SDNN."""
        if not hrv_records:
            return 0.0

        return sum(r.value for r in hrv_records) / len(hrv_records)

    def _count_high_hr_episodes(self, hr_records: list[HeartRateRecord]) -> int:
        """Count episodes of high heart rate."""
        return sum(1 for r in hr_records if r.is_high_heart_rate)

    def _count_low_hr_episodes(self, hr_records: list[HeartRateRecord]) -> int:
        """Count episodes of low heart rate."""
        return sum(1 for r in hr_records if r.is_low_heart_rate)

    def _calculate_circadian_markers(
        self, hr_records: list[HeartRateRecord]
    ) -> tuple[float | None, float | None]:
        """Calculate morning and evening heart rate averages."""
        # Morning: 6-9 AM
        morning_records = [
            r
            for r in hr_records
            if 6 <= r.timestamp.hour < 9 and r.motion_context != MotionContext.ACTIVE
        ]

        # Evening: 6-10 PM
        evening_records = [
            r
            for r in hr_records
            if 18 <= r.timestamp.hour < 22 and r.motion_context != MotionContext.ACTIVE
        ]

        morning_hr = None
        if morning_records:
            morning_hr = sum(r.value for r in morning_records) / len(morning_records)

        evening_hr = None
        if evening_records:
            evening_hr = sum(r.value for r in evening_records) / len(evening_records)

        return morning_hr, evening_hr
