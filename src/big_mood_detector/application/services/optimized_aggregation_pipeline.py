"""
Optimized Aggregation Pipeline with O(n+m) performance.

This implements pre-indexing to avoid scanning all records for each day.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationConfig,
    AggregationPipeline,
)
from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord


@dataclass
class OptimizationConfig:
    """Configuration for optimization thresholds."""

    # Use optimization if we have more than this many days
    optimization_threshold_days: int = 7
    # Use optimization if we have more than this many records
    optimization_threshold_records: int = 1000


class OptimizedAggregationPipeline(AggregationPipeline):
    """
    Aggregation pipeline with O(n+m) performance optimization.

    Pre-indexes records by date to avoid repeated full scans.
    """

    def __init__(
        self,
        config: AggregationConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize with optimization configuration."""
        super().__init__(config=config, **kwargs)
        self.optimization_config = optimization_config or OptimizationConfig()

    def aggregate_daily_features(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
        start_date: date,
        end_date: date,
        min_window_size: int | None = None,
        parallel: bool = False,
    ) -> list[Any]:
        """
        Aggregate features with O(n+m) optimization.

        Pre-indexes records by date before processing.
        """
        # Check if we should use optimization
        num_days = (end_date - start_date).days + 1
        total_records = len(sleep_records) + len(activity_records) + len(heart_records)

        should_optimize = (
            num_days >= self.optimization_config.optimization_threshold_days
            or total_records >= self.optimization_config.optimization_threshold_records
        )

        if not should_optimize:
            # Use base implementation for small datasets
            return super().aggregate_daily_features(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                start_date=start_date,
                end_date=end_date,
                min_window_size=min_window_size,
                parallel=parallel,
            )

        # Pre-index all records by date
        sleep_by_date = self._index_sleep_by_date(sleep_records)
        activity_by_date = self._index_activity_by_date(activity_records)
        heart_by_date = self._index_heart_by_date(heart_records)

        # Process each day using indexed lookups
        daily_features = []
        current_date = start_date

        # Keep rolling windows for statistics
        sleep_metrics_window: list[dict[str, float]] = []
        circadian_metrics_window: list[dict[str, float]] = []

        window_size = min_window_size or self.config.min_window_size

        while current_date <= end_date:
            # 1. Sleep Window Analysis - O(1) lookup
            day_sleep = sleep_by_date.get(current_date, [])

            sleep_windows = self.sleep_analyzer.analyze_sleep_episodes(
                day_sleep, current_date
            )

            # 2. Activity Sequence Extraction - O(1) lookup
            day_activity = activity_by_date.get(current_date, [])

            activity_sequence = None
            if day_activity:
                activity_sequence = self.activity_extractor.extract_daily_sequence(
                    day_activity, current_date
                )

            # 3. Activity Metrics - O(1) lookup instead of O(n) scan
            activity_metrics = self._calculate_activity_metrics_optimized(
                day_activity
            )

            # 4. Heart Rate Metrics - O(1) lookup
            day_heart = heart_by_date.get(current_date, [])
            heart_metrics = self._calculate_heart_rate_metrics_optimized(
                day_heart, current_date
            )

            # 5. Circadian Rhythm Analysis (optional - expensive)
            circadian_metrics = None
            if self.config.enable_circadian_analysis:
                circadian_metrics = self._calculate_circadian_metrics_optimized(
                    activity_by_date, current_date
                )

            # 6. DLMO Calculation (optional - very expensive)
            dlmo_result = None
            if self.config.enable_dlmo_calculation:
                dlmo_result = self._calculate_dlmo_optimized(
                    sleep_by_date, current_date
                )

            # Rest is the same as base implementation...
            daily_metrics = self.calculate_daily_metrics(
                sleep_windows, activity_sequence, circadian_metrics, dlmo_result
            )

            # Add activity metrics to daily_metrics
            if daily_metrics:
                daily_metrics["activity"] = activity_metrics

                # Fix sleep duration
                if "sleep" in daily_metrics:
                    accurate_hours = self._get_actual_sleep_duration(
                        sleep_records, current_date
                    )
                    daily_metrics["sleep"]["sleep_duration_hours"] = accurate_hours

            # Update rolling windows
            if daily_metrics:
                if "sleep" in daily_metrics:
                    sleep_metrics_window.append(daily_metrics["sleep"])
                    if len(sleep_metrics_window) > self.config.window_size:
                        sleep_metrics_window.pop(0)

                if "circadian" in daily_metrics and daily_metrics["circadian"]:
                    circadian_metrics_window.append(daily_metrics["circadian"])
                    if len(circadian_metrics_window) > self.config.window_size:
                        circadian_metrics_window.pop(0)

            # Calculate statistics if we have enough data
            if len(sleep_metrics_window) >= window_size:
                features = self._calculate_features_with_stats(
                    current_date,
                    daily_metrics,
                    sleep_metrics_window,
                    circadian_metrics_window,
                    activity_metrics,
                    heart_metrics,
                    sleep_records,
                )

                if features:
                    daily_features.append(features)

            current_date += timedelta(days=1)

        return daily_features

    def _index_sleep_by_date(
        self, sleep_records: list[SleepRecord]
    ) -> dict[date, list[SleepRecord]]:
        """
        Pre-index sleep records by date.

        Sleep records can span multiple days, so we index them
        for each day they overlap.
        """
        records_by_date = defaultdict(list)

        for record in sleep_records:
            # Get all dates this sleep record overlaps
            current = record.start_date.date()
            end = record.end_date.date()

            while current <= end:
                records_by_date[current].append(record)
                current += timedelta(days=1)

        return dict(records_by_date)

    def _index_activity_by_date(
        self, records: list[ActivityRecord]
    ) -> dict[date, list[ActivityRecord]]:
        """
        Pre-index activity records by their start date.
        """
        records_by_date = defaultdict(list)

        for record in records:
            record_date = record.start_date.date()
            records_by_date[record_date].append(record)

        return dict(records_by_date)

    def _index_heart_by_date(
        self, records: list[HeartRateRecord]
    ) -> dict[date, list[HeartRateRecord]]:
        """
        Pre-index heart rate records by their timestamp date.
        """
        records_by_date = defaultdict(list)

        for record in records:
            record_date = record.timestamp.date()
            records_by_date[record_date].append(record)

        return dict(records_by_date)

    def _calculate_activity_metrics_optimized(
        self, day_activity: list[ActivityRecord]
    ) -> dict[str, float]:
        """
        Calculate activity metrics from pre-filtered records.

        This receives only the records for the target date,
        avoiding the O(n) scan of all records.
        """
        if not day_activity:
            # Return defaults when no activity data
            return {
                "daily_steps": 0.0,
                "activity_variance": 0.0,
                "sedentary_hours": 24.0,
                "activity_fragmentation": 0.0,
                "sedentary_bout_mean": 24.0,
                "activity_intensity_ratio": 0.0,
            }

        # Calculate metrics using only the day's records
        import numpy as np

        from big_mood_detector.domain.entities.activity_record import ActivityType

        step_records = [
            r for r in day_activity if r.activity_type == ActivityType.STEP_COUNT
        ]
        total_steps = sum(r.value for r in step_records)

        # Calculate activity variance using hourly bins
        hourly_activity = [0.0] * 24
        for record in step_records:
            hour = record.start_date.hour
            hourly_activity[hour] += record.value / max(1, record.duration_hours)

        activity_variance = np.var(hourly_activity) if hourly_activity else 0.0

        # Calculate sedentary hours
        active_hours = sum(1 for h in hourly_activity if h >= 250)
        sedentary_hours = 24 - active_hours

        # Simple fragmentation
        transitions = 0
        for i in range(1, len(hourly_activity)):
            if (hourly_activity[i - 1] < 250) != (hourly_activity[i] < 250):
                transitions += 1
        activity_fragmentation = transitions / 23.0 if len(hourly_activity) > 1 else 0.0

        # Sedentary bout mean
        sedentary_bouts = []
        current_bout = 0
        for h in hourly_activity:
            if h < 250:
                current_bout += 1
            elif current_bout > 0:
                sedentary_bouts.append(current_bout)
                current_bout = 0
        if current_bout > 0:
            sedentary_bouts.append(current_bout)

        sedentary_bout_mean = (
            sum(sedentary_bouts) / len(sedentary_bouts) if sedentary_bouts else 24.0
        )

        # Activity intensity ratio
        high_activity_hours = sum(1 for h in hourly_activity if h >= 1000)
        activity_intensity_ratio = (
            high_activity_hours / max(1, active_hours) if active_hours > 0 else 0.0
        )

        return {
            "daily_steps": float(total_steps),
            "activity_variance": float(activity_variance),
            "sedentary_hours": float(sedentary_hours),
            "activity_fragmentation": float(activity_fragmentation),
            "sedentary_bout_mean": float(sedentary_bout_mean),
            "activity_intensity_ratio": float(activity_intensity_ratio),
        }

    def _calculate_heart_rate_metrics_optimized(
        self, day_heart: list[HeartRateRecord], target_date: date
    ) -> dict[str, float | None]:
        """
        Calculate heart rate metrics from pre-filtered records.

        This is already optimized in the base class, but we
        override to use pre-filtered records.
        """
        if not day_heart:
            return {
                "avg_resting_hr": None,
                "hrv_sdnn": None,
                "hr_circadian_range": 0.0,
                "hr_minimum_hour": 0.0,
            }

        # Use aggregator on just this day's records
        summaries = self.heart_rate_aggregator.aggregate_daily(day_heart)

        if target_date not in summaries:
            return {
                "avg_resting_hr": None,
                "hrv_sdnn": None,
                "hr_circadian_range": 0.0,
                "hr_minimum_hour": 0.0,
            }

        summary = summaries[target_date]

        # Find hour of minimum HR
        hr_minimum_hour = 0.0
        if day_heart:
            min_record = min(day_heart, key=lambda r: r.value)
            hr_minimum_hour = float(min_record.timestamp.hour)

        return {
            "avg_resting_hr": (
                summary.avg_resting_hr if summary.avg_resting_hr > 0 else None
            ),
            "hrv_sdnn": summary.avg_hrv_sdnn if summary.avg_hrv_sdnn > 0 else None,
            "hr_circadian_range": summary.circadian_hr_range,
            "hr_minimum_hour": hr_minimum_hour,
        }

    def _calculate_circadian_metrics_optimized(
        self, 
        activity_by_date: dict[date, list[ActivityRecord]], 
        target_date: date
    ) -> Any:
        """
        Calculate circadian metrics using pre-indexed data.
        
        This is O(k) where k is lookback days, not O(n*k).
        """
        sequences = []
        
        for days_back in range(self.config.lookback_days_circadian):
            seq_date = target_date - timedelta(days=days_back)
            # O(1) lookup instead of O(n) scan
            day_activity = activity_by_date.get(seq_date, [])
            
            if day_activity:
                seq = self.activity_extractor.extract_daily_sequence(
                    day_activity, seq_date
                )
                sequences.append(seq)
        
        # Calculate metrics if we have enough data
        if len(sequences) >= 3:
            return self.circadian_analyzer.calculate_metrics(sequences)
        
        return None

    def _calculate_dlmo_optimized(
        self,
        sleep_by_date: dict[date, list[SleepRecord]],
        target_date: date,
    ) -> Any:
        """
        Calculate DLMO using pre-indexed data.
        
        This is O(k) where k is lookback days, not O(n*k).
        """
        dlmo_sleep = []
        
        # Collect sleep records from the past 2 weeks
        for days_back in range(self.config.lookback_days_dlmo):
            lookup_date = target_date - timedelta(days=days_back)
            # O(1) lookup instead of O(n) scan
            day_sleep = sleep_by_date.get(lookup_date, [])
            dlmo_sleep.extend(day_sleep)
        
        # Calculate DLMO if we have enough data
        if len(dlmo_sleep) >= 3:
            return self.dlmo_calculator.calculate_dlmo(
                sleep_records=dlmo_sleep,
                target_date=target_date,
                days_to_model=min(7, len(dlmo_sleep)),
            )
        
        return None
