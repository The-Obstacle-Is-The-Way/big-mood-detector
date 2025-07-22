"""
Activity Feature Calculator Service

Calculates activity-specific features from daily activity summaries.
Extracted from AdvancedFeatureEngineer following Single Responsibility Principle.

Design Patterns:
- Strategy Pattern: Different calculation strategies for activity patterns
- Value Objects: Immutable result objects
- Pure Functions: All calculations are side-effect free
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.utils.math_helpers import clamp, safe_std, safe_var


@dataclass(frozen=True)
class ActivityIntensityMetrics:
    """Activity intensity distribution metrics."""

    intensity_ratio: float  # High/low activity ratio
    high_intensity_days: int
    low_intensity_days: int
    moderate_intensity_days: int


@dataclass(frozen=True)
class ActivityAnomalies:
    """Detected activity anomalies."""

    has_hyperactivity: bool
    has_hypoactivity: bool
    has_irregular_timing: bool
    anomaly_days: list[date]


class ActivityFeatureCalculator:
    """
    Calculates activity-specific features for mood prediction.

    This service focuses on physical activity patterns that indicate
    manic/depressive episodes in bipolar disorder.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize with optional configuration.

        Args:
            config: Configuration dict with activity thresholds
        """
        if config is None:
            # Default thresholds
            self.high_activity_threshold = 10000  # steps
            self.low_activity_threshold = 5000  # steps
            self.hyperactivity_threshold = 15000  # steps
            self.hypoactivity_threshold = 2000  # steps
            self.sedentary_threshold_hours = 18  # hours
            self.fragmentation_window = 3  # days for rolling variance
        else:
            activity_config = config.get("activity", {})
            self.high_activity_threshold = activity_config.get("high_threshold", 10000)
            self.low_activity_threshold = activity_config.get("low_threshold", 5000)
            self.hyperactivity_threshold = activity_config.get(
                "hyperactivity_threshold", 15000
            )
            self.hypoactivity_threshold = activity_config.get(
                "hypoactivity_threshold", 2000
            )
            self.sedentary_threshold_hours = activity_config.get(
                "sedentary_threshold_hours", 18
            )
            self.fragmentation_window = activity_config.get("fragmentation_window", 3)

    def calculate_activity_fragmentation(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> float:
        """
        Calculate activity fragmentation (0-1).

        High fragmentation indicates erratic patterns common in mood episodes.

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            Fragmentation score from 0 to 1
        """
        if not activity_summaries or len(activity_summaries) < 2:
            return 0.0

        # Multiple fragmentation indicators
        indicators = []

        # 1. Day-to-day step changes
        step_counts = [a.total_steps for a in activity_summaries]
        daily_changes = []

        for i in range(1, len(step_counts)):
            if step_counts[i - 1] > 0:
                change_ratio = (
                    abs(step_counts[i] - step_counts[i - 1]) / step_counts[i - 1]
                )
                daily_changes.append(change_ratio)

        if daily_changes:
            # High changes indicate fragmentation
            avg_change = np.mean(daily_changes)
            indicators.append(
                clamp(float(avg_change * 2), 0.0, 1.0)
            )  # Scale appropriately

        # 2. Activity variance from summaries
        activity_variances = [a.activity_variance for a in activity_summaries]
        if activity_variances:
            avg_variance = np.mean(activity_variances)
            indicators.append(float(avg_variance))

        # 3. Coefficient of variation in steps
        if len(step_counts) > 1:
            step_std = safe_std(step_counts)
            step_mean = np.mean(step_counts)
            if step_mean > 0:
                cv = step_std / step_mean
                indicators.append(clamp(float(cv), 0.0, 1.0))

        # 4. Energy expenditure variability
        energy_values = [a.total_active_energy for a in activity_summaries]
        if len(energy_values) > 1:
            energy_std = safe_std(energy_values)
            energy_mean = np.mean(energy_values)
            if energy_mean > 0:
                energy_cv = energy_std / energy_mean
                indicators.append(clamp(float(energy_cv), 0.0, 1.0))

        if not indicators:
            return 0.0

        # Average all fragmentation indicators
        return float(np.mean(indicators))

    def calculate_sedentary_bouts(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> tuple[float, float, int]:
        """
        Calculate sedentary bout statistics.

        Long sedentary bouts indicate depressive patterns.

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            Tuple of (mean_bout_minutes, max_bout_minutes, longest_streak_days)
        """
        if not activity_summaries:
            return 0.0, 0.0, 0

        sedentary_hours = [a.sedentary_hours for a in activity_summaries]

        # Convert to minutes
        sedentary_minutes = [h * 60 for h in sedentary_hours]

        mean_bout = float(np.mean(sedentary_minutes))
        max_bout = float(max(sedentary_minutes))

        # Calculate longest streak of high sedentary days
        streak = 0
        max_streak = 0

        for hours in sedentary_hours:
            if hours >= self.sedentary_threshold_hours:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        return mean_bout, max_bout, max_streak

    def calculate_activity_intensity_metrics(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> ActivityIntensityMetrics:
        """
        Calculate activity intensity distribution.

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            Activity intensity metrics
        """
        if not activity_summaries:
            return ActivityIntensityMetrics(
                intensity_ratio=0.0,
                high_intensity_days=0,
                low_intensity_days=0,
                moderate_intensity_days=0,
            )

        high_days = 0
        low_days = 0
        moderate_days = 0

        for summary in activity_summaries:
            steps = summary.total_steps

            if steps >= self.high_activity_threshold:
                high_days += 1
            elif steps <= self.low_activity_threshold:
                low_days += 1
            else:
                moderate_days += 1

        # Calculate intensity ratio
        if low_days > 0:
            intensity_ratio = high_days / low_days
        elif high_days > 0:
            intensity_ratio = float(high_days)  # All high intensity
        else:
            intensity_ratio = 1.0  # Neutral

        return ActivityIntensityMetrics(
            intensity_ratio=float(intensity_ratio),
            high_intensity_days=high_days,
            low_intensity_days=low_days,
            moderate_intensity_days=moderate_days,
        )

    def calculate_activity_rhythm_strength(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> float:
        """
        Calculate strength of activity rhythm (0-1).

        Strong rhythms indicate healthy patterns.

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            Rhythm strength from 0 to 1
        """
        if not activity_summaries:
            return 0.0

        # Use multiple indicators
        indicators = []

        # 1. Consistency in peak activity hour
        peak_hours = [
            float(a.peak_activity_hour)
            for a in activity_summaries
            if a.peak_activity_hour is not None
        ]
        if peak_hours:
            hour_variance = safe_var(peak_hours)
            # Low variance = strong rhythm
            hour_consistency = 1.0 / (
                1.0 + hour_variance / 12.0
            )  # Normalize by half day
            indicators.append(hour_consistency)

        # 2. Consistency in active/sedentary balance
        active_ratios = []
        for summary in activity_summaries:
            total_hours = summary.active_hours + summary.sedentary_hours
            if total_hours > 0:
                active_ratio = summary.active_hours / total_hours
                active_ratios.append(active_ratio)

        if active_ratios:
            ratio_std = safe_std(active_ratios)
            ratio_consistency = 1.0 - clamp(ratio_std * 2, 0.0, 1.0)
            indicators.append(ratio_consistency)

        # 3. Step count regularity
        step_counts = [a.total_steps for a in activity_summaries]
        if step_counts and len(step_counts) > 1:
            step_cv = safe_std(step_counts) / (np.mean(step_counts) + 1)
            step_regularity = 1.0 / (1.0 + step_cv)
            indicators.append(float(step_regularity))

        if not indicators:
            return 0.0

        # Average all indicators
        return float(np.mean(indicators))

    def calculate_activity_timing_consistency(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> tuple[float, float]:
        """
        Calculate consistency of activity start/end times.

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            Tuple of (onset_consistency, offset_consistency) from 0 to 1
        """
        if not activity_summaries:
            return 0.0, 0.0

        onset_hours = []
        offset_hours = []

        for summary in activity_summaries:
            if summary.earliest_activity:
                onset_hour = (
                    summary.earliest_activity.hour
                    + summary.earliest_activity.minute / 60
                )
                onset_hours.append(onset_hour)

            if summary.latest_activity:
                offset_hour = (
                    summary.latest_activity.hour + summary.latest_activity.minute / 60
                )
                offset_hours.append(offset_hour)

        # Calculate consistency (inverse of variance)
        onset_consistency = 0.0
        if onset_hours and len(onset_hours) > 1:
            onset_var = safe_var(onset_hours)
            onset_consistency = 1.0 / (
                1.0 + onset_var / 6.0
            )  # Normalize by quarter day

        offset_consistency = 0.0
        if offset_hours and len(offset_hours) > 1:
            offset_var = safe_var(offset_hours)
            offset_consistency = 1.0 / (1.0 + offset_var / 6.0)

        return float(onset_consistency), float(offset_consistency)

    def detect_activity_anomalies(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> ActivityAnomalies:
        """
        Detect anomalous activity patterns.

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            Detected anomalies
        """
        if not activity_summaries:
            return ActivityAnomalies(
                has_hyperactivity=False,
                has_hypoactivity=False,
                has_irregular_timing=False,
                anomaly_days=[],
            )

        hyperactive_days = []
        hypoactive_days = []
        irregular_days = []

        # Calculate baseline for irregularity detection
        step_counts = [a.total_steps for a in activity_summaries]
        if len(step_counts) > 3:
            median_steps = np.median(step_counts)
            mad = np.median(
                [abs(s - median_steps) for s in step_counts]
            )  # Median absolute deviation
            threshold_high = median_steps + 3 * mad
            threshold_low = median_steps - 3 * mad
        else:
            threshold_high = float("inf")
            threshold_low = 0

        for summary in activity_summaries:
            # Hyperactivity detection
            if (
                summary.total_steps > self.hyperactivity_threshold
                or summary.active_hours > 16
                or summary.is_high_activity
            ):
                hyperactive_days.append(summary.date)

            # Hypoactivity detection
            if (
                summary.total_steps < self.hypoactivity_threshold
                or summary.sedentary_hours > 20
                or summary.is_low_activity
            ):
                hypoactive_days.append(summary.date)

            # Irregular timing detection
            if (
                summary.activity_variance > 0.7
                or summary.total_steps > threshold_high
                or summary.total_steps < threshold_low
            ):
                irregular_days.append(summary.date)

        # Combine all anomaly days
        all_anomaly_days = list(
            set(hyperactive_days + hypoactive_days + irregular_days)
        )
        all_anomaly_days.sort()

        return ActivityAnomalies(
            has_hyperactivity=len(hyperactive_days) > 0,
            has_hypoactivity=len(hypoactive_days) > 0,
            has_irregular_timing=len(irregular_days) > 0,
            anomaly_days=all_anomaly_days,
        )

    def calculate_step_acceleration(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> float:
        """
        Calculate step count acceleration/deceleration.

        Positive values indicate increasing activity (potential mania onset).
        Negative values indicate decreasing activity (potential depression onset).

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            Acceleration in steps per day
        """
        if not activity_summaries or len(activity_summaries) < 3:
            return 0.0

        # Get step counts and days
        data_points = [(i, a.total_steps) for i, a in enumerate(activity_summaries)]

        if len(data_points) < 2:
            return 0.0

        # Calculate linear regression slope
        x_values = [p[0] for p in data_points]
        y_values = [p[1] for p in data_points]

        # Simple linear regression
        x_mean = np.mean(x_values)
        y_mean = np.mean(y_values)

        numerator = sum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(x_values, y_values, strict=False)
        )
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Slope represents change in steps per day
        return float(slope)
