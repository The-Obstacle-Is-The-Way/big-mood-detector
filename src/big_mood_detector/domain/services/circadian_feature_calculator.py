"""
Circadian Feature Calculator Service

Calculates circadian rhythm features from sleep and activity data.
Critical for detecting phase shifts in bipolar disorder.

Design Patterns:
- Strategy Pattern: Different calculation methods can be swapped
- Value Objects: Immutable result objects
- Pure Functions: All calculations are side-effect free
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary
from big_mood_detector.domain.utils.math_helpers import clamp


@dataclass(frozen=True)
class L5M10Result:
    """Least active 5 hours and most active 10 hours metrics."""

    l5_value: float  # Activity level during least active 5 hours
    m10_value: float  # Activity level during most active 10 hours
    l5_onset: datetime | None  # When L5 period starts
    m10_onset: datetime | None  # When M10 period starts


@dataclass(frozen=True)
class PhaseShiftResult:
    """Circadian phase shift detection result."""

    phase_advance_hours: float  # Hours earlier than normal
    phase_delay_hours: float  # Hours later than normal
    phase_type: str  # "normal", "advanced", "delayed"


class CircadianFeatureCalculator:
    """
    Calculates circadian rhythm features for mood disorder detection.

    This service focuses on circadian disruption patterns,
    a key biomarker in bipolar disorder.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize with optional configuration.

        Args:
            config: Configuration dict with circadian section
        """
        if config is None:
            # Default population norms
            self.population_sleep_time = 23.0  # 11 PM
            self.population_wake_time = 7.0  # 7 AM
            self.phase_threshold = 2.0  # 2 hour deviation
        else:
            circadian_config = config.get("circadian", {})
            self.population_sleep_time = circadian_config.get(
                "population_sleep_time", 23.0
            )
            self.population_wake_time = circadian_config.get(
                "population_wake_time", 7.0
            )
            self.phase_threshold = circadian_config.get("phase_advance_threshold", 2.0)

    def calculate_l5_m10_metrics(
        self, activity_summaries: list[DailyActivitySummary]
    ) -> L5M10Result:
        """
        Calculate L5 (least active 5 hours) and M10 (most active 10 hours).

        These metrics indicate circadian rhythm strength and timing.

        Args:
            activity_summaries: Daily activity summaries

        Returns:
            L5M10Result with activity metrics
        """
        if not activity_summaries:
            return L5M10Result(l5_value=0, m10_value=0, l5_onset=None, m10_onset=None)

        # For daily summaries, we approximate from available data
        # In production, this would use hourly activity data

        # Estimate L5 from sedentary hours (lower is better for L5)
        l5_values = []
        for summary in activity_summaries:
            # L5 represents activity level during least active period
            # Use inverse of sedentary time as activity metric
            l5_estimate = (24 - summary.sedentary_hours) * 10  # Activity units
            l5_values.append(l5_estimate)

        # Estimate M10 from active hours
        m10_values = []
        for summary in activity_summaries:
            # M10 represents activity level during most active period
            m10_estimate = (
                summary.active_hours * 60 + summary.total_active_energy / 5
            )  # Activity units
            m10_values.append(m10_estimate)

        l5_value = float(np.mean(l5_values)) if l5_values else 0
        m10_value = float(np.mean(m10_values)) if m10_values else 0

        # Estimate timing based on typical patterns
        # L5 typically occurs during sleep (2-7 AM)
        # M10 typically occurs during day (10 AM - 8 PM)
        now = datetime.now()
        l5_onset = now.replace(hour=2, minute=0, second=0, microsecond=0)
        m10_onset = now.replace(hour=10, minute=0, second=0, microsecond=0)

        return L5M10Result(
            l5_value=l5_value,
            m10_value=m10_value,
            l5_onset=l5_onset,
            m10_onset=m10_onset,
        )

    def calculate_phase_shifts(
        self, sleep_summaries: list[DailySleepSummary]
    ) -> PhaseShiftResult:
        """
        Calculate circadian phase shifts from sleep timing.

        Args:
            sleep_summaries: Daily sleep summaries

        Returns:
            Phase shift detection result
        """
        if not sleep_summaries:
            return PhaseShiftResult(
                phase_advance_hours=0.0, phase_delay_hours=0.0, phase_type="unknown"
            )

        # Calculate average sleep timing
        sleep_times = []
        wake_times = []

        for summary in sleep_summaries:
            if summary.earliest_bedtime and summary.latest_wake_time:
                # Convert to hours since midnight
                sleep_hour = (
                    summary.earliest_bedtime.hour + summary.earliest_bedtime.minute / 60
                )

                # Handle late night times
                if sleep_hour < 12:  # After midnight
                    sleep_hour += 24

                sleep_times.append(sleep_hour)

                wake_hour = (
                    summary.latest_wake_time.hour + summary.latest_wake_time.minute / 60
                )
                wake_times.append(wake_hour)

        if not sleep_times:
            return PhaseShiftResult(
                phase_advance_hours=0.0, phase_delay_hours=0.0, phase_type="unknown"
            )

        avg_sleep_time = np.mean(sleep_times)
        avg_wake_time = np.mean(wake_times)

        # Compare to population norms
        sleep_shift = avg_sleep_time - self.population_sleep_time
        wake_shift = avg_wake_time - self.population_wake_time

        # Average the shifts
        total_shift = (sleep_shift + wake_shift) / 2

        # Determine phase type
        if total_shift > self.phase_threshold:
            phase_type = "delayed"
            phase_delay = float(total_shift)
            phase_advance = 0.0
        elif total_shift < -self.phase_threshold:
            phase_type = "advanced"
            phase_advance = float(-total_shift)
            phase_delay = 0.0
        else:
            phase_type = "normal"
            phase_advance = 0.0
            phase_delay = 0.0

        return PhaseShiftResult(
            phase_advance_hours=float(phase_advance),
            phase_delay_hours=float(phase_delay),
            phase_type=phase_type,
        )

    def estimate_dlmo(
        self, sleep_summaries: list[DailySleepSummary]
    ) -> datetime | None:
        """
        Estimate Dim Light Melatonin Onset from sleep patterns.

        DLMO typically occurs ~2 hours before habitual sleep onset.

        Args:
            sleep_summaries: Daily sleep summaries

        Returns:
            Estimated DLMO time or None
        """
        if not sleep_summaries:
            return None

        # Calculate average sleep onset time
        sleep_times = []

        for summary in sleep_summaries:
            if summary.earliest_bedtime:
                sleep_hour = (
                    summary.earliest_bedtime.hour + summary.earliest_bedtime.minute / 60
                )

                # Handle late night times
                if sleep_hour < 12:
                    sleep_hour += 24

                sleep_times.append(sleep_hour)

        if not sleep_times:
            return None

        avg_sleep_time = np.mean(sleep_times)

        # DLMO is approximately 2 hours before sleep onset
        dlmo_hour = (avg_sleep_time - 2) % 24

        # Create datetime for today with estimated DLMO time
        now = datetime.now()
        dlmo_time = now.replace(
            hour=int(dlmo_hour),
            minute=int((dlmo_hour % 1) * 60),
            second=0,
            microsecond=0,
        )

        return dlmo_time

    def estimate_core_temp_nadir(
        self, sleep_summaries: list[DailySleepSummary]
    ) -> datetime | None:
        """
        Estimate core body temperature nadir from sleep patterns.

        Temperature nadir typically occurs ~2 hours before habitual wake time.

        Args:
            sleep_summaries: Daily sleep summaries

        Returns:
            Estimated temperature nadir time or None
        """
        if not sleep_summaries:
            return None

        # Calculate average wake time
        wake_times = []

        for summary in sleep_summaries:
            if summary.latest_wake_time:
                wake_hour = (
                    summary.latest_wake_time.hour + summary.latest_wake_time.minute / 60
                )
                wake_times.append(wake_hour)

        if not wake_times:
            return None

        avg_wake_time = np.mean(wake_times)

        # Nadir is approximately 2 hours before wake time
        nadir_hour = (avg_wake_time - 2) % 24

        # Create datetime for today with estimated nadir time
        now = datetime.now()
        nadir_time = now.replace(
            hour=int(nadir_hour),
            minute=int((nadir_hour % 1) * 60),
            second=0,
            microsecond=0,
        )

        return nadir_time

    def calculate_circadian_amplitude(
        self,
        activity_summaries: list[DailyActivitySummary],
        sleep_summaries: list[DailySleepSummary],
    ) -> float:
        """
        Calculate circadian rhythm amplitude (strength).

        Higher amplitude indicates stronger, more regular rhythms.

        Args:
            activity_summaries: Daily activity data
            sleep_summaries: Daily sleep data

        Returns:
            Amplitude score (0-1)
        """
        if not activity_summaries or not sleep_summaries:
            return 0.0

        # Calculate L5/M10 ratio as one indicator
        l5_m10 = self.calculate_l5_m10_metrics(activity_summaries)

        if l5_m10.m10_value > 0:
            activity_ratio = 1 - (l5_m10.l5_value / l5_m10.m10_value)
        else:
            activity_ratio = 0.0

        # Calculate sleep regularity as another indicator
        sleep_efficiencies = [s.sleep_efficiency for s in sleep_summaries]
        sleep_consistency = np.mean(sleep_efficiencies) if sleep_efficiencies else 0.0

        # Combine metrics
        amplitude = (activity_ratio + sleep_consistency) / 2

        return clamp(float(amplitude), 0.0, 1.0)

    def calculate_phase_angle(
        self,
        sleep_summaries: list[DailySleepSummary],
        activity_summaries: list[DailyActivitySummary],
    ) -> float:
        """
        Calculate phase angle between sleep and activity rhythms.

        Misalignment indicates circadian disruption.

        Args:
            sleep_summaries: Daily sleep data
            activity_summaries: Daily activity data

        Returns:
            Phase angle in hours (positive = activity leads sleep)
        """
        if not sleep_summaries or not activity_summaries:
            return 0.0

        # Calculate mid-sleep time
        mid_sleep_times = []
        for sleep in sleep_summaries:
            if sleep.mid_sleep_time:
                hour = sleep.mid_sleep_time.hour + sleep.mid_sleep_time.minute / 60
                mid_sleep_times.append(hour)

        # Calculate peak activity time
        peak_activity_times = []
        for activity in activity_summaries:
            if activity.peak_activity_hour is not None:
                peak_activity_times.append(float(activity.peak_activity_hour))

        if not mid_sleep_times or not peak_activity_times:
            return 0.0

        avg_mid_sleep = np.mean(mid_sleep_times)
        avg_peak_activity = np.mean(peak_activity_times)

        # Calculate phase difference
        # Expected: peak activity ~8-10 hours after mid-sleep
        expected_offset = 9.0  # hours
        actual_offset = (avg_peak_activity - avg_mid_sleep) % 24

        # Calculate deviation from expected
        phase_angle = actual_offset - expected_offset

        # Normalize to -12 to +12 range
        if phase_angle > 12:
            phase_angle -= 24
        elif phase_angle < -12:
            phase_angle += 24

        return float(phase_angle)
