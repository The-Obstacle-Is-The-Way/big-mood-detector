"""
Sleep Feature Calculator Service

Calculates sleep-specific features from daily sleep summaries.
Extracted from AdvancedFeatureEngineer following Single Responsibility Principle.

Design Patterns:
- Strategy Pattern: Different calculation strategies can be implemented
- Value Objects: Immutable feature results
- Pure Functions: All calculations are side-effect free
"""


import numpy as np

from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


class SleepFeatureCalculator:
    """
    Calculates sleep-specific features for mood prediction.

    This service focuses solely on sleep feature calculations,
    extracted from the monolithic AdvancedFeatureEngineer.
    """

    def calculate_regularity_index(self, sleep_summaries: list[DailySleepSummary]) -> float:
        """
        Calculate sleep regularity index (0-100).

        Higher values indicate more regular sleep schedule.
        Based on Seoul National Study methodology.

        Args:
            sleep_summaries: List of daily sleep summaries

        Returns:
            Sleep regularity index from 0 to 100
        """
        if not sleep_summaries:
            return 0.0

        if len(sleep_summaries) == 1:
            # Perfect regularity with single data point
            return 100.0

        # Extract sleep and wake times in hours
        sleep_times = []
        wake_times = []

        for summary in sleep_summaries:
            if summary.earliest_bedtime is None or summary.latest_wake_time is None:
                continue

            # Convert to hours since midnight
            sleep_hour = summary.earliest_bedtime.hour + summary.earliest_bedtime.minute / 60

            # Handle late night times (after midnight)
            if sleep_hour < 12:  # Assume times before noon are late night
                sleep_hour += 24

            sleep_times.append(sleep_hour)

            # Wake times
            wake_hour = summary.latest_wake_time.hour + summary.latest_wake_time.minute / 60
            wake_times.append(wake_hour)

        # Calculate standard deviations
        sleep_std = np.std(sleep_times) if len(sleep_times) > 1 else 0
        wake_std = np.std(wake_times) if len(wake_times) > 1 else 0

        # Regularity index: 100 - (combined variance * scaling factor)
        regularity_raw = 100 - (sleep_std + wake_std) * 10

        # Clamp to 0-100 range
        return float(max(0.0, min(100.0, regularity_raw)))

    def calculate_interdaily_stability(self, sleep_summaries: list[DailySleepSummary]) -> float:
        """
        Calculate interdaily stability (IS) - consistency across days.

        Range 0-1, higher values indicate more stable circadian rhythm.
        Uses non-parametric circadian rhythm analysis.

        Args:
            sleep_summaries: List of daily sleep summaries

        Returns:
            Interdaily stability value from 0 to 1
        """
        if not sleep_summaries or len(sleep_summaries) < 3:
            return 0.0

        # Create activity-like pattern from sleep data
        # 1 during sleep, 0 during wake
        hourly_pattern = []

        for summary in sleep_summaries:
            day_pattern = [0] * 24  # 24 hours

            # Mark sleep hours as active (1) using bedtime and wake time
            if summary.earliest_bedtime and summary.latest_wake_time:
                sleep_start = summary.earliest_bedtime.hour
                sleep_end = summary.latest_wake_time.hour

                if sleep_start > sleep_end:  # Crosses midnight
                    for h in range(sleep_start, 24):
                        day_pattern[h] = 1
                    for h in range(0, sleep_end):
                        day_pattern[h] = 1
                else:
                    for h in range(sleep_start, sleep_end):
                        day_pattern[h] = 1

            hourly_pattern.extend(day_pattern)

        # Calculate IS using variance ratio
        n = len(hourly_pattern)
        if n < 24:
            return 0.0

        # Average across same hours of different days
        p = 24  # Period (24 hours)
        n_days = n // p

        hourly_means = []
        for hour in range(p):
            hour_values = [hourly_pattern[day * p + hour] for day in range(n_days)]
            hourly_means.append(np.mean(hour_values))

        # Interdaily stability calculation
        grand_mean = np.mean(hourly_pattern)

        # Between-hour variance
        between_var = n_days * np.sum([(m - grand_mean) ** 2 for m in hourly_means])

        # Total variance
        total_var = np.sum([(x - grand_mean) ** 2 for x in hourly_pattern])

        if total_var == 0:
            return 1.0  # Perfect stability

        is_value = between_var / total_var

        return float(min(1.0, max(0.0, is_value)))

    def calculate_intradaily_variability(self, sleep_summaries: list[DailySleepSummary]) -> float:
        """
        Calculate intradaily variability (IV) - fragmentation within days.

        Range 0-2, higher values indicate more fragmented rhythm.

        Args:
            sleep_summaries: List of daily sleep summaries

        Returns:
            Intradaily variability value from 0 to 2
        """
        if not sleep_summaries:
            return 0.0

        # Use fragmentation index and efficiency as indicators
        fragmentation_scores = []

        for summary in sleep_summaries:
            # Use direct fragmentation index
            frag_score = summary.sleep_fragmentation_index

            # Lower efficiency = more fragmentation
            efficiency_score = 1.0 - summary.sleep_efficiency

            # Combined fragmentation
            fragmentation = (frag_score + efficiency_score) / 2
            fragmentation_scores.append(fragmentation)

        # Average fragmentation scaled to 0-2 range
        avg_fragmentation = np.mean(fragmentation_scores)

        return float(avg_fragmentation * 2.0)

    def calculate_relative_amplitude(self, sleep_summaries: list[DailySleepSummary]) -> float:
        """
        Calculate relative amplitude (RA) - strength of rhythm.

        Range 0-1, higher values indicate stronger circadian rhythm.

        Args:
            sleep_summaries: List of daily sleep summaries

        Returns:
            Relative amplitude value from 0 to 1
        """
        if not sleep_summaries:
            return 0.0

        # Use sleep duration and efficiency as rhythm strength indicators
        sleep_durations = [s.total_sleep_hours for s in sleep_summaries]
        sleep_efficiencies = [s.sleep_efficiency for s in sleep_summaries]

        # Strong rhythm = consistent duration and high efficiency
        duration_consistency = 1.0 - (np.std(sleep_durations) / (np.mean(sleep_durations) + 1e-6))
        avg_efficiency = np.mean(sleep_efficiencies)

        # Combine metrics
        ra_value = (duration_consistency + avg_efficiency) / 2

        return float(max(0.0, min(1.0, ra_value)))

    def calculate_sleep_window_percentages(
        self, sleep_summaries: list[DailySleepSummary]
    ) -> tuple[float, float]:
        """
        Calculate percentages of short (<6h) and long (>10h) sleep windows.

        Args:
            sleep_summaries: List of daily sleep summaries

        Returns:
            Tuple of (short_sleep_pct, long_sleep_pct)
        """
        if not sleep_summaries:
            return 0.0, 0.0

        durations = [s.total_sleep_hours for s in sleep_summaries]
        total_days = len(durations)

        short_days = sum(1 for d in durations if d < 6)
        long_days = sum(1 for d in durations if d > 10)

        short_pct = (short_days / total_days) * 100
        long_pct = (long_days / total_days) * 100

        return float(short_pct), float(long_pct)

    def calculate_timing_variances(
        self, sleep_summaries: list[DailySleepSummary]
    ) -> tuple[float, float]:
        """
        Calculate variance in sleep onset and wake times.

        Args:
            sleep_summaries: List of daily sleep summaries

        Returns:
            Tuple of (onset_variance, wake_variance) in hours²
        """
        if not sleep_summaries or len(sleep_summaries) < 2:
            return 0.0, 0.0

        sleep_times = []
        wake_times = []

        for summary in sleep_summaries:
            if summary.earliest_bedtime is None or summary.latest_wake_time is None:
                continue

            # Convert to hours since midnight
            sleep_hour = summary.earliest_bedtime.hour + summary.earliest_bedtime.minute / 60

            # Handle late night times
            if sleep_hour < 12:
                sleep_hour += 24

            sleep_times.append(sleep_hour)

            wake_hour = summary.latest_wake_time.hour + summary.latest_wake_time.minute / 60
            wake_times.append(wake_hour)

        onset_variance = float(np.var(sleep_times))
        wake_variance = float(np.var(wake_times))

        return onset_variance, wake_variance
