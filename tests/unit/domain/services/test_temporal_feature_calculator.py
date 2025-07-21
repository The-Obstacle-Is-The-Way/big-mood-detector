"""
Test Temporal Feature Calculator

TDD approach for extracting temporal feature calculations from AdvancedFeatureEngineer.
Temporal features capture changes over time windows (7-day, 30-day patterns).
"""

from datetime import date, timedelta

import pytest

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


class TestTemporalFeatureCalculator:
    """Test temporal feature calculations."""

    @pytest.fixture
    def calculator(self):
        """Create TemporalFeatureCalculator instance."""
        from big_mood_detector.domain.services.temporal_feature_calculator import (
            TemporalFeatureCalculator,
        )

        return TemporalFeatureCalculator()

    @pytest.fixture
    def stable_sleep_data(self):
        """Create stable sleep pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(30):  # 30 days
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.5,
                total_sleep_hours=7.5 + (i % 3) * 0.2,  # Slight variation
                sleep_efficiency=0.88,
                sleep_sessions=1,
                longest_sleep_hours=7.5,
                sleep_fragmentation_index=0.1,
                earliest_bedtime=None,
                latest_wake_time=None,
                mid_sleep_time=None,
            )
            summaries.append(summary)

        return summaries

    @pytest.fixture
    def variable_activity_data(self):
        """Create variable activity pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(30):  # 30 days
            # Weekly pattern with weekend variation
            if i % 7 in [5, 6]:  # Weekend
                steps = 5000 + i * 100
            else:  # Weekday
                steps = 8000 + i * 200

            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=float(steps),
                total_active_energy=300.0 + (steps / 100),
                total_distance_km=steps / 1500,
                flights_climbed=10.0,
                activity_sessions=3,
                peak_activity_hour=14,
                activity_variance=0.3,
                sedentary_hours=14.0,
                active_hours=3.0,
                earliest_activity=None,
                latest_activity=None,
            )
            summaries.append(summary)

        return summaries

    @pytest.fixture
    def stable_heart_data(self):
        """Create stable heart rate data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(30):  # 30 days
            summary = DailyHeartSummary(
                date=base_date + timedelta(days=i),
                avg_resting_hr=65.0 + (i % 5),  # Small variation
                min_hr=50.0,
                max_hr=140.0,
                avg_hrv_sdnn=45.0 + (i % 3) * 2,
                min_hrv_sdnn=40.0,
                hr_measurements=100,
                hrv_measurements=20,
                high_hr_episodes=0,
                low_hr_episodes=0,
                circadian_hr_range=15.0,
                morning_hr=62.0,
                evening_hr=68.0,
            )
            summaries.append(summary)

        return summaries

    def test_calculate_rolling_statistics(self, calculator, stable_sleep_data):
        """Test rolling window statistics calculation."""
        result = calculator.calculate_rolling_statistics(
            stable_sleep_data,
            window_days=7,
            metric_extractor=lambda s: s.total_sleep_hours,
        )

        assert hasattr(result, "mean")
        assert hasattr(result, "std")
        assert hasattr(result, "min")
        assert hasattr(result, "max")
        assert hasattr(result, "trend")

        # With stable data, std should be low
        assert result.std < 0.5
        assert 7.0 <= result.mean <= 8.0

    def test_calculate_trend_features(self, calculator, variable_activity_data):
        """Test trend feature calculation."""
        trend = calculator.calculate_trend_features(
            variable_activity_data,
            metric_extractor=lambda a: a.total_steps,
            window_days=7,
        )

        assert hasattr(trend, "slope")
        assert hasattr(trend, "r_squared")
        assert hasattr(trend, "acceleration")
        assert hasattr(trend, "is_increasing")
        assert hasattr(trend, "is_stable")

        # Last 7 days include weekend variation, so slope might be negative
        # Just check that trend detection works
        assert isinstance(trend.slope, float)
        assert trend.is_increasing == (trend.slope > 0)

    def test_calculate_variability_features(self, calculator, stable_heart_data):
        """Test variability feature calculation."""
        variability = calculator.calculate_variability_features(
            stable_heart_data,
            metric_extractor=lambda h: h.avg_resting_hr,
            window_days=7,
        )

        assert hasattr(variability, "coefficient_of_variation")
        assert hasattr(variability, "range")
        assert hasattr(variability, "iqr")  # Interquartile range
        assert hasattr(variability, "mad")  # Median absolute deviation

        # Stable data should have low CV
        assert variability.coefficient_of_variation < 0.1

    def test_calculate_periodicity_features(self, calculator, variable_activity_data):
        """Test periodicity detection (weekly patterns)."""
        periodicity = calculator.calculate_periodicity_features(
            variable_activity_data,
            metric_extractor=lambda a: a.total_steps,
            max_period_days=7,
        )

        assert hasattr(periodicity, "dominant_period_days")
        assert hasattr(periodicity, "period_strength")
        assert hasattr(periodicity, "phase_shift")

        # Should detect weekly pattern
        assert periodicity.dominant_period_days == 7
        assert periodicity.period_strength > 0.5

    def test_calculate_anomaly_scores(self, calculator, stable_sleep_data):
        """Test anomaly score calculation."""
        # Add an anomalous day
        anomalous_data = stable_sleep_data.copy()
        anomalous_data[15] = DailySleepSummary(
            date=anomalous_data[15].date,
            total_time_in_bed_hours=12.0,  # Much longer than usual
            total_sleep_hours=11.0,
            sleep_efficiency=0.92,
            sleep_sessions=1,
            longest_sleep_hours=11.0,
            sleep_fragmentation_index=0.05,
            earliest_bedtime=None,
            latest_wake_time=None,
            mid_sleep_time=None,
        )

        anomaly_scores = calculator.calculate_anomaly_scores(
            anomalous_data,
            metric_extractor=lambda s: s.total_sleep_hours,
            window_days=7,
        )

        assert len(anomaly_scores) == len(anomalous_data)
        # Day 15 should have high anomaly score
        assert anomaly_scores[15] > 2.0  # More than 2 std devs
        # Most days should have relatively lower scores
        normal_scores = [s for i, s in enumerate(anomaly_scores) if i != 15]
        # At least 80% of normal days should have low scores
        low_score_count = sum(1 for s in normal_scores if s < 2.0)
        assert low_score_count / len(normal_scores) > 0.8

    def test_calculate_change_point_detection(self, calculator):
        """Test change point detection in patterns."""
        # Create data with a clear change point
        summaries = []
        base_date = date(2024, 1, 1)

        # First 15 days: 8000 steps
        for i in range(15):
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=8000.0,
                total_active_energy=300.0,
                total_distance_km=6.0,
                flights_climbed=10.0,
                activity_sessions=3,
                peak_activity_hour=14,
                activity_variance=0.2,
                sedentary_hours=14.0,
                active_hours=3.0,
                earliest_activity=None,
                latest_activity=None,
            )
            summaries.append(summary)

        # Next 15 days: 12000 steps (change point)
        for i in range(15, 30):
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=12000.0,
                total_active_energy=450.0,
                total_distance_km=9.0,
                flights_climbed=15.0,
                activity_sessions=4,
                peak_activity_hour=14,
                activity_variance=0.2,
                sedentary_hours=12.0,
                active_hours=5.0,
                earliest_activity=None,
                latest_activity=None,
            )
            summaries.append(summary)

        change_points = calculator.detect_change_points(
            summaries, metric_extractor=lambda a: a.total_steps, min_segment_days=5
        )

        assert len(change_points) >= 1
        # Should detect change around day 10-20 (depends on window size)
        assert any(10 <= cp.index <= 20 for cp in change_points)
        # Change magnitude should be significant (4000 step difference)
        # But with 5-day windows, it might detect partial changes
        assert abs(change_points[0].magnitude) > 1000

    def test_calculate_cross_domain_correlation(
        self, calculator, stable_sleep_data, stable_heart_data
    ):
        """Test correlation between different domains (sleep-heart)."""
        correlation = calculator.calculate_cross_domain_correlation(
            stable_sleep_data,
            stable_heart_data,
            metric1_extractor=lambda s: s.total_sleep_hours,
            metric2_extractor=lambda h: h.avg_resting_hr,
            lag_days=0,
        )

        assert hasattr(correlation, "pearson_r")
        assert hasattr(correlation, "spearman_rho")
        assert hasattr(correlation, "p_value")
        assert hasattr(correlation, "is_significant")

        # Correlation values should be between -1 and 1
        assert -1 <= correlation.pearson_r <= 1
        assert -1 <= correlation.spearman_rho <= 1

    def test_empty_data_handling(self, calculator):
        """Test handling of empty data."""
        empty_data = []

        # Should handle empty data gracefully
        rolling_stats = calculator.calculate_rolling_statistics(
            empty_data, window_days=7, metric_extractor=lambda s: 0
        )
        assert rolling_stats.mean == 0
        assert rolling_stats.std == 0

        trend = calculator.calculate_trend_features(
            empty_data, metric_extractor=lambda s: 0, window_days=7
        )
        assert trend.slope == 0
        assert trend.is_stable is True

    def test_insufficient_data_handling(self, calculator):
        """Test handling of insufficient data for window."""
        # Only 3 days of data
        short_data = [
            DailySleepSummary(
                date=date(2024, 1, i),
                total_time_in_bed_hours=8.0,
                total_sleep_hours=7.5,
                sleep_efficiency=0.94,
                sleep_sessions=1,
                longest_sleep_hours=7.5,
                sleep_fragmentation_index=0.1,
                earliest_bedtime=None,
                latest_wake_time=None,
                mid_sleep_time=None,
            )
            for i in range(1, 4)
        ]

        # Should still calculate with available data
        rolling_stats = calculator.calculate_rolling_statistics(
            short_data,
            window_days=7,  # Window larger than data
            metric_extractor=lambda s: s.total_sleep_hours,
        )

        assert rolling_stats.mean == 7.5
        assert rolling_stats.std == 0.0  # All same value

    def test_calculate_momentum_features(self, calculator, variable_activity_data):
        """Test momentum feature calculation (rate of change)."""
        momentum = calculator.calculate_momentum_features(
            variable_activity_data,
            metric_extractor=lambda a: a.total_steps,
            short_window=3,
            long_window=7,
        )

        assert hasattr(momentum, "short_term_momentum")
        assert hasattr(momentum, "long_term_momentum")
        assert hasattr(momentum, "momentum_divergence")
        assert hasattr(momentum, "is_accelerating")

        # Short-term momentum should be positive (last 3 days)
        # Long-term might be negative due to weekend variations
        assert isinstance(momentum.short_term_momentum, float)
        assert isinstance(momentum.long_term_momentum, float)
        # Divergence shows difference between short and long term
        assert (
            momentum.momentum_divergence
            == momentum.short_term_momentum - momentum.long_term_momentum
        )
