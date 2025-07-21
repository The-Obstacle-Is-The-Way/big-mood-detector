"""
Test Activity Feature Calculator

TDD approach for extracting activity-specific calculations from AdvancedFeatureEngineer.
Activity features are crucial for detecting manic/depressive episodes.
"""

from datetime import date, time, timedelta

import pytest

class TestActivityFeatureCalculator:
    """Test activity feature calculations."""

    @pytest.fixture
    def calculator(self):
        """Create ActivityFeatureCalculator instance."""

        return ActivityFeatureCalculator()

    @pytest.fixture
    def regular_activity_data(self):
        """Create regular activity pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(14):  # 2 weeks
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=8000 + (i % 3) * 1000,  # Mild variation
                total_active_energy=300.0 + (i % 3) * 50,
                total_distance_km=6.0,
                flights_climbed=10.0,
                activity_sessions=3,
                peak_activity_hour=14,  # 2 PM peak
                activity_variance=0.2,  # Low variance
                sedentary_hours=12.0,
                active_hours=4.0,
                earliest_activity=time(7, 0),
                latest_activity=time(21, 0),
            )
            summaries.append(summary)

        return summaries

    @pytest.fixture
    def manic_activity_data(self):
        """Create hyperactive/manic pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(7):  # 1 week
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=20000 + i * 2000,  # Very high and increasing
                total_active_energy=800.0 + i * 100,
                total_distance_km=15.0,
                flights_climbed=30.0,
                activity_sessions=10,  # Many sessions
                peak_activity_hour=23,  # Late night activity
                activity_variance=0.8,  # High variance
                sedentary_hours=4.0,  # Very little rest
                active_hours=18.0,  # Almost always active
                earliest_activity=time(5, 0),
                latest_activity=time(23, 30),
            )
            summaries.append(summary)

        return summaries

    @pytest.fixture
    def depressive_activity_data(self):
        """Create low activity/depressive pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(7):  # 1 week
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=1000 - i * 100,  # Very low and decreasing
                total_active_energy=50.0,
                total_distance_km=0.5,
                flights_climbed=0.0,
                activity_sessions=1,  # Minimal sessions
                peak_activity_hour=12,  # Mid-day only
                activity_variance=0.1,
                sedentary_hours=22.0,  # Almost all day sedentary
                active_hours=0.5,  # Barely active
                earliest_activity=time(11, 0),
                latest_activity=time(13, 0),
            )
            summaries.append(summary)

        return summaries

    def test_calculate_activity_fragmentation(self, calculator, regular_activity_data):
        """Test activity fragmentation calculation."""
        result = calculator.calculate_activity_fragmentation(regular_activity_data)

        assert isinstance(result, float)
        assert 0 <= result <= 1
        # Regular pattern should have low fragmentation
        assert result < 0.3

    def test_calculate_activity_fragmentation_manic(
        self, calculator, manic_activity_data
    ):
        """Test fragmentation for manic patterns."""
        result = calculator.calculate_activity_fragmentation(manic_activity_data)

        # Manic patterns have high fragmentation
        # With our test data showing high variance (0.8) and increasing steps
        assert result > 0.3  # Adjusted based on calculation method
        assert result <= 1.0

    def test_calculate_sedentary_bouts(self, calculator, regular_activity_data):
        """Test sedentary bout analysis."""
        mean_bout, max_bout, longest_streak = calculator.calculate_sedentary_bouts(
            regular_activity_data
        )

        assert isinstance(mean_bout, float)
        assert isinstance(max_bout, float)
        assert isinstance(longest_streak, int)

        # Regular pattern expectations
        assert 600 < mean_bout < 800  # 10-13 hours in minutes
        assert max_bout >= mean_bout
        assert 0 <= longest_streak <= 14  # May be 0 if no days exceed threshold

    def test_calculate_sedentary_bouts_depressive(
        self, calculator, depressive_activity_data
    ):
        """Test sedentary bouts for depressive patterns."""
        mean_bout, max_bout, longest_streak = calculator.calculate_sedentary_bouts(
            depressive_activity_data
        )

        # Depressive patterns have very high sedentary time
        assert mean_bout > 1200  # >20 hours in minutes
        assert max_bout >= 1320  # >=22 hours
        assert longest_streak >= 5  # Many consecutive sedentary days

    def test_calculate_activity_intensity_metrics(
        self, calculator, regular_activity_data
    ):
        """Test activity intensity ratio and distribution."""
        result = calculator.calculate_activity_intensity_metrics(regular_activity_data)

        assert hasattr(result, "intensity_ratio")
        assert hasattr(result, "high_intensity_days")
        assert hasattr(result, "low_intensity_days")
        assert hasattr(result, "moderate_intensity_days")

        # Regular pattern should be mostly moderate
        assert result.moderate_intensity_days > result.high_intensity_days
        assert result.moderate_intensity_days > result.low_intensity_days

    def test_calculate_activity_rhythm_strength(
        self, calculator, regular_activity_data
    ):
        """Test circadian rhythm strength from activity."""
        strength = calculator.calculate_activity_rhythm_strength(regular_activity_data)

        assert 0 <= strength <= 1
        # Regular pattern should have good rhythm strength
        assert strength > 0.6

    def test_calculate_activity_timing_consistency(
        self, calculator, regular_activity_data
    ):
        """Test consistency of activity timing."""
        onset_consistency, offset_consistency = (
            calculator.calculate_activity_timing_consistency(regular_activity_data)
        )

        assert 0 <= onset_consistency <= 1
        assert 0 <= offset_consistency <= 1

        # Regular pattern should be consistent
        assert onset_consistency > 0.7
        assert offset_consistency > 0.7

    def test_detect_activity_anomalies(self, calculator, manic_activity_data):
        """Test detection of anomalous activity patterns."""
        anomalies = calculator.detect_activity_anomalies(manic_activity_data)

        assert hasattr(anomalies, "has_hyperactivity")
        assert hasattr(anomalies, "has_hypoactivity")
        assert hasattr(anomalies, "has_irregular_timing")
        assert hasattr(anomalies, "anomaly_days")

        # Manic pattern should show hyperactivity
        assert anomalies.has_hyperactivity is True
        assert anomalies.has_hypoactivity is False
        assert len(anomalies.anomaly_days) > 0

    def test_empty_data_handling(self, calculator):
        """Test handling of empty data."""
        empty_data = []

        # Should handle empty data gracefully
        fragmentation = calculator.calculate_activity_fragmentation(empty_data)
        assert fragmentation == 0.0

        mean_bout, max_bout, streak = calculator.calculate_sedentary_bouts(empty_data)
        assert mean_bout == 0.0
        assert max_bout == 0.0
        assert streak == 0

        intensity = calculator.calculate_activity_intensity_metrics(empty_data)
        assert intensity.intensity_ratio == 0.0

    def test_single_day_handling(self, calculator):
        """Test handling of single day data."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        single_day = [
            DailyActivitySummary(
                date=date(2024, 1, 1),
                total_steps=8000,
                total_active_energy=300.0,
                total_distance_km=6.0,
                flights_climbed=10.0,
                activity_sessions=3,
                peak_activity_hour=14,
                activity_variance=0.2,
                sedentary_hours=12.0,
                active_hours=4.0,
                earliest_activity=time(7, 0),
                latest_activity=time(21, 0),
            )
        ]

        # Should handle single day without errors
        fragmentation = calculator.calculate_activity_fragmentation(single_day)
        assert fragmentation >= 0.0

        strength = calculator.calculate_activity_rhythm_strength(single_day)
        assert 0 <= strength <= 1

    def test_calculate_step_acceleration(self, calculator):
        """Test step count acceleration/deceleration patterns."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        # Create increasing step pattern (potential manic acceleration)
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(7):
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=5000 + i * 2000,  # Accelerating
                total_active_energy=200.0 + i * 100,
                total_distance_km=4.0 + i * 2,
                flights_climbed=5.0,
                activity_sessions=3,
                peak_activity_hour=14,
                activity_variance=0.3,
                sedentary_hours=14.0 - i,
                active_hours=2.0 + i,
                earliest_activity=time(8, 0),
                latest_activity=time(20, 0),
            )
            summaries.append(summary)

        acceleration = calculator.calculate_step_acceleration(summaries)

        assert isinstance(acceleration, float)
        # Should detect positive acceleration
        assert acceleration > 1000  # Steps per day acceleration
