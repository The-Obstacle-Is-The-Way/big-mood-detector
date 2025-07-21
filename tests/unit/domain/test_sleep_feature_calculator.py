"""
Test Sleep Feature Calculator

TDD approach for extracting sleep feature calculation from AdvancedFeatureEngineer.
Following clean code principles and focusing on single responsibility.
"""

from datetime import date, datetime, time, timedelta

import pytest

from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


class TestSleepFeatureCalculator:
    """Test sleep-specific feature calculations."""

    @pytest.fixture
    def calculator(self):
        """Create SleepFeatureCalculator instance."""
        from big_mood_detector.domain.services.sleep_feature_calculator import (
            SleepFeatureCalculator,
        )

        return SleepFeatureCalculator()

    @pytest.fixture
    def regular_sleep_data(self):
        """Create regular sleep pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(14):  # 2 weeks of data
            # Create mid-sleep time around 2:45 AM (consistent)
            mid_sleep = datetime.combine(base_date + timedelta(days=i + 1), time(2, 45))

            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.0,
                total_sleep_hours=7.5 + (i % 2) * 0.2,  # Small variation
                sleep_efficiency=0.90,
                sleep_sessions=1,
                longest_sleep_hours=7.5 + (i % 2) * 0.2,
                sleep_fragmentation_index=0.1,  # Low fragmentation
                earliest_bedtime=time(23, 0),  # Consistent 11 PM
                latest_wake_time=time(6, 30),  # Consistent 6:30 AM
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)

        return summaries

    @pytest.fixture
    def irregular_sleep_data(self):
        """Create irregular sleep pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)

        # Irregular times ranging from 9 PM to 2 AM
        bedtime_hours = [21, 23, 22, 1, 0, 22, 2, 23, 21, 0, 23, 1, 22, 0]
        wake_hours = [5, 8, 6, 9, 7, 6, 10, 7, 5, 8, 7, 9, 6, 8]

        for i in range(14):
            # Calculate mid-sleep time based on bedtime and wake time
            bed_hour = bedtime_hours[i]
            wake_hour = wake_hours[i]

            # Handle crossing midnight
            if bed_hour < 12:  # After midnight
                mid_hour = (bed_hour + 24 + wake_hour) / 2
            else:
                mid_hour = (bed_hour + wake_hour + 24) / 2

            mid_hour = mid_hour % 24
            mid_sleep = datetime.combine(
                base_date + timedelta(days=i + 1 if mid_hour < 12 else i),
                time(int(mid_hour), int((mid_hour % 1) * 60)),
            )

            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=6 + (i % 4),  # Varying 6-9 hours
                total_sleep_hours=5 + (i % 5),  # Varying from 5-9 hours
                sleep_efficiency=0.70 + (i % 3) * 0.1,
                sleep_sessions=1 + (i % 2),  # Sometimes fragmented
                longest_sleep_hours=4 + (i % 4),
                sleep_fragmentation_index=0.1 + (i % 4) * 0.1,  # Varying fragmentation
                earliest_bedtime=time(bedtime_hours[i], i * 5 % 60),
                latest_wake_time=time(wake_hours[i], 0),
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)

        return summaries

    def test_calculate_regularity_index_regular_pattern(
        self, calculator, regular_sleep_data
    ):
        """Test regularity index for consistent sleep schedule."""
        regularity_index = calculator.calculate_regularity_index(regular_sleep_data)

        # Regular pattern should have high index (>80)
        assert regularity_index > 80
        assert regularity_index <= 100
        assert isinstance(regularity_index, float)

    def test_calculate_regularity_index_irregular_pattern(
        self, calculator, irregular_sleep_data
    ):
        """Test regularity index for inconsistent sleep schedule."""
        regularity_index = calculator.calculate_regularity_index(irregular_sleep_data)

        # Irregular pattern should have lower index than regular
        assert regularity_index < 80  # Not as regular as the regular pattern
        assert regularity_index >= 0
        assert isinstance(regularity_index, float)

    def test_calculate_interdaily_stability(self, calculator, regular_sleep_data):
        """Test interdaily stability calculation."""
        is_value = calculator.calculate_interdaily_stability(regular_sleep_data)

        # IS ranges from 0-1, higher = more stable
        assert 0 <= is_value <= 1
        assert is_value > 0.7  # Regular pattern should be stable
        assert isinstance(is_value, float)

    def test_calculate_intradaily_variability(self, calculator, irregular_sleep_data):
        """Test intradaily variability calculation."""
        iv_value = calculator.calculate_intradaily_variability(irregular_sleep_data)

        # IV ranges from 0-2, higher = more fragmented
        assert 0 <= iv_value <= 2
        assert iv_value > 0.3  # Some fragmentation expected
        assert isinstance(iv_value, float)

    def test_calculate_relative_amplitude(self, calculator, regular_sleep_data):
        """Test relative amplitude calculation."""
        ra_value = calculator.calculate_relative_amplitude(regular_sleep_data)

        # RA ranges from 0-1, higher = stronger rhythm
        assert 0 <= ra_value <= 1
        assert ra_value > 0.6  # Regular pattern should have strong rhythm
        assert isinstance(ra_value, float)

    def test_calculate_sleep_window_percentages(self, calculator):
        """Test calculation of short and long sleep window percentages."""
        # Create data with specific sleep durations
        summaries = []
        base_date = date(2024, 1, 1)

        # 3 short nights (<6h), 2 long nights (>10h), 5 normal nights
        durations = [5, 5.5, 5, 11, 11, 7, 7.5, 8, 8.5, 7]

        for i, duration in enumerate(durations):
            mid_sleep = datetime.combine(
                base_date + timedelta(days=i + 1), time(3, 0)  # Roughly 3 AM mid-sleep
            )

            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=duration + 0.5,
                total_sleep_hours=duration,
                sleep_efficiency=0.85,
                sleep_sessions=1,
                longest_sleep_hours=duration,
                sleep_fragmentation_index=0.15,
                earliest_bedtime=time(23, 0),
                latest_wake_time=time(int(23 + duration) % 24, 0),
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)

        short_pct, long_pct = calculator.calculate_sleep_window_percentages(summaries)

        assert short_pct == 30.0  # 3 out of 10 nights
        assert long_pct == 20.0  # 2 out of 10 nights

    def test_calculate_timing_variances(self, calculator, irregular_sleep_data):
        """Test calculation of sleep onset and wake time variances."""
        onset_var, wake_var = calculator.calculate_timing_variances(
            irregular_sleep_data
        )

        # Variances should be positive
        assert onset_var > 0
        assert wake_var > 0

        # Irregular pattern should have high variance
        assert onset_var > 1.0  # More than 1 hour² variance
        assert isinstance(onset_var, float)
        assert isinstance(wake_var, float)

    def test_empty_data_handling(self, calculator):
        """Test handling of empty data."""
        empty_data = []

        # Should return sensible defaults without crashing
        regularity = calculator.calculate_regularity_index(empty_data)
        assert regularity == 0.0

        is_value = calculator.calculate_interdaily_stability(empty_data)
        assert is_value == 0.0

        iv_value = calculator.calculate_intradaily_variability(empty_data)
        assert iv_value == 0.0

        ra_value = calculator.calculate_relative_amplitude(empty_data)
        assert ra_value == 0.0

        short_pct, long_pct = calculator.calculate_sleep_window_percentages(empty_data)
        assert short_pct == 0.0
        assert long_pct == 0.0

        onset_var, wake_var = calculator.calculate_timing_variances(empty_data)
        assert onset_var == 0.0
        assert wake_var == 0.0

    def test_single_day_handling(self, calculator):
        """Test handling of single day data."""
        single_day = [
            DailySleepSummary(
                date=date(2024, 1, 1),
                total_time_in_bed_hours=8.0,
                total_sleep_hours=7.5,
                sleep_efficiency=0.85,
                sleep_sessions=1,
                longest_sleep_hours=7.5,
                sleep_fragmentation_index=0.15,
                earliest_bedtime=time(23, 0),
                latest_wake_time=time(6, 30),
                mid_sleep_time=datetime(2024, 1, 2, 2, 45),
            )
        ]

        # Should handle gracefully without division by zero
        regularity = calculator.calculate_regularity_index(single_day)
        assert isinstance(regularity, float)
        assert 0 <= regularity <= 100

    def test_late_night_sleep_handling(self, calculator):
        """Test handling of sleep times crossing midnight."""
        summaries = []
        base_date = date(2024, 1, 1)

        # Create pattern with late night sleep (after midnight)
        for i in range(7):
            # Sleep at 1 AM, wake at 9 AM
            mid_sleep = datetime.combine(
                base_date + timedelta(days=i + 1),
                time(5, 0),  # 5 AM mid-sleep for 1 AM - 9 AM sleep
            )

            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.5,
                total_sleep_hours=8,
                sleep_efficiency=0.85,
                sleep_sessions=1,
                longest_sleep_hours=8,
                sleep_fragmentation_index=0.15,
                earliest_bedtime=time(1, 0),  # 1 AM
                latest_wake_time=time(9, 0),  # 9 AM
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)

        # Should handle midnight crossing correctly
        regularity = calculator.calculate_regularity_index(summaries)
        assert regularity > 80  # Should still be regular

        onset_var, wake_var = calculator.calculate_timing_variances(summaries)
        assert onset_var < 0.5  # Low variance for consistent times
        assert wake_var < 0.5

    def test_regularity_excludes_daytime_naps(self, calculator):
        """Daytime naps should not affect sleep regularity calculations."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(7):
            # Main sleep: 11 PM to 7 AM with a daytime nap at 12:30 PM
            mid_sleep = datetime.combine(
                base_date + timedelta(days=i + 1),
                time(3, 0),
            )

            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=9.0,
                total_sleep_hours=8.5,
                sleep_efficiency=0.9,
                sleep_sessions=2,
                longest_sleep_hours=8.0,
                sleep_fragmentation_index=0.1,
                earliest_bedtime=time(12, 30),  # Daytime nap start
                latest_wake_time=time(13, 30),  # Nap end later than morning wake
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)

        regularity = calculator.calculate_regularity_index(summaries)
        assert regularity > 80
