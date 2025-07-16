"""
Test Sleep Feature Calculator

TDD approach for extracting sleep feature calculation from AdvancedFeatureEngineer.
Following clean code principles and focusing on single responsibility.
"""

from datetime import datetime, date, timedelta

import pytest
import numpy as np

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
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_sleep_hours=7.5 + (i % 2) * 0.2,  # Small variation
                sleep_efficiency=0.90,
                sleep_onset=datetime.combine(
                    base_date + timedelta(days=i),
                    datetime.min.time()
                ).replace(hour=23, minute=0),  # Consistent 11 PM
                wake_time=datetime.combine(
                    base_date + timedelta(days=i+1),
                    datetime.min.time()
                ).replace(hour=6, minute=30),  # Consistent 6:30 AM
                awakenings=2,
                time_awake_minutes=15,
                rem_sleep_hours=1.8,
                deep_sleep_hours=1.5,
                light_sleep_hours=4.2,
            )
            summaries.append(summary)
        
        return summaries

    @pytest.fixture
    def irregular_sleep_data(self):
        """Create irregular sleep pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)
        
        # Irregular times ranging from 9 PM to 2 AM
        sleep_hours = [21, 23, 22, 1, 0, 22, 2, 23, 21, 0, 23, 1, 22, 0]
        
        for i in range(14):
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_sleep_hours=5 + (i % 5),  # Varying from 5-9 hours
                sleep_efficiency=0.70 + (i % 3) * 0.1,
                sleep_onset=datetime.combine(
                    base_date + timedelta(days=i if sleep_hours[i] >= 12 else i+1),
                    datetime.min.time()
                ).replace(hour=sleep_hours[i], minute=i * 5 % 60),
                wake_time=datetime.combine(
                    base_date + timedelta(days=i+1),
                    datetime.min.time()
                ).replace(hour=5 + (i % 4), minute=0),
                awakenings=1 + (i % 4),
                time_awake_minutes=10 + (i % 3) * 10,
                rem_sleep_hours=1.0 + (i % 3) * 0.5,
                deep_sleep_hours=1.0 + (i % 2) * 0.5,
                light_sleep_hours=3.0 + (i % 3) * 0.5,
            )
            summaries.append(summary)
        
        return summaries

    def test_calculate_regularity_index_regular_pattern(self, calculator, regular_sleep_data):
        """Test regularity index for consistent sleep schedule."""
        regularity_index = calculator.calculate_regularity_index(regular_sleep_data)
        
        # Regular pattern should have high index (>80)
        assert regularity_index > 80
        assert regularity_index <= 100
        assert isinstance(regularity_index, float)

    def test_calculate_regularity_index_irregular_pattern(self, calculator, irregular_sleep_data):
        """Test regularity index for inconsistent sleep schedule."""
        regularity_index = calculator.calculate_regularity_index(irregular_sleep_data)
        
        # Irregular pattern should have low index (<50)
        assert regularity_index < 50
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
        assert iv_value > 0.5  # Irregular pattern should show fragmentation
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
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_sleep_hours=duration,
                sleep_efficiency=0.85,
                sleep_onset=datetime.combine(
                    base_date + timedelta(days=i),
                    datetime.min.time()
                ).replace(hour=23, minute=0),
                wake_time=datetime.combine(
                    base_date + timedelta(days=i+1),
                    datetime.min.time()
                ).replace(hour=int(23 + duration) % 24, minute=0),
                awakenings=2,
                time_awake_minutes=15,
                rem_sleep_hours=duration * 0.2,
                deep_sleep_hours=duration * 0.2,
                light_sleep_hours=duration * 0.6,
            )
            summaries.append(summary)
        
        short_pct, long_pct = calculator.calculate_sleep_window_percentages(summaries)
        
        assert short_pct == 30.0  # 3 out of 10 nights
        assert long_pct == 20.0   # 2 out of 10 nights

    def test_calculate_timing_variances(self, calculator, irregular_sleep_data):
        """Test calculation of sleep onset and wake time variances."""
        onset_var, wake_var = calculator.calculate_timing_variances(irregular_sleep_data)
        
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
                total_sleep_hours=7.5,
                sleep_efficiency=0.85,
                sleep_onset=datetime(2024, 1, 1, 23, 0),
                wake_time=datetime(2024, 1, 2, 6, 30),
                awakenings=2,
                time_awake_minutes=20,
                rem_sleep_hours=1.5,
                deep_sleep_hours=1.5,
                light_sleep_hours=4.5,
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
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_sleep_hours=8,
                sleep_efficiency=0.85,
                sleep_onset=datetime.combine(
                    base_date + timedelta(days=i+1),  # Next day
                    datetime.min.time()
                ).replace(hour=1, minute=0),
                wake_time=datetime.combine(
                    base_date + timedelta(days=i+1),
                    datetime.min.time()
                ).replace(hour=9, minute=0),
                awakenings=2,
                time_awake_minutes=15,
                rem_sleep_hours=1.6,
                deep_sleep_hours=1.6,
                light_sleep_hours=4.8,
            )
            summaries.append(summary)
        
        # Should handle midnight crossing correctly
        regularity = calculator.calculate_regularity_index(summaries)
        assert regularity > 80  # Should still be regular
        
        onset_var, wake_var = calculator.calculate_timing_variances(summaries)
        assert onset_var < 0.5  # Low variance for consistent times
        assert wake_var < 0.5