"""
Test Circadian Feature Calculator

TDD approach for extracting circadian rhythm calculations from AdvancedFeatureEngineer.
Circadian features are critical for bipolar disorder detection.
"""

from datetime import date, datetime, time, timedelta

import pytest

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


class TestCircadianFeatureCalculator:
    """Test circadian rhythm feature calculations."""

    @pytest.fixture
    def calculator(self):
        """Create CircadianFeatureCalculator instance."""
        from big_mood_detector.domain.services.circadian_feature_calculator import (
            CircadianFeatureCalculator,
        )
        return CircadianFeatureCalculator()

    @pytest.fixture
    def regular_activity_data(self):
        """Create regular activity pattern data."""
        summaries = []
        base_date = date(2024, 1, 1)
        
        for i in range(14):  # 2 weeks
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=8000 + (i % 2) * 500,  # Slight variation
                total_active_energy=300.0,
                total_distance_km=6.0,
                flights_climbed=10.0,
                activity_sessions=3,
                peak_activity_hour=14,  # 2 PM peak
                activity_variance=100.0,
                sedentary_hours=12.0,
                active_hours=4.0,
                earliest_activity=time(7, 0),
                latest_activity=time(21, 0),
            )
            summaries.append(summary)
        
        return summaries

    @pytest.fixture
    def phase_delayed_sleep_data(self):
        """Create sleep data with delayed phase (late sleeper)."""
        summaries = []
        base_date = date(2024, 1, 1)
        
        for i in range(14):
            # Sleep at 2-3 AM, wake at 10-11 AM
            mid_sleep = datetime.combine(
                base_date + timedelta(days=i+1),
                time(6, 30)  # Mid-sleep around 6:30 AM
            )
            
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.5,
                total_sleep_hours=8.0,
                sleep_efficiency=0.85,
                sleep_sessions=1,
                longest_sleep_hours=8.0,
                sleep_fragmentation_index=0.15,
                earliest_bedtime=time(2, 30),  # 2:30 AM
                latest_wake_time=time(10, 30),  # 10:30 AM
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)
        
        return summaries

    @pytest.fixture
    def phase_advanced_sleep_data(self):
        """Create sleep data with advanced phase (early sleeper)."""
        summaries = []
        base_date = date(2024, 1, 1)
        
        for i in range(14):
            # Sleep at 8-9 PM, wake at 4-5 AM
            mid_sleep = datetime.combine(
                base_date + timedelta(days=i+1),
                time(0, 30)  # Mid-sleep around 12:30 AM
            )
            
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.5,
                total_sleep_hours=8.0,
                sleep_efficiency=0.85,
                sleep_sessions=1,
                longest_sleep_hours=8.0,
                sleep_fragmentation_index=0.15,
                earliest_bedtime=time(20, 30),  # 8:30 PM
                latest_wake_time=time(4, 30),  # 4:30 AM
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)
        
        return summaries

    def test_calculate_l5_m10_metrics(self, calculator, regular_activity_data):
        """Test L5 (least active 5 hours) and M10 (most active 10 hours) calculation."""
        result = calculator.calculate_l5_m10_metrics(regular_activity_data)
        
        assert hasattr(result, 'l5_value')
        assert hasattr(result, 'm10_value')
        assert hasattr(result, 'l5_onset')
        assert hasattr(result, 'm10_onset')
        
        # L5 should be lower than M10
        assert result.l5_value < result.m10_value
        assert result.l5_value >= 0
        assert result.m10_value >= 0

    def test_calculate_phase_shifts_delayed(self, calculator, phase_delayed_sleep_data):
        """Test detection of delayed sleep phase."""
        result = calculator.calculate_phase_shifts(phase_delayed_sleep_data)
        
        assert hasattr(result, 'phase_advance_hours')
        assert hasattr(result, 'phase_delay_hours')
        assert hasattr(result, 'phase_type')
        
        # Should detect phase delay
        assert result.phase_delay_hours > 2.0  # More than 2 hours delayed
        assert result.phase_advance_hours == 0.0
        assert result.phase_type == "delayed"

    def test_calculate_phase_shifts_advanced(self, calculator, phase_advanced_sleep_data):
        """Test detection of advanced sleep phase."""
        result = calculator.calculate_phase_shifts(phase_advanced_sleep_data)
        
        # Should detect phase advance
        assert result.phase_advance_hours > 2.0  # More than 2 hours advanced
        assert result.phase_delay_hours == 0.0
        assert result.phase_type == "advanced"

    def test_calculate_phase_shifts_normal(self, calculator):
        """Test normal sleep phase detection."""
        # Create normal sleep pattern (11 PM - 7 AM)
        summaries = []
        base_date = date(2024, 1, 1)
        
        for i in range(7):
            mid_sleep = datetime.combine(
                base_date + timedelta(days=i+1),
                time(3, 0)  # Mid-sleep at 3 AM
            )
            
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.0,
                total_sleep_hours=7.5,
                sleep_efficiency=0.90,
                sleep_sessions=1,
                longest_sleep_hours=7.5,
                sleep_fragmentation_index=0.1,
                earliest_bedtime=time(23, 0),  # 11 PM
                latest_wake_time=time(7, 0),   # 7 AM
                mid_sleep_time=mid_sleep,
            )
            summaries.append(summary)
        
        result = calculator.calculate_phase_shifts(summaries)
        
        # Should detect normal phase
        assert result.phase_advance_hours < 1.0
        assert result.phase_delay_hours < 1.0
        assert result.phase_type == "normal"

    def test_estimate_dlmo(self, calculator, phase_delayed_sleep_data):
        """Test Dim Light Melatonin Onset estimation."""
        dlmo = calculator.estimate_dlmo(phase_delayed_sleep_data)
        
        assert isinstance(dlmo, datetime)
        # DLMO should be ~2 hours before sleep onset
        # For 2:30 AM sleep, DLMO should be around 12:30 AM
        assert dlmo.hour in [0, 1]  # Around midnight to 1 AM

    def test_estimate_core_temp_nadir(self, calculator, phase_delayed_sleep_data):
        """Test core body temperature nadir estimation."""
        nadir = calculator.estimate_core_temp_nadir(phase_delayed_sleep_data)
        
        assert isinstance(nadir, datetime)
        # Nadir should be ~2 hours before wake time
        # For 10:30 AM wake, nadir should be around 8:30 AM
        assert nadir.hour in [8, 9]  # Around 8-9 AM

    def test_calculate_circadian_amplitude(self, calculator, regular_activity_data):
        """Test circadian amplitude calculation from activity data."""
        amplitude = calculator.calculate_circadian_amplitude(
            regular_activity_data,
            phase_delayed_sleep_data
        )
        
        assert 0 <= amplitude <= 1
        # Regular pattern should have decent amplitude
        assert amplitude > 0.5

    def test_empty_data_handling(self, calculator):
        """Test handling of empty data."""
        empty_activity = []
        empty_sleep = []
        
        # Should handle empty data gracefully
        l5_m10_result = calculator.calculate_l5_m10_metrics(empty_activity)
        assert l5_m10_result.l5_value == 0
        assert l5_m10_result.m10_value == 0
        
        phase_result = calculator.calculate_phase_shifts(empty_sleep)
        assert phase_result.phase_type == "unknown"
        
        dlmo = calculator.estimate_dlmo(empty_sleep)
        assert dlmo is None
        
        nadir = calculator.estimate_core_temp_nadir(empty_sleep)
        assert nadir is None

    def test_phase_angle_calculation(self, calculator):
        """Test phase angle between sleep and activity rhythms."""
        # Create misaligned sleep and activity patterns
        sleep_data = []
        activity_data = []
        base_date = date(2024, 1, 1)
        
        for i in range(7):
            # Late sleep pattern
            sleep = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.0,
                total_sleep_hours=7.5,
                sleep_efficiency=0.85,
                sleep_sessions=1,
                longest_sleep_hours=7.5,
                sleep_fragmentation_index=0.15,
                earliest_bedtime=time(2, 0),  # 2 AM
                latest_wake_time=time(10, 0),  # 10 AM
                mid_sleep_time=datetime.combine(
                    base_date + timedelta(days=i+1),
                    time(6, 0)
                ),
            )
            
            # Early activity pattern (misaligned)
            activity = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=8000,
                total_active_energy=300.0,
                total_distance_km=6.0,
                flights_climbed=10.0,
                activity_sessions=3,
                peak_activity_hour=10,  # Peak at 10 AM (right when waking)
                activity_variance=100.0,
                sedentary_hours=12.0,
                active_hours=4.0,
                earliest_activity=time(10, 0),
                latest_activity=time(20, 0),
            )
            
            sleep_data.append(sleep)
            activity_data.append(activity)
        
        phase_angle = calculator.calculate_phase_angle(sleep_data, activity_data)
        
        assert isinstance(phase_angle, float)
        # Large phase angle indicates misalignment
        assert abs(phase_angle) > 2.0  # More than 2 hours misaligned