"""
Unit tests for Advanced Feature Engineering Service

Tests research-based feature extraction for mood prediction.
"""

import numpy as np
import pytest
from datetime import date, datetime, time, timedelta

from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
    AdvancedFeatures,
)
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary
from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary


class TestAdvancedFeatureEngineer:
    """Test advanced feature engineering for mood prediction."""

    def test_extract_features_with_minimal_data(self):
        """Test feature extraction with minimal valid data."""
        engineer = AdvancedFeatureEngineer()
        current_date = date(2024, 1, 15)
        
        # Create minimal sleep data
        sleep_data = [
            DailySleepSummary(
                date=current_date - timedelta(days=i),
                total_sleep_hours=7.5,
                sleep_efficiency=0.85,
                sleep_fragmentation_index=0.1,
                sleep_sessions=1,
                longest_sleep_hours=7.5,
                mid_sleep_time=datetime.combine(current_date - timedelta(days=i), time(2, 45)),
                earliest_bedtime=time(23, 0),
                latest_wake_time=time(6, 30),
                total_time_in_bed_hours=7.5,
            )
            for i in range(10)
        ]
        
        # Create minimal activity data
        activity_data = [
            DailyActivitySummary(
                date=current_date - timedelta(days=i),
                total_steps=8000,
                active_hours=1.0,
                total_active_energy=600.0,
                sedentary_hours=10.0,
                activity_variance=1000.0,
                peak_activity_hour=14,
            )
            for i in range(10)
        ]
        
        # Create minimal heart data
        heart_data = [
            DailyHeartSummary(
                date=current_date - timedelta(days=i),
                hr_measurements=100,
                morning_hr=70.0,
                min_hr=50.0,
                max_hr=120.0,
                evening_hr=10.0,
                avg_resting_hr=60.0,
                avg_hrv_sdnn=50.0,
                circadian_hr_range=15.0,
            )
            for i in range(10)
        ]
        
        # Extract features
        features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=sleep_data,
            historical_activity=activity_data,
            historical_heart=heart_data,
            lookback_days=30,
        )
        
        # Verify basic structure
        assert features.date == current_date
        assert features.sleep_duration_hours == 7.5
        assert features.sleep_efficiency == 0.85
        assert features.total_steps == 8000
        assert features.avg_resting_hr == 60.0
        assert features.hrv_sdnn == 50.0
        
        # Verify advanced features exist
        assert 0 <= features.sleep_regularity_index <= 100
        assert 0 <= features.interdaily_stability <= 1
        assert 0 <= features.intradaily_variability <= 2
        assert 0 <= features.relative_amplitude <= 1
        
        # Check ML features
        ml_features = features.to_ml_features()
        assert len(ml_features) == 36  # Seoul study requirement
        assert all(isinstance(f, (int, float)) for f in ml_features)

    def test_sleep_regularity_calculation(self):
        """Test sleep regularity index calculation."""
        engineer = AdvancedFeatureEngineer()
        current_date = date(2024, 1, 15)
        
        # Create regular sleep pattern
        regular_sleep = []
        for i in range(30):
            regular_sleep.append(
                DailySleepSummary(
                    date=current_date - timedelta(days=i),
                    total_sleep_hours=8.0,
                    sleep_efficiency=0.85,
                    sleep_fragmentation_index=0.1,
                    sleep_sessions=1,
                    mid_sleep_time=datetime.combine(
                        current_date - timedelta(days=i), 
                        time(3, 0)  # Consistent mid-sleep time
                    ),
                    earliest_bedtime=time(23, 0),
                    latest_wake_time=time(7, 0),
                    total_time_in_bed_hours=8.0,
                longest_sleep_hours=8.0,
                    )
            )
        
        # Create irregular sleep pattern
        irregular_sleep = []
        for i in range(30):
            # Vary sleep times significantly
            hour_offset = (i % 5) * 2  # 0, 2, 4, 6, 8 hour variations
            irregular_sleep.append(
                DailySleepSummary(
                    date=current_date - timedelta(days=i),
                    total_sleep_hours=6 + (i % 4),  # 6-9 hours
                    sleep_efficiency=0.85,
                    sleep_fragmentation_index=0.1,
                    sleep_sessions=1,
                    mid_sleep_time=datetime.combine(
                        current_date - timedelta(days=i),
                        time((1 + hour_offset) % 24, 0)
                    ),
                    earliest_bedtime=time((21 + hour_offset) % 24, 0),
                    latest_wake_time=time((5 + hour_offset) % 24, 0),
                    total_time_in_bed_hours=8.0,
                longest_sleep_hours=8.0,
                    )
            )
        
        # Empty activity and heart data for simplicity
        empty_activity = []
        empty_heart = []
        
        # Test regular pattern
        regular_features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=regular_sleep,
            historical_activity=empty_activity,
            historical_heart=empty_heart,
            lookback_days=30,
        )
        
        # Test irregular pattern
        irregular_features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=irregular_sleep,
            historical_activity=empty_activity,
            historical_heart=empty_heart,
            lookback_days=30,
        )
        
        # Regular pattern should have higher regularity index
        assert regular_features.sleep_regularity_index > 80  # Very regular
        assert irregular_features.sleep_regularity_index < 50  # Very irregular
        assert regular_features.sleep_regularity_index > irregular_features.sleep_regularity_index

    def test_circadian_phase_detection(self):
        """Test circadian phase advance/delay detection."""
        engineer = AdvancedFeatureEngineer()
        current_date = date(2024, 1, 15)
        
        # Create phase delayed pattern (late sleeper)
        delayed_sleep = []
        for i in range(15):
            delayed_sleep.append(
                DailySleepSummary(
                    date=current_date - timedelta(days=i),
                    total_sleep_hours=8.0,
                    sleep_efficiency=0.85,
                    sleep_fragmentation_index=0.1,
                    sleep_sessions=1,
                    mid_sleep_time=datetime.combine(
                        current_date - timedelta(days=i),
                        time(6, 0)  # 6 AM mid-sleep (delayed)
                    ),
                    earliest_bedtime=time(2, 0),
                    latest_wake_time=time(10, 0),
                    total_time_in_bed_hours=8.0,
                longest_sleep_hours=8.0,
                    )
            )
        
        # Create phase advanced pattern (early sleeper)
        advanced_sleep = []
        for i in range(15):
            advanced_sleep.append(
                DailySleepSummary(
                    date=current_date - timedelta(days=i),
                    total_sleep_hours=8.0,
                    sleep_efficiency=0.85,
                    sleep_fragmentation_index=0.1,
                    sleep_sessions=1,
                    mid_sleep_time=datetime.combine(
                        current_date - timedelta(days=i),
                        time(0, 0)  # Midnight mid-sleep (advanced)
                    ),
                    earliest_bedtime=time(20, 0),
                    latest_wake_time=time(4, 0),
                    total_time_in_bed_hours=8.0,
                longest_sleep_hours=8.0,
                    )
            )
        
        empty_activity = []
        empty_heart = []
        
        # Test delayed pattern
        delayed_features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=delayed_sleep,
            historical_activity=empty_activity,
            historical_heart=empty_heart,
            lookback_days=30,
        )
        
        # Test advanced pattern
        advanced_features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=advanced_sleep,
            historical_activity=empty_activity,
            historical_heart=empty_heart,
            lookback_days=30,
        )
        
        # Phase delayed should have positive delay value
        assert delayed_features.circadian_phase_delay > 2  # >2 hours delayed
        assert delayed_features.circadian_phase_advance == 0
        
        # Phase advanced should have positive advance value
        assert advanced_features.circadian_phase_advance > 2  # >2 hours advanced
        assert advanced_features.circadian_phase_delay == 0

    def test_clinical_indicators(self):
        """Test clinical indicator detection."""
        engineer = AdvancedFeatureEngineer()
        current_date = date(2024, 1, 15)
        
        # Create hypersomnia pattern (>10 hours sleep)
        hypersomnia_sleep = []
        for i in range(20):
            hypersomnia_sleep.append(
                DailySleepSummary(
                    date=current_date - timedelta(days=i),
                    total_sleep_hours=11.5,  # Excessive sleep
                    sleep_efficiency=0.85,
                    sleep_fragmentation_index=0.1,
                    sleep_sessions=1,
                    longest_sleep_hours=11.5,
                    earliest_bedtime=time(22, 0),
                    latest_wake_time=time(9, 30),
                    total_time_in_bed_hours=11.5,
                )
            )
        
        # Create insomnia pattern (<6 hours sleep)
        insomnia_sleep = []
        for i in range(20):
            insomnia_sleep.append(
                DailySleepSummary(
                    date=current_date - timedelta(days=i),
                    total_sleep_hours=4.5,  # Insufficient sleep
                    sleep_efficiency=0.65,
                    sleep_fragmentation_index=0.3,
                    sleep_sessions=3,
                    longest_sleep_hours=2.0,
                    earliest_bedtime=time(1, 0),
                    latest_wake_time=time(5, 30),
                    total_time_in_bed_hours=6.0,
                )
            )
        
        empty_activity = []
        empty_heart = []
        
        # Test hypersomnia
        hypersomnia_features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=hypersomnia_sleep,
            historical_activity=empty_activity,
            historical_heart=empty_heart,
            lookback_days=30,
        )
        
        # Test insomnia
        insomnia_features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=insomnia_sleep,
            historical_activity=empty_activity,
            historical_heart=empty_heart,
            lookback_days=30,
        )
        
        # Check clinical indicators
        assert hypersomnia_features.is_hypersomnia_pattern is True
        assert hypersomnia_features.is_insomnia_pattern is False
        assert hypersomnia_features.long_sleep_window_pct > 50
        
        assert insomnia_features.is_insomnia_pattern is True
        assert insomnia_features.is_hypersomnia_pattern is False
        assert insomnia_features.short_sleep_window_pct > 50
        
        # Both should have elevated mood risk scores
        assert hypersomnia_features.mood_risk_score > 0.1
        assert insomnia_features.mood_risk_score > 0.1

    def test_temporal_features(self):
        """Test rolling window temporal features."""
        engineer = AdvancedFeatureEngineer()
        current_date = date(2024, 1, 15)
        
        # Create variable sleep pattern
        sleep_data = []
        for i in range(14):
            # Alternate between short and long sleep
            hours = 6 if i % 2 == 0 else 9
            sleep_data.append(
                DailySleepSummary(
                    date=current_date - timedelta(days=i),
                    total_sleep_hours=hours,
                    sleep_efficiency=0.85,
                    sleep_fragmentation_index=0.1,
                    sleep_sessions=1,
                    earliest_bedtime=time(23, 0),
                    latest_wake_time=time(5 + hours % 12, 0),
                    total_time_in_bed_hours=hours,
                longest_sleep_hours=hours,
                    )
            )
        
        # Create variable activity pattern
        activity_data = []
        for i in range(14):
            # Alternate between low and high activity
            steps = 3000 if i % 2 == 0 else 15000
            activity_data.append(
                DailyActivitySummary(
                    date=current_date - timedelta(days=i),
                    total_steps=steps,
                    active_hours=0.5 if i % 2 == 0 else 90,
                    total_active_energy=100.0 if i % 2 == 0 else 400,
                    sedentary_hours=13.3 if i % 2 == 0 else 6.7,
                    activity_variance=100.0 if i % 2 == 0 else 2000.0,
                )
            )
        
        empty_heart = []
        
        features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=sleep_data,
            historical_activity=activity_data,
            historical_heart=empty_heart,
            lookback_days=30,
        )
        
        # Check 7-day windows capture variability
        assert features.sleep_7day_mean > 0
        assert features.sleep_7day_std > 1  # High variability
        assert features.activity_7day_mean > 0
        assert features.activity_7day_std > 1000  # High variability

    def test_individual_normalization(self):
        """Test Z-score normalization."""
        engineer = AdvancedFeatureEngineer()
        
        # Create consistent baseline data
        baseline_dates = [date(2024, 1, i) for i in range(1, 20)]
        baseline_sleep = []
        baseline_activity = []
        baseline_heart = []
        
        for d in baseline_dates:
            baseline_sleep.append(
                DailySleepSummary(
                    date=d,
                    total_sleep_hours=8.0,
                    sleep_efficiency=0.85,
                    sleep_fragmentation_index=0.1,
                    sleep_sessions=1,
                    earliest_bedtime=time(23, 0),
                    latest_wake_time=time(7, 0),
                    total_time_in_bed_hours=8.0,
                longest_sleep_hours=8.0,
                    )
            )
            
            baseline_activity.append(
                DailyActivitySummary(
                    date=d,
                    total_steps=8000,
                    active_hours=1.0,
                    total_active_energy=600.0,
                    sedentary_hours=10.0,
                    activity_variance=1000.0,
                )
            )
            
            baseline_heart.append(
                DailyHeartSummary(
                    date=d,
                    hr_measurements=100,
                    morning_hr=70.0,
                    min_hr=50.0,
                    max_hr=120.0,
                    evening_hr=10.0,
                    avg_resting_hr=60.0,
                    avg_hrv_sdnn=50.0,                    circadian_hr_range=15.0,
                )
            )
        
        # Extract features to establish baseline
        for d in baseline_dates:
            _ = engineer.extract_advanced_features(
                current_date=d,
                historical_sleep=baseline_sleep,
                historical_activity=baseline_activity,
                historical_heart=baseline_heart,
                lookback_days=30,
            )
        
        # Now add outlier day
        outlier_date = date(2024, 1, 25)
        outlier_sleep = baseline_sleep + [
            DailySleepSummary(
                date=outlier_date,
                total_sleep_hours=3.0,  # Very low
                sleep_efficiency=0.85,
                sleep_fragmentation_index=0.1,
                sleep_sessions=1,
                earliest_bedtime=time(3, 0),
                latest_wake_time=time(6, 0),
                total_time_in_bed_hours=3.0,
                longest_sleep_hours=3.0,
            )
        ]
        
        outlier_activity = baseline_activity + [
            DailyActivitySummary(
                date=outlier_date,
                total_steps=20000,  # Very high
                active_hours=3.0,
                total_active_energy=300.0,
                sedentary_hours=5.0,
                activity_variance=5000.0,
            )
        ]
        
        outlier_heart = baseline_heart + [
            DailyHeartSummary(
                date=outlier_date,
                hr_measurements=100,
                morning_hr=90.0,
                min_hr=70.0,
                max_hr=160.0,
                evening_hr=20.0,
                avg_resting_hr=80.0,  # Elevated
                avg_hrv_sdnn=20.0,  # Low
                circadian_hr_range=30.0,
            )
        ]
        
        outlier_features = engineer.extract_advanced_features(
            current_date=outlier_date,
            historical_sleep=outlier_sleep,
            historical_activity=outlier_activity,
            historical_heart=outlier_heart,
            lookback_days=30,
        )
        
        # Z-scores should reflect significant deviations
        assert outlier_features.sleep_duration_zscore < -2  # >2 SD below mean
        assert outlier_features.activity_zscore > 2  # >2 SD above mean
        assert outlier_features.hr_zscore > 1  # Elevated
        assert outlier_features.hrv_zscore < -1  # Reduced

    def test_empty_data_handling(self):
        """Test handling of missing data."""
        engineer = AdvancedFeatureEngineer()
        current_date = date(2024, 1, 15)
        
        # Test with empty data
        features = engineer.extract_advanced_features(
            current_date=current_date,
            historical_sleep=[],
            historical_activity=[],
            historical_heart=[],
            lookback_days=30,
        )
        
        # Should return features with zero/default values
        assert features.date == current_date
        assert features.sleep_duration_hours == 0
        assert features.total_steps == 0
        assert features.avg_resting_hr == 0
        
        # ML features should still be 36 elements
        ml_features = features.to_ml_features()
        assert len(ml_features) == 36
        # Features should be mostly zero, but some clinical indicators might be True (1.0)
        # when data is missing (e.g., is_irregular_pattern = True when no regular sleep data)
        import numpy as np
        non_zero_count = sum(1 for f in ml_features if f != 0 and f != 0.0 and not np.isnan(f))
        # We expect most features to be zero, but a few clinical indicators might be 1.0
        assert non_zero_count <= 5, f"Too many non-zero features: {non_zero_count}"