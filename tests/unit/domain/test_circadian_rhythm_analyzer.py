"""
Unit tests for Circadian Rhythm Analyzer

Tests the calculation of circadian rhythm metrics (IS, IV, RA, L5/M10)
critical for bipolar disorder detection and phase analysis.
"""

from datetime import datetime, date, timedelta
import numpy as np
import pytest

from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.services.circadian_rhythm_analyzer import (
    CircadianRhythmAnalyzer,
    CircadianMetrics,
    L5M10Analysis
)
from big_mood_detector.domain.services.activity_sequence_extractor import MinuteLevelSequence


class TestCircadianRhythmAnalyzer:
    """Test circadian rhythm metric calculations."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default settings."""
        return CircadianRhythmAnalyzer()
    
    def _create_regular_pattern(self, base_date: date, days: int = 7) -> list[MinuteLevelSequence]:
        """Helper to create regular activity pattern."""
        sequences = []
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            activity_values = [0.0] * 1440
            
            # Regular pattern: active 8 AM - 10 PM
            for hour in range(8, 22):
                for minute in range(60):
                    idx = hour * 60 + minute
                    # Higher activity during day, with peak at noon
                    if 10 <= hour <= 14:
                        activity_values[idx] = 150.0
                    else:
                        activity_values[idx] = 100.0
            
            sequences.append(MinuteLevelSequence(
                date=current_date,
                activity_values=activity_values
            ))
        
        return sequences
    
    def _create_irregular_pattern(self, base_date: date, days: int = 7) -> list[MinuteLevelSequence]:
        """Helper to create irregular activity pattern."""
        sequences = []
        np.random.seed(42)
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            activity_values = [0.0] * 1440
            
            # Very irregular: completely different patterns each day
            if day % 3 == 0:
                # Night shift pattern
                for hour in range(20, 24):
                    for minute in range(60):
                        idx = hour * 60 + minute
                        activity_values[idx] = np.random.uniform(100, 200)
                for hour in range(0, 8):
                    for minute in range(60):
                        idx = hour * 60 + minute
                        activity_values[idx] = np.random.uniform(100, 200)
            elif day % 3 == 1:
                # Early morning pattern
                for hour in range(4, 12):
                    for minute in range(60):
                        idx = hour * 60 + minute
                        activity_values[idx] = np.random.uniform(50, 150)
            else:
                # Afternoon/evening pattern
                for hour in range(14, 22):
                    for minute in range(60):
                        idx = hour * 60 + minute
                        activity_values[idx] = np.random.uniform(75, 175)
            
            sequences.append(MinuteLevelSequence(
                date=current_date,
                activity_values=activity_values
            ))
        
        return sequences
    
    def test_interdaily_stability_regular_pattern(self, analyzer):
        """IS should be high (close to 1) for regular patterns."""
        # Arrange
        sequences = self._create_regular_pattern(date(2024, 1, 1), days=7)
        
        # Act
        is_score = analyzer.calculate_interdaily_stability(sequences)
        
        # Assert
        assert 0.8 <= is_score <= 1.0  # High stability
    
    def test_interdaily_stability_irregular_pattern(self, analyzer):
        """IS should be low for irregular patterns."""
        # Arrange
        sequences = self._create_irregular_pattern(date(2024, 1, 1), days=7)
        
        # Act
        is_score = analyzer.calculate_interdaily_stability(sequences)
        
        # Assert
        assert is_score < 0.5  # Low stability
    
    def test_intradaily_variability_smooth_pattern(self, analyzer):
        """IV should be low for smooth activity transitions."""
        # Arrange - smooth transitions
        sequences = self._create_regular_pattern(date(2024, 1, 1), days=7)
        
        # Act
        iv_score = analyzer.calculate_intradaily_variability(sequences)
        
        # Assert
        assert iv_score < 1.0  # Low variability (smooth)
    
    def test_intradaily_variability_fragmented_pattern(self, analyzer):
        """IV should be high for fragmented activity."""
        # Arrange - create fragmented pattern
        sequences = []
        for day in range(7):
            current_date = date(2024, 1, 1) + timedelta(days=day)
            activity_values = [0.0] * 1440
            
            # Fragmented: alternating high/low every hour
            for hour in range(8, 20):
                for minute in range(60):
                    idx = hour * 60 + minute
                    if hour % 2 == 0:
                        activity_values[idx] = 200.0
                    else:
                        activity_values[idx] = 50.0
            
            sequences.append(MinuteLevelSequence(
                date=current_date,
                activity_values=activity_values
            ))
        
        # Act
        iv_score = analyzer.calculate_intradaily_variability(sequences)
        
        # Assert
        assert iv_score > 1.5  # High variability (fragmented)
    
    def test_relative_amplitude_strong_rhythm(self, analyzer):
        """RA should be high for strong day/night differences."""
        # Arrange - strong day/night rhythm
        sequences = self._create_regular_pattern(date(2024, 1, 1), days=7)
        
        # Act
        ra_score = analyzer.calculate_relative_amplitude(sequences)
        
        # Assert
        assert 0.8 <= ra_score <= 1.0  # Strong rhythm
    
    def test_relative_amplitude_weak_rhythm(self, analyzer):
        """RA should be low for weak day/night differences."""
        # Arrange - constant low activity
        sequences = []
        for day in range(7):
            current_date = date(2024, 1, 1) + timedelta(days=day)
            # Constant low activity all day
            activity_values = [50.0] * 1440
            
            sequences.append(MinuteLevelSequence(
                date=current_date,
                activity_values=activity_values
            ))
        
        # Act
        ra_score = analyzer.calculate_relative_amplitude(sequences)
        
        # Assert
        assert ra_score < 0.2  # Weak rhythm
    
    def test_l5_m10_calculation(self, analyzer):
        """Test L5 (least active 5 hours) and M10 (most active 10 hours)."""
        # Arrange - clear pattern
        sequences = []
        for day in range(7):
            current_date = date(2024, 1, 1) + timedelta(days=day)
            activity_values = [0.0] * 1440
            
            # Night: 12 AM - 6 AM (very low)
            for hour in range(0, 6):
                for minute in range(60):
                    idx = hour * 60 + minute
                    activity_values[idx] = 10.0
            
            # Day: 10 AM - 8 PM (high)
            for hour in range(10, 20):
                for minute in range(60):
                    idx = hour * 60 + minute
                    activity_values[idx] = 200.0
            
            sequences.append(MinuteLevelSequence(
                date=current_date,
                activity_values=activity_values
            ))
        
        # Act
        l5m10 = analyzer.calculate_l5_m10(sequences)
        
        # Assert
        assert l5m10.l5_value < 1200  # Low activity in L5 (less than 20/min avg)
        assert l5m10.m10_value > 10000  # High activity in M10 (>160/min avg)
        assert 0 <= l5m10.l5_start_hour < 6  # L5 during night
        assert 10 <= l5m10.m10_start_hour < 20  # M10 during day
    
    def test_l5_timing_consistency(self, analyzer):
        """L5 timing should be consistent for regular patterns."""
        # Arrange
        sequences = self._create_regular_pattern(date(2024, 1, 1), days=14)
        
        # Act
        l5m10 = analyzer.calculate_l5_m10(sequences)
        consistency = analyzer.calculate_l5_timing_consistency(sequences)
        
        # Assert
        assert consistency > 0.8  # High consistency for regular pattern
    
    def test_comprehensive_metrics(self, analyzer):
        """Test all metrics calculated together."""
        # Arrange
        sequences = self._create_regular_pattern(date(2024, 1, 1), days=7)
        
        # Act
        metrics = analyzer.calculate_metrics(sequences)
        
        # Assert
        assert isinstance(metrics, CircadianMetrics)
        assert 0.0 <= metrics.interdaily_stability <= 1.0
        assert metrics.intradaily_variability >= 0.0
        assert 0.0 <= metrics.relative_amplitude <= 1.0
        assert metrics.l5_value >= 0.0
        assert metrics.m10_value >= metrics.l5_value
        assert 0 <= metrics.l5_start_hour < 24
        assert 0 <= metrics.m10_start_hour < 24
        assert 0.0 <= metrics.l5_timing_consistency <= 1.0
    
    def test_insufficient_data_handling(self, analyzer):
        """Should handle insufficient data gracefully."""
        # Arrange - only 2 days
        sequences = self._create_regular_pattern(date(2024, 1, 1), days=2)
        
        # Act
        metrics = analyzer.calculate_metrics(sequences)
        
        # Assert
        assert metrics is not None
        # With limited data, some metrics may be less reliable
        assert metrics.interdaily_stability >= 0
    
    def test_phase_shift_detection(self, analyzer):
        """Should detect phase shifts in activity patterns."""
        # Arrange - create phase shift
        sequences = []
        for day in range(14):
            current_date = date(2024, 1, 1) + timedelta(days=day)
            activity_values = [0.0] * 1440
            
            # Shift active period later each week
            if day < 7:
                active_start = 8  # Week 1: 8 AM start
            else:
                active_start = 12  # Week 2: 12 PM start (4h shift)
            
            for hour in range(active_start, min(active_start + 10, 24)):
                for minute in range(60):
                    idx = hour * 60 + minute
                    if idx < 1440:
                        activity_values[idx] = 150.0
            
            sequences.append(MinuteLevelSequence(
                date=current_date,
                activity_values=activity_values
            ))
        
        # Act
        phase_shift = analyzer.detect_phase_shift(sequences[:7], sequences[7:])
        
        # Assert
        assert phase_shift >= 3.0  # Detected ~4 hour shift
    
    def test_circadian_disruption_score(self, analyzer):
        """Test overall circadian disruption scoring."""
        # Arrange - disrupted pattern
        sequences = self._create_irregular_pattern(date(2024, 1, 1), days=7)
        
        # Act
        disruption_score = analyzer.calculate_disruption_score(sequences)
        
        # Assert
        assert 0.0 <= disruption_score <= 1.0
        assert disruption_score > 0.4  # Moderate-high disruption for irregular pattern