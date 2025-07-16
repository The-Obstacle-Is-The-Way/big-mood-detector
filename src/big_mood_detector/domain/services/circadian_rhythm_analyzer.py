"""
Circadian Rhythm Analyzer Service

Calculates circadian rhythm metrics (IS, IV, RA, L5/M10) for bipolar disorder detection.
Based on non-parametric circadian rhythm analysis from actigraphy research.

Design Principles:
- Pure domain logic (no external dependencies)
- Immutable value objects for thread safety
- Single Responsibility: Only circadian analysis
"""

from dataclasses import dataclass
from datetime import date, time
from typing import List, Optional, Tuple
import math

from big_mood_detector.domain.services.activity_sequence_extractor import MinuteLevelSequence


@dataclass(frozen=True)
class CircadianMetrics:
    """
    Immutable circadian rhythm metrics.
    
    Based on Van Someren et al. (1999) non-parametric methods.
    """
    interdaily_stability: float  # IS: 0-1, rhythm consistency across days
    intradaily_variability: float  # IV: 0-2+, fragmentation within days
    relative_amplitude: float  # RA: 0-1, difference between active/rest
    l5_value: float  # Least active 5 hours average
    m10_value: float  # Most active 10 hours average
    l5_start_hour: int  # Hour when L5 period starts (0-23)
    m10_start_hour: int  # Hour when M10 period starts (0-23)
    l5_timing_consistency: float  # 0-1, how consistent L5 timing is


@dataclass(frozen=True)
class L5M10Analysis:
    """Details of L5 (least active 5h) and M10 (most active 10h) periods."""
    l5_value: float
    l5_start_hour: int
    l5_end_hour: int
    m10_value: float
    m10_start_hour: int
    m10_end_hour: int


class CircadianRhythmAnalyzer:
    """
    Analyzes circadian rhythm patterns from activity data.
    
    Implements non-parametric circadian rhythm analysis methods
    validated for bipolar disorder detection.
    """
    
    MINUTES_PER_HOUR = 60
    HOURS_PER_DAY = 24
    MINUTES_PER_DAY = 1440
    L5_HOURS = 5  # Least active 5 hours
    M10_HOURS = 10  # Most active 10 hours
    
    def calculate_metrics(self, sequences: List[MinuteLevelSequence]) -> CircadianMetrics:
        """
        Calculate comprehensive circadian rhythm metrics.
        
        Args:
            sequences: Daily activity sequences (minimum 3 days recommended)
            
        Returns:
            Complete circadian metrics
        """
        if len(sequences) < 3:
            # Return default metrics for insufficient data
            return self._default_metrics()
        
        is_score = self.calculate_interdaily_stability(sequences)
        iv_score = self.calculate_intradaily_variability(sequences)
        ra_score = self.calculate_relative_amplitude(sequences)
        l5m10 = self.calculate_l5_m10(sequences)
        l5_consistency = self.calculate_l5_timing_consistency(sequences)
        
        return CircadianMetrics(
            interdaily_stability=is_score,
            intradaily_variability=iv_score,
            relative_amplitude=ra_score,
            l5_value=l5m10.l5_value,
            m10_value=l5m10.m10_value,
            l5_start_hour=l5m10.l5_start_hour,
            m10_start_hour=l5m10.m10_start_hour,
            l5_timing_consistency=l5_consistency
        )
    
    def calculate_interdaily_stability(self, sequences: List[MinuteLevelSequence]) -> float:
        """
        Calculate Interdaily Stability (IS).
        
        IS quantifies the invariability between days, indicating how well
        the circadian rhythm is synchronized to the 24-hour light-dark cycle.
        
        Range: 0-1 (higher = more stable/regular)
        """
        if len(sequences) < 2:
            return 0.0
        
        # Create average 24-hour profile
        hourly_averages = self._calculate_average_hourly_profile(sequences)
        
        # Calculate variance of the average profile
        profile_mean = sum(hourly_averages) / len(hourly_averages)
        profile_variance = sum((x - profile_mean) ** 2 for x in hourly_averages) / len(hourly_averages)
        
        # Calculate total variance across all data
        all_hourly_values = []
        for seq in sequences:
            hour_totals = seq.get_hour_totals()
            all_hourly_values.extend(hour_totals)
        
        total_mean = sum(all_hourly_values) / len(all_hourly_values)
        total_variance = sum((x - total_mean) ** 2 for x in all_hourly_values) / len(all_hourly_values)
        
        # IS = variance of average profile / total variance
        if total_variance == 0:
            return 1.0  # Perfect stability (no activity)
        
        is_score = profile_variance / total_variance
        return max(0.0, min(1.0, is_score))
    
    def calculate_intradaily_variability(self, sequences: List[MinuteLevelSequence]) -> float:
        """
        Calculate Intradaily Variability (IV).
        
        IV quantifies the fragmentation of the rhythm, indicating the
        frequency and extent of transitions between rest and activity.
        
        Range: 0-2 (lower = less fragmented, more consolidated)
        """
        if not sequences:
            return 0.0
        
        # Combine all hourly data
        all_hourly_values = []
        for seq in sequences:
            hour_totals = seq.get_hour_totals()
            all_hourly_values.extend(hour_totals)
        
        if len(all_hourly_values) < 2:
            return 0.0
        
        # Calculate first derivative (hourly differences)
        differences = []
        for i in range(1, len(all_hourly_values)):
            diff = all_hourly_values[i] - all_hourly_values[i-1]
            differences.append(diff ** 2)
        
        # Calculate mean square of differences
        mean_sq_diff = sum(differences) / len(differences)
        
        # Calculate overall variance
        mean_activity = sum(all_hourly_values) / len(all_hourly_values)
        variance = sum((x - mean_activity) ** 2 for x in all_hourly_values) / len(all_hourly_values)
        
        # IV = mean square successive difference / variance
        if variance == 0:
            return 0.0
        
        iv_score = mean_sq_diff / variance
        return iv_score
    
    def calculate_relative_amplitude(self, sequences: List[MinuteLevelSequence]) -> float:
        """
        Calculate Relative Amplitude (RA).
        
        RA indicates the difference between the most and least active periods,
        normalized by their sum. Higher values indicate stronger rhythms.
        
        Range: 0-1 (higher = stronger rhythm)
        """
        if not sequences:
            return 0.0
        
        l5m10 = self.calculate_l5_m10(sequences)
        
        # RA = (M10 - L5) / (M10 + L5)
        if l5m10.m10_value + l5m10.l5_value == 0:
            return 0.0
        
        ra = (l5m10.m10_value - l5m10.l5_value) / (l5m10.m10_value + l5m10.l5_value)
        return max(0.0, min(1.0, ra))
    
    def calculate_l5_m10(self, sequences: List[MinuteLevelSequence]) -> L5M10Analysis:
        """
        Calculate L5 (least active 5 hours) and M10 (most active 10 hours).
        
        These periods don't have to be sleep/wake but represent the most
        and least active continuous periods in the 24-hour cycle.
        """
        if not sequences:
            return L5M10Analysis(0, 0, 5, 0, 8, 18)
        
        # Create average 24-hour profile
        hourly_averages = self._calculate_average_hourly_profile(sequences)
        
        # Find L5 (least active 5 consecutive hours)
        l5_value = float('inf')
        l5_start = 0
        
        for start_hour in range(24):
            # Calculate 5-hour window average (with wraparound)
            window_sum = 0
            for i in range(self.L5_HOURS):
                hour = (start_hour + i) % 24
                window_sum += hourly_averages[hour]
            
            window_avg = window_sum / self.L5_HOURS
            if window_avg < l5_value:
                l5_value = window_avg
                l5_start = start_hour
        
        # Find M10 (most active 10 consecutive hours)
        m10_value = 0
        m10_start = 0
        
        for start_hour in range(24):
            # Calculate 10-hour window average (with wraparound)
            window_sum = 0
            for i in range(self.M10_HOURS):
                hour = (start_hour + i) % 24
                window_sum += hourly_averages[hour]
            
            window_avg = window_sum / self.M10_HOURS
            if window_avg > m10_value:
                m10_value = window_avg
                m10_start = start_hour
        
        return L5M10Analysis(
            l5_value=l5_value,
            l5_start_hour=l5_start,
            l5_end_hour=(l5_start + self.L5_HOURS) % 24,
            m10_value=m10_value,
            m10_start_hour=m10_start,
            m10_end_hour=(m10_start + self.M10_HOURS) % 24
        )
    
    def calculate_l5_timing_consistency(self, sequences: List[MinuteLevelSequence]) -> float:
        """
        Calculate how consistent the L5 (rest) period timing is across days.
        
        High consistency indicates regular sleep patterns.
        Low consistency may indicate circadian disruption.
        
        Range: 0-1 (higher = more consistent)
        """
        if len(sequences) < 3:
            return 0.5  # Neutral score for insufficient data
        
        # Calculate L5 start hour for each day
        l5_starts = []
        for seq in sequences:
            # Find L5 for this specific day
            hour_totals = seq.get_hour_totals()
            
            min_avg = float('inf')
            min_start = 0
            
            for start in range(24):
                window_sum = sum(hour_totals[(start + i) % 24] for i in range(self.L5_HOURS))
                window_avg = window_sum / self.L5_HOURS
                
                if window_avg < min_avg:
                    min_avg = window_avg
                    min_start = start
            
            l5_starts.append(min_start)
        
        # Calculate circular variance of L5 start times
        # Convert to radians (24h = 2π)
        angles = [start * 2 * math.pi / 24 for start in l5_starts]
        
        # Calculate mean angle
        sin_sum = sum(math.sin(a) for a in angles)
        cos_sum = sum(math.cos(a) for a in angles)
        
        # Calculate resultant vector length
        r = math.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
        
        # Consistency score (0-1, higher = more consistent)
        return r
    
    def detect_phase_shift(
        self, 
        week1_sequences: List[MinuteLevelSequence],
        week2_sequences: List[MinuteLevelSequence]
    ) -> float:
        """
        Detect phase shift between two periods.
        
        Returns hours of phase shift (positive = delayed, negative = advanced).
        """
        if not week1_sequences or not week2_sequences:
            return 0.0
        
        # Calculate M10 start for each period
        l5m10_week1 = self.calculate_l5_m10(week1_sequences)
        l5m10_week2 = self.calculate_l5_m10(week2_sequences)
        
        # Calculate phase shift
        shift = l5m10_week2.m10_start_hour - l5m10_week1.m10_start_hour
        
        # Adjust for wraparound (e.g., 23 to 1 = +2, not -22)
        if shift > 12:
            shift -= 24
        elif shift < -12:
            shift += 24
        
        return shift
    
    def calculate_disruption_score(self, sequences: List[MinuteLevelSequence]) -> float:
        """
        Calculate overall circadian disruption score.
        
        Combines multiple metrics into a single disruption indicator.
        
        Range: 0-1 (higher = more disrupted)
        """
        metrics = self.calculate_metrics(sequences)
        
        # Normalize each component to 0-1 (bad to good)
        is_normalized = metrics.interdaily_stability  # Already 0-1, high is good
        iv_normalized = 1.0 - min(metrics.intradaily_variability / 2.0, 1.0)  # High is bad
        ra_normalized = metrics.relative_amplitude  # Already 0-1, high is good
        consistency_normalized = metrics.l5_timing_consistency  # Already 0-1, high is good
        
        # Combine (equal weights for now)
        good_rhythm_score = (
            is_normalized + 
            iv_normalized + 
            ra_normalized + 
            consistency_normalized
        ) / 4.0
        
        # Disruption is inverse of good rhythm
        disruption_score = 1.0 - good_rhythm_score
        
        return max(0.0, min(1.0, disruption_score))
    
    def _calculate_average_hourly_profile(self, sequences: List[MinuteLevelSequence], per_minute: bool = False) -> List[float]:
        """Calculate average activity for each hour across all days."""
        if not sequences:
            return [0.0] * 24
        
        hourly_sums = [0.0] * 24
        
        for seq in sequences:
            hour_totals = seq.get_hour_totals()
            for hour, total in enumerate(hour_totals):
                hourly_sums[hour] += total
        
        # Calculate averages
        num_days = len(sequences)
        hourly_averages = [total / num_days for total in hourly_sums]
        
        # If per_minute, divide by 60 to get average activity per minute
        if per_minute:
            hourly_averages = [avg / 60 for avg in hourly_averages]
        
        return hourly_averages
    
    def _default_metrics(self) -> CircadianMetrics:
        """Return default metrics when insufficient data."""
        return CircadianMetrics(
            interdaily_stability=0.5,
            intradaily_variability=1.0,
            relative_amplitude=0.5,
            l5_value=0.0,
            m10_value=0.0,
            l5_start_hour=0,
            m10_start_hour=8,
            l5_timing_consistency=0.5
        )