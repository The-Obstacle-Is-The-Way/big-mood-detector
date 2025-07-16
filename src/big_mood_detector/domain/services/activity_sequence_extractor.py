"""
Activity Sequence Extractor Service

Extracts minute-level activity sequences for circadian rhythm analysis.
Critical for PAT (Principal Activity Time) calculation in bipolar disorder detection.

Design Patterns:
- Builder Pattern: Builds minute-level sequences incrementally
- Strategy Pattern: Different smoothing and analysis strategies
- Value Objects: Immutable sequence and analysis results
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import List, Optional, Tuple
import numpy as np

from big_mood_detector.domain.entities.activity_record import ActivityRecord


@dataclass(frozen=True)
class MinuteLevelSequence:
    """
    Immutable value object representing 24-hour minute-level activity.
    
    Contains 1440 values (24 * 60) representing activity for each minute.
    """
    date: date
    activity_values: List[float]  # Length 1440
    
    @property
    def total_activity(self) -> float:
        """Sum of all activity values."""
        return sum(self.activity_values)
    
    @property
    def active_minutes(self) -> int:
        """Count of minutes with non-zero activity."""
        return sum(1 for v in self.activity_values if v > 0)
    
    @property
    def max_minute_activity(self) -> float:
        """Maximum activity in any single minute."""
        return max(self.activity_values) if self.activity_values else 0
    
    def get_hour_totals(self) -> List[float]:
        """Get total activity for each hour (24 values)."""
        hour_totals = []
        for hour in range(24):
            start_idx = hour * 60
            end_idx = start_idx + 60
            hour_totals.append(sum(self.activity_values[start_idx:end_idx]))
        return hour_totals
    
    def get_percentile(self, percentile: float) -> float:
        """Get activity value at given percentile."""
        if not self.activity_values:
            return 0
        sorted_values = sorted(self.activity_values)
        idx = int(len(sorted_values) * percentile / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]
    
    def get_smoothed_values(self, window_size: int = 5) -> List[float]:
        """Get moving average smoothed values."""
        if window_size <= 1:
            return self.activity_values.copy()
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(self.activity_values)):
            start = max(0, i - half_window)
            end = min(len(self.activity_values), i + half_window + 1)
            window_values = self.activity_values[start:end]
            smoothed.append(sum(window_values) / len(window_values))
        
        return smoothed


@dataclass(frozen=True)
class PATAnalysisResult:
    """
    Principal Activity Time analysis results.
    
    PAT is the weighted center of daily activity, indicating
    when the person is most active (circadian phase marker).
    """
    pat_hour: float  # Principal activity time (0-23.99)
    pat_minute: int  # PAT in minutes from midnight
    morning_activity: float  # Activity 6 AM - 12 PM
    afternoon_activity: float  # Activity 12 PM - 6 PM
    evening_activity: float  # Activity 6 PM - 12 AM
    night_activity: float  # Activity 12 AM - 6 AM
    activity_concentration: float  # 0-1, how concentrated activity is
    peak_activity_minutes: int  # Minutes in peak activity period
    is_evening_type: bool  # True if evening > morning activity
    

class ActivitySequenceExtractor:
    """
    Extracts and analyzes minute-level activity sequences.
    
    Following Single Responsibility Principle: only handles
    sequence extraction and basic time-series analysis.
    """
    
    MINUTES_PER_DAY = 1440
    MINUTES_PER_HOUR = 60
    
    def extract_daily_sequence(
        self, 
        records: List[ActivityRecord], 
        target_date: date
    ) -> MinuteLevelSequence:
        """
        Extract minute-level activity sequence for a specific date.
        
        Args:
            records: Activity records to process
            target_date: Date to extract sequence for
            
        Returns:
            Minute-level sequence with 1440 values
        """
        # Initialize empty sequence
        activity_values = [0.0] * self.MINUTES_PER_DAY
        
        # Filter records for target date
        date_records = [
            r for r in records 
            if r.timestamp.date() == target_date
        ]
        
        # Aggregate by minute
        for record in date_records:
            minute_index = self._get_minute_index(record.timestamp)
            if 0 <= minute_index < self.MINUTES_PER_DAY:
                activity_values[minute_index] += record.step_count
        
        return MinuteLevelSequence(
            date=target_date,
            activity_values=activity_values
        )
    
    def calculate_pat(
        self, 
        records: List[ActivityRecord], 
        target_date: date
    ) -> PATAnalysisResult:
        """
        Calculate Principal Activity Time and related metrics.
        
        PAT is the weighted center of mass of daily activity,
        indicating circadian phase preference.
        
        Args:
            records: Activity records to analyze
            target_date: Date to analyze
            
        Returns:
            PAT analysis results
        """
        sequence = self.extract_daily_sequence(records, target_date)
        
        # Calculate time-weighted center of activity
        total_weighted = 0.0
        total_activity = 0.0
        
        for minute, activity in enumerate(sequence.activity_values):
            if activity > 0:
                total_weighted += minute * activity
                total_activity += activity
        
        # Calculate PAT
        if total_activity > 0:
            pat_minute = int(total_weighted / total_activity)
            pat_hour = pat_minute / 60.0
        else:
            pat_minute = 0
            pat_hour = 0.0
        
        # Calculate period activities
        morning = self._calculate_period_activity(sequence, 6, 12)
        afternoon = self._calculate_period_activity(sequence, 12, 18)
        evening = self._calculate_period_activity(sequence, 18, 24)
        night = self._calculate_period_activity(sequence, 0, 6)
        
        # Calculate concentration (how focused activity is)
        concentration = self._calculate_activity_concentration(sequence)
        
        # Find peak activity period
        peak_minutes = self._find_peak_activity_duration(sequence)
        
        return PATAnalysisResult(
            pat_hour=pat_hour,
            pat_minute=pat_minute,
            morning_activity=morning,
            afternoon_activity=afternoon,
            evening_activity=evening,
            night_activity=night,
            activity_concentration=concentration,
            peak_activity_minutes=peak_minutes,
            is_evening_type=(evening > morning)
        )
    
    def calculate_circadian_alignment(
        self,
        records: List[ActivityRecord],
        target_date: date,
        sleep_start_hour: int = 23,
        sleep_end_hour: int = 7
    ) -> float:
        """
        Calculate how well activity aligns with expected sleep/wake cycle.
        
        Args:
            records: Activity records
            target_date: Date to analyze
            sleep_start_hour: Expected sleep start (24h)
            sleep_end_hour: Expected wake time (24h)
            
        Returns:
            Alignment score 0-1 (1 = perfect alignment)
        """
        sequence = self.extract_daily_sequence(records, target_date)
        
        # Calculate activity during sleep hours
        sleep_activity = 0.0
        wake_activity = 0.0
        
        for minute, activity in enumerate(sequence.activity_values):
            hour = minute // 60
            
            # Handle sleep period crossing midnight
            if sleep_start_hour > sleep_end_hour:
                is_sleep_time = hour >= sleep_start_hour or hour < sleep_end_hour
            else:
                is_sleep_time = sleep_start_hour <= hour < sleep_end_hour
            
            if is_sleep_time:
                sleep_activity += activity
            else:
                wake_activity += activity
        
        # Calculate alignment score
        total_activity = sleep_activity + wake_activity
        if total_activity == 0:
            return 1.0  # No activity = perfect alignment
        
        # Penalize activity during sleep hours
        sleep_ratio = sleep_activity / total_activity
        alignment_score = 1.0 - sleep_ratio
        
        return max(0.0, min(1.0, alignment_score))
    
    def _get_minute_index(self, timestamp: datetime) -> int:
        """Convert timestamp to minute index (0-1439)."""
        return timestamp.hour * 60 + timestamp.minute
    
    def _calculate_period_activity(
        self, 
        sequence: MinuteLevelSequence, 
        start_hour: int, 
        end_hour: int
    ) -> float:
        """Calculate total activity in hour period."""
        start_idx = start_hour * 60
        end_idx = end_hour * 60
        return sum(sequence.activity_values[start_idx:end_idx])
    
    def _calculate_activity_concentration(
        self, 
        sequence: MinuteLevelSequence
    ) -> float:
        """
        Calculate how concentrated activity is (0-1).
        
        Uses coefficient of variation approach.
        High concentration = activity in short bursts.
        Low concentration = activity spread throughout day.
        """
        if sequence.total_activity == 0:
            return 0.0
        
        # Get hourly distribution
        hour_totals = sequence.get_hour_totals()
        active_hours = [h for h in hour_totals if h > 0]
        
        if not active_hours:
            return 0.0
        
        # Calculate concentration using Gini coefficient
        sorted_hours = sorted(active_hours)
        n = len(sorted_hours)
        cumsum = 0.0
        
        for i, value in enumerate(sorted_hours):
            cumsum += (2 * (i + 1) - n - 1) * value
        
        gini = cumsum / (n * sum(sorted_hours))
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, gini))
    
    def _find_peak_activity_duration(
        self, 
        sequence: MinuteLevelSequence
    ) -> int:
        """Find duration of peak activity period in minutes."""
        if sequence.total_activity == 0:
            return 0
        
        # Find 90th percentile threshold
        threshold = sequence.get_percentile(90)
        
        # Count consecutive minutes above threshold
        max_duration = 0
        current_duration = 0
        
        for activity in sequence.activity_values:
            if activity >= threshold:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration