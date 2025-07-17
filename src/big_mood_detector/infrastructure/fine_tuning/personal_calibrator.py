"""
Personal Calibrator Module

User-level adaptation and baseline extraction for personalized mood predictions.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


class BaselineExtractor:
    """Extract personal baseline patterns from health data."""
    
    def __init__(self, baseline_window_days: int = 30, min_data_days: int = 14):
        """Initialize baseline extractor.
        
        Args:
            baseline_window_days: Number of days to use for baseline calculation
            min_data_days: Minimum days of data required for valid baseline
        """
        self.baseline_window_days = baseline_window_days
        self.min_data_days = min_data_days
    
    def extract_sleep_baseline(self, sleep_data: pd.DataFrame) -> Dict[str, float]:
        """Extract baseline sleep patterns from health data.
        
        Args:
            sleep_data: DataFrame with columns: date, sleep_duration, sleep_efficiency, sleep_onset
            
        Returns:
            Dictionary with baseline sleep metrics
        """
        # Use most recent baseline_window_days of data
        if len(sleep_data) > self.baseline_window_days:
            sleep_data = sleep_data.tail(self.baseline_window_days)
        
        baseline = {
            "mean_sleep_duration": sleep_data["sleep_duration"].mean(),
            "std_sleep_duration": sleep_data["sleep_duration"].std(),
            "mean_sleep_efficiency": sleep_data["sleep_efficiency"].mean(),
            "mean_sleep_onset": sleep_data["sleep_onset"].mean(),
        }
        
        return baseline
    
    def extract_activity_baseline(self, activity_data: pd.DataFrame) -> Dict[str, float]:
        """Extract baseline activity patterns from minute-level data.
        
        Args:
            activity_data: DataFrame with columns: date, activity
            
        Returns:
            Dictionary with baseline activity metrics
        """
        # Group by day and calculate daily totals
        activity_data['date_only'] = pd.to_datetime(activity_data['date']).dt.date
        daily_activity = activity_data.groupby('date_only')['activity'].sum()
        
        # Use most recent baseline_window_days
        if len(daily_activity) > self.baseline_window_days:
            daily_activity = daily_activity.tail(self.baseline_window_days)
            
        # Calculate hourly pattern for rhythm analysis
        activity_data['hour'] = pd.to_datetime(activity_data['date']).dt.hour
        hourly_pattern = activity_data.groupby('hour')['activity'].mean()
        
        # Find peak activity time
        peak_hour = hourly_pattern.idxmax()
        
        # Calculate activity amplitude (difference between most and least active hours)
        activity_amplitude = hourly_pattern.max() - hourly_pattern.min()
        
        baseline = {
            "mean_daily_activity": daily_activity.mean(),
            "activity_rhythm": hourly_pattern.std(),  # Variability across hours
            "peak_activity_time": float(peak_hour),
            "activity_amplitude": activity_amplitude,
        }
        
        return baseline
    
    def calculate_circadian_baseline(self, circadian_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate circadian rhythm baseline from hourly activity data.
        
        Args:
            circadian_data: DataFrame with columns: timestamp, activity
            
        Returns:
            Dictionary with circadian rhythm metrics
        """
        # Extract hour from timestamp
        circadian_data['hour'] = pd.to_datetime(circadian_data['timestamp']).dt.hour
        circadian_data['date'] = pd.to_datetime(circadian_data['timestamp']).dt.date
        
        # Calculate hourly averages across all days
        hourly_pattern = circadian_data.groupby('hour')['activity'].mean()
        
        # Find circadian phase (time of peak activity)
        circadian_phase = float(hourly_pattern.idxmax())
        
        # Calculate amplitude (difference between peak and trough)
        circadian_amplitude = hourly_pattern.max() - hourly_pattern.min()
        
        # Calculate stability (how consistent the pattern is across days)
        # Use coefficient of variation for each hour
        hourly_cv = circadian_data.groupby('hour')['activity'].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        circadian_stability = 1 - hourly_cv.mean()  # Higher value = more stable
        
        baseline = {
            "circadian_phase": circadian_phase,
            "circadian_amplitude": circadian_amplitude,
            "circadian_stability": circadian_stability,
        }
        
        return baseline


class EpisodeLabeler:
    """Label episodes and baseline periods for training."""
    
    def __init__(self):
        """Initialize episode labeler."""
        self.episodes = []
        self.baseline_periods = []


class PersonalCalibrator:
    """Calibrate models to individual users."""
    pass