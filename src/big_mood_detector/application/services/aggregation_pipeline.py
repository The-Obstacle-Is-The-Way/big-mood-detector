"""
Aggregation Pipeline Service

Handles all feature aggregation and statistical calculations for mood prediction.
Extracted from MoodPredictionPipeline to follow Single Responsibility Principle.

Design Patterns:
- Pipeline Pattern: Sequential processing of aggregation steps
- Builder Pattern: Building complex feature sets incrementally
- Template Method: Common aggregation structure with customizable steps
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Protocol

import numpy as np

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
)
from big_mood_detector.domain.services.circadian_rhythm_analyzer import (
    CircadianRhythmAnalyzer,
)
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator
from big_mood_detector.domain.services.sleep_window_analyzer import SleepWindowAnalyzer


@dataclass
class AggregationConfig:
    """Configuration for aggregation pipeline."""
    
    window_size: int = 30  # Rolling window size for statistics
    min_window_size: int = 3  # Minimum data points for statistics
    enable_parallel: bool = False  # Parallel processing support
    lookback_days_circadian: int = 7  # Days for circadian analysis
    lookback_days_dlmo: int = 14  # Days for DLMO calculation


@dataclass
class DailyMetrics:
    """Raw metrics for a single day."""
    
    date: date
    sleep: dict[str, float]
    circadian: dict[str, float] | None = None
    activity: dict[str, float] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'date': self.date,
            'sleep': self.sleep
        }
        if self.circadian:
            result['circadian'] = self.circadian
        if self.activity:
            result['activity'] = self.activity
        return result


@dataclass
class DailyFeatures:
    """
    Complete feature set for one day (36 features).
    
    Matches the Seoul study's XGBoost input format.
    """
    
    date: date
    
    # Sleep features (10 × 3 = 30)
    sleep_percentage_mean: float
    sleep_percentage_std: float
    sleep_percentage_zscore: float
    
    sleep_amplitude_mean: float
    sleep_amplitude_std: float
    sleep_amplitude_zscore: float
    
    long_sleep_num_mean: float
    long_sleep_num_std: float
    long_sleep_num_zscore: float
    
    long_sleep_len_mean: float
    long_sleep_len_std: float
    long_sleep_len_zscore: float
    
    long_sleep_st_mean: float
    long_sleep_st_std: float
    long_sleep_st_zscore: float
    
    long_sleep_wt_mean: float
    long_sleep_wt_std: float
    long_sleep_wt_zscore: float
    
    short_sleep_num_mean: float
    short_sleep_num_std: float
    short_sleep_num_zscore: float
    
    short_sleep_len_mean: float
    short_sleep_len_std: float
    short_sleep_len_zscore: float
    
    short_sleep_st_mean: float
    short_sleep_st_std: float
    short_sleep_st_zscore: float
    
    short_sleep_wt_mean: float
    short_sleep_wt_std: float
    short_sleep_wt_zscore: float
    
    # Circadian features (2 × 3 = 6)
    circadian_amplitude_mean: float
    circadian_amplitude_std: float
    circadian_amplitude_zscore: float
    
    circadian_phase_mean: float  # DLMO hour
    circadian_phase_std: float
    circadian_phase_zscore: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "date": self.date,
            "sleep_percentage_MN": self.sleep_percentage_mean,
            "sleep_percentage_SD": self.sleep_percentage_std,
            "sleep_percentage_Z": self.sleep_percentage_zscore,
            "sleep_amplitude_MN": self.sleep_amplitude_mean,
            "sleep_amplitude_SD": self.sleep_amplitude_std,
            "sleep_amplitude_Z": self.sleep_amplitude_zscore,
            "long_num_MN": self.long_sleep_num_mean,
            "long_num_SD": self.long_sleep_num_std,
            "long_num_Z": self.long_sleep_num_zscore,
            "long_len_MN": self.long_sleep_len_mean,
            "long_len_SD": self.long_sleep_len_std,
            "long_len_Z": self.long_sleep_len_zscore,
            "long_ST_MN": self.long_sleep_st_mean,
            "long_ST_SD": self.long_sleep_st_std,
            "long_ST_Z": self.long_sleep_st_zscore,
            "long_WT_MN": self.long_sleep_wt_mean,
            "long_WT_SD": self.long_sleep_wt_std,
            "long_WT_Z": self.long_sleep_wt_zscore,
            "short_num_MN": self.short_sleep_num_mean,
            "short_num_SD": self.short_sleep_num_std,
            "short_num_Z": self.short_sleep_num_zscore,
            "short_len_MN": self.short_sleep_len_mean,
            "short_len_SD": self.short_sleep_len_std,
            "short_len_Z": self.short_sleep_len_zscore,
            "short_ST_MN": self.short_sleep_st_mean,
            "short_ST_SD": self.short_sleep_st_std,
            "short_ST_Z": self.short_sleep_st_zscore,
            "short_WT_MN": self.short_sleep_wt_mean,
            "short_WT_SD": self.short_sleep_wt_std,
            "short_WT_Z": self.short_sleep_wt_zscore,
            "circadian_amplitude_MN": self.circadian_amplitude_mean,
            "circadian_amplitude_SD": self.circadian_amplitude_std,
            "circadian_amplitude_Z": self.circadian_amplitude_zscore,
            "circadian_phase_MN": self.circadian_phase_mean,
            "circadian_phase_SD": self.circadian_phase_std,
            "circadian_phase_Z": self.circadian_phase_zscore,
        }


class AggregationPipeline:
    """
    Service responsible for feature aggregation and statistical calculations.
    
    This service encapsulates:
    - Daily feature extraction
    - Metric calculation (sleep, circadian, activity)
    - Statistical aggregation (mean, std, z-score)
    - Rolling window management
    - Feature normalization
    """
    
    def __init__(
        self,
        config: AggregationConfig | None = None,
        sleep_analyzer: SleepWindowAnalyzer | None = None,
        activity_extractor: ActivitySequenceExtractor | None = None,
        circadian_analyzer: CircadianRhythmAnalyzer | None = None,
        dlmo_calculator: DLMOCalculator | None = None,
    ):
        """
        Initialize with dependencies.
        
        Args:
            config: Aggregation configuration
            sleep_analyzer: Sleep window analysis service
            activity_extractor: Activity sequence extraction service
            circadian_analyzer: Circadian rhythm analysis service
            dlmo_calculator: DLMO calculation service
        """
        self.config = config or AggregationConfig()
        self.sleep_analyzer = sleep_analyzer or SleepWindowAnalyzer()
        self.activity_extractor = activity_extractor or ActivitySequenceExtractor()
        self.circadian_analyzer = circadian_analyzer or CircadianRhythmAnalyzer()
        self.dlmo_calculator = dlmo_calculator or DLMOCalculator()
        
        # Default normalization parameters
        self._default_normalization_params = {
            'sleep_percentage': {'min': 0.0, 'max': 0.5},
            'sleep_amplitude': {'min': 0.0, 'max': 1.0},
            'long_num': {'min': 0, 'max': 5},
            'short_num': {'min': 0, 'max': 10},
            'circadian_amplitude': {'min': 0.0, 'max': 1.0},
            'circadian_phase': {'min': 0.0, 'max': 24.0},
        }
    
    @property
    def supports_parallel_processing(self) -> bool:
        """Check if parallel processing is supported."""
        return self.config.enable_parallel
    
    def aggregate_daily_features(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
        start_date: date,
        end_date: date,
        min_window_size: int | None = None,
        parallel: bool = False,
    ) -> list[DailyFeatures]:
        """
        Aggregate features for a date range.
        
        Args:
            sleep_records: Sleep records
            activity_records: Activity records
            heart_records: Heart rate records
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            min_window_size: Minimum window size for statistics
            parallel: Whether to use parallel processing
            
        Returns:
            List of daily features
        """
        # Process each day
        daily_features = []
        current_date = start_date
        
        # Keep rolling windows for statistics
        sleep_metrics_window: list[dict[str, float]] = []
        circadian_metrics_window: list[dict[str, float]] = []
        
        window_size = min_window_size or self.config.min_window_size
        
        while current_date <= end_date:
            # 1. Sleep Window Analysis
            day_sleep = [
                s
                for s in sleep_records
                if s.start_date.date() <= current_date <= s.end_date.date()
            ]
            
            sleep_windows = self.sleep_analyzer.analyze_sleep_episodes(
                day_sleep, current_date
            )
            
            # 2. Activity Sequence Extraction
            day_activity = [
                a for a in activity_records if a.start_date.date() == current_date
            ]
            
            activity_sequence = None
            if day_activity:
                activity_sequence = self.activity_extractor.extract_daily_sequence(
                    day_activity, current_date
                )
            
            # 3. Circadian Rhythm Analysis
            circadian_metrics = self._calculate_circadian_metrics(
                activity_records, current_date
            )
            
            # 4. DLMO Calculation
            dlmo_result = self._calculate_dlmo(sleep_records, current_date)
            
            # 5. Extract daily metrics
            daily_metrics = self.calculate_daily_metrics(
                sleep_windows, activity_sequence, circadian_metrics, dlmo_result
            )
            
            # 6. Update rolling windows
            if daily_metrics:
                if 'sleep' in daily_metrics:
                    sleep_metrics_window.append(daily_metrics['sleep'])
                    if len(sleep_metrics_window) > self.config.window_size:
                        sleep_metrics_window.pop(0)
                
                if 'circadian' in daily_metrics and daily_metrics['circadian']:
                    circadian_metrics_window.append(daily_metrics['circadian'])
                    if len(circadian_metrics_window) > self.config.window_size:
                        circadian_metrics_window.pop(0)
            
            # 7. Calculate statistics if we have enough data
            if len(sleep_metrics_window) >= window_size:
                features = self._calculate_features_with_stats(
                    current_date,
                    daily_metrics,
                    sleep_metrics_window,
                    circadian_metrics_window,
                )
                
                if features:
                    daily_features.append(features)
            
            current_date += timedelta(days=1)
        
        return daily_features
    
    def calculate_daily_metrics(
        self,
        sleep_windows: Any,
        activity_sequence: Any,
        circadian_metrics: Any,
        dlmo_result: Any,
    ) -> dict[str, Any]:
        """
        Calculate metrics for a single day.
        
        Args:
            sleep_windows: Sleep window analysis results
            activity_sequence: Activity sequence for the day
            circadian_metrics: Circadian rhythm metrics
            dlmo_result: DLMO calculation result
            
        Returns:
            Dictionary with sleep and circadian metrics
        """
        metrics: dict[str, Any] = {}
        
        # Calculate sleep metrics
        if sleep_windows:
            metrics['sleep'] = self.calculate_sleep_metrics(sleep_windows)
        
        # Calculate circadian metrics if available
        if circadian_metrics and dlmo_result:
            metrics['circadian'] = {
                'amplitude': circadian_metrics.relative_amplitude,
                'phase': dlmo_result.dlmo_hour,
            }
        
        return metrics
    
    def calculate_sleep_metrics(self, sleep_windows: list[Any]) -> dict[str, float]:
        """
        Calculate sleep metrics from sleep windows.
        
        Args:
            sleep_windows: List of sleep window objects
            
        Returns:
            Dictionary with sleep metrics
        """
        # Sleep percentage (% of day)
        total_sleep_minutes = sum(w.total_duration_hours * 60 for w in sleep_windows)
        sleep_percentage = total_sleep_minutes / 1440.0
        
        # Sleep amplitude (coefficient of variation of wake amounts)
        wake_periods = [g for w in sleep_windows for g in w.gap_hours if g > 0]
        if wake_periods:
            sleep_amplitude = np.std(wake_periods) / np.mean(wake_periods)
        else:
            sleep_amplitude = 0.0
        
        # Long/short window counts
        long_windows = [w for w in sleep_windows if w.total_duration_hours >= 3.75]
        short_windows = [w for w in sleep_windows if w.total_duration_hours < 3.75]
        
        return {
            'sleep_percentage': sleep_percentage,
            'sleep_amplitude': sleep_amplitude,
            'long_num': len(long_windows),
            'long_len': sum(w.total_duration_hours for w in long_windows),
            'long_st': sum(w.total_duration_hours for w in long_windows),  # Simplified
            'long_wt': sum(sum(w.gap_hours) for w in long_windows),
            'short_num': len(short_windows),
            'short_len': sum(w.total_duration_hours for w in short_windows),
            'short_st': sum(w.total_duration_hours for w in short_windows),  # Simplified
            'short_wt': sum(sum(w.gap_hours) for w in short_windows),
        }
    
    def calculate_statistics(
        self,
        metric_name: str,
        window_values: list[float],
        current_value: float,
    ) -> dict[str, float]:
        """
        Calculate statistical measures for a metric.
        
        Args:
            metric_name: Name of the metric
            window_values: Historical values in the window
            current_value: Current value to compare
            
        Returns:
            Dictionary with mean, std, and z-score
        """
        mean_val = np.mean(window_values)
        std_val = np.std(window_values)
        
        # Calculate z-score
        if std_val > 0:
            zscore = (current_value - mean_val) / std_val
        else:
            zscore = 0.0
        
        return {
            'mean': mean_val,
            'std': std_val,
            'zscore': zscore,
        }
    
    def create_rolling_window(self, size: int) -> list[dict[str, Any]]:
        """
        Create a new rolling window.
        
        Args:
            size: Maximum window size
            
        Returns:
            Empty list for the window
        """
        return []
    
    def update_rolling_window(
        self,
        window: list[dict[str, Any]],
        metrics: dict[str, Any],
        size: int | None = None,
    ) -> None:
        """
        Update rolling window with new metrics.
        
        Args:
            window: Rolling window to update
            metrics: New metrics to add
            size: Optional custom window size (overrides config)
        """
        window.append(metrics)
        
        # Use provided size or default from config
        max_size = size if size is not None else self.config.window_size
        
        # Keep only the configured window size
        while len(window) > max_size:
            window.pop(0)
    
    def aggregate_circadian_window(
        self,
        activity_sequences: list[Any],
        lookback_days: int,
    ) -> list[Any]:
        """
        Aggregate activity sequences for circadian analysis.
        
        Args:
            activity_sequences: List of daily activity sequences
            lookback_days: Number of days to look back
            
        Returns:
            Window of activity sequences
        """
        # Return the most recent sequences up to lookback_days
        return activity_sequences[-lookback_days:]
    
    def aggregate_dlmo_window(
        self,
        sleep_records: list[Any],
        target_date: date,
        lookback_days: int,
    ) -> list[Any]:
        """
        Aggregate sleep records for DLMO calculation.
        
        Args:
            sleep_records: All sleep records
            target_date: Target date for DLMO
            lookback_days: Number of days to look back
            
        Returns:
            Window of sleep records
        """
        # Filter records within the lookback window
        filtered = [
            r for r in sleep_records
            if (target_date - r.start_date.date()).days < lookback_days
            and r.start_date.date() <= target_date
        ]
        
        return filtered
    
    def normalize_features(
        self,
        features: dict[str, float],
        normalization_params: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """
        Normalize features to standard ranges.
        
        Args:
            features: Raw feature values
            normalization_params: Optional custom normalization parameters
            
        Returns:
            Normalized features
        """
        params = normalization_params or self._default_normalization_params
        normalized = {}
        
        for key, value in features.items():
            # Extract base feature name (remove _mean, _std, _zscore suffixes)
            base_name = key.rsplit('_', 1)[0] if '_' in key else key
            
            if base_name in params:
                min_val = params[base_name]['min']
                max_val = params[base_name]['max']
                
                # Clamp and normalize
                clamped = max(min_val, min(value, max_val))
                if max_val > min_val:
                    normalized[key] = (clamped - min_val) / (max_val - min_val)
                else:
                    normalized[key] = 0.0
            else:
                # No normalization for z-scores and unknown features
                normalized[key] = value
        
        return normalized
    
    def export_to_dataframe(self, features: list[DailyFeatures]) -> list[dict[str, Any]]:
        """
        Export features to DataFrame-ready format.
        
        Args:
            features: List of daily features
            
        Returns:
            List of dictionaries ready for DataFrame
        """
        return [f.to_dict() for f in features]
    
    def _calculate_circadian_metrics(
        self,
        activity_records: list[ActivityRecord],
        target_date: date,
    ) -> Any:
        """Calculate circadian metrics for a date."""
        # Get activity sequences for the past week
        sequences = []
        
        for days_back in range(self.config.lookback_days_circadian):
            seq_date = target_date - timedelta(days=days_back)
            day_activity = [
                a for a in activity_records if a.start_date.date() == seq_date
            ]
            
            if day_activity:
                seq = self.activity_extractor.extract_daily_sequence(
                    day_activity, seq_date
                )
                sequences.append(seq)
        
        # Calculate metrics if we have enough data
        if len(sequences) >= 3:
            return self.circadian_analyzer.calculate_metrics(sequences)
        
        return None
    
    def _calculate_dlmo(
        self,
        sleep_records: list[SleepRecord],
        target_date: date,
    ) -> Any:
        """Calculate DLMO for a date."""
        # Get sleep records for the past 2 weeks
        dlmo_sleep = [
            s
            for s in sleep_records
            if (target_date - s.start_date.date()).days < self.config.lookback_days_dlmo
            and s.start_date.date() <= target_date
        ]
        
        # Calculate DLMO if we have enough data
        if len(dlmo_sleep) >= 3:
            return self.dlmo_calculator.calculate_dlmo(
                sleep_records=dlmo_sleep,
                target_date=target_date,
                days_to_model=min(7, len(dlmo_sleep)),
            )
        
        return None
    
    def _calculate_features_with_stats(
        self,
        current_date: date,
        daily_metrics: dict[str, Any],
        sleep_window: list[dict[str, float]],
        circadian_window: list[dict[str, float]],
    ) -> DailyFeatures | None:
        """Calculate features with mean, std, and z-scores."""
        if not daily_metrics or 'sleep' not in daily_metrics:
            return None
        
        # Calculate sleep statistics
        sleep_features = {}
        for metric in [
            'sleep_percentage',
            'sleep_amplitude',
            'long_num',
            'long_len',
            'long_st',
            'long_wt',
            'short_num',
            'short_len',
            'short_st',
            'short_wt',
        ]:
            values = [s[metric] for s in sleep_window]
            current_val = daily_metrics['sleep'][metric]
            
            stats = self.calculate_statistics(metric, values, current_val)
            sleep_features[f'{metric}_mean'] = stats['mean']
            sleep_features[f'{metric}_std'] = stats['std']
            sleep_features[f'{metric}_zscore'] = stats['zscore']
        
        # Calculate circadian statistics
        circadian_features = {
            'circadian_amplitude_mean': 0.0,
            'circadian_amplitude_std': 0.0,
            'circadian_amplitude_zscore': 0.0,
            'circadian_phase_mean': 0.0,
            'circadian_phase_std': 0.0,
            'circadian_phase_zscore': 0.0,
        }
        
        if circadian_window and 'circadian' in daily_metrics and daily_metrics['circadian']:
            for metric in ['amplitude', 'phase']:
                values = [c[metric] for c in circadian_window if c]
                if values:
                    current_val = daily_metrics['circadian'][metric]
                    stats = self.calculate_statistics(metric, values, current_val)
                    
                    circadian_features[f'circadian_{metric}_mean'] = stats['mean']
                    circadian_features[f'circadian_{metric}_std'] = stats['std']
                    circadian_features[f'circadian_{metric}_zscore'] = stats['zscore']
        
        # Create DailyFeatures object
        return DailyFeatures(
            date=current_date,
            # Sleep percentage
            sleep_percentage_mean=sleep_features['sleep_percentage_mean'],
            sleep_percentage_std=sleep_features['sleep_percentage_std'],
            sleep_percentage_zscore=sleep_features['sleep_percentage_zscore'],
            # Sleep amplitude
            sleep_amplitude_mean=sleep_features['sleep_amplitude_mean'],
            sleep_amplitude_std=sleep_features['sleep_amplitude_std'],
            sleep_amplitude_zscore=sleep_features['sleep_amplitude_zscore'],
            # Long sleep
            long_sleep_num_mean=sleep_features['long_num_mean'],
            long_sleep_num_std=sleep_features['long_num_std'],
            long_sleep_num_zscore=sleep_features['long_num_zscore'],
            long_sleep_len_mean=sleep_features['long_len_mean'],
            long_sleep_len_std=sleep_features['long_len_std'],
            long_sleep_len_zscore=sleep_features['long_len_zscore'],
            long_sleep_st_mean=sleep_features['long_st_mean'],
            long_sleep_st_std=sleep_features['long_st_std'],
            long_sleep_st_zscore=sleep_features['long_st_zscore'],
            long_sleep_wt_mean=sleep_features['long_wt_mean'],
            long_sleep_wt_std=sleep_features['long_wt_std'],
            long_sleep_wt_zscore=sleep_features['long_wt_zscore'],
            # Short sleep
            short_sleep_num_mean=sleep_features['short_num_mean'],
            short_sleep_num_std=sleep_features['short_num_std'],
            short_sleep_num_zscore=sleep_features['short_num_zscore'],
            short_sleep_len_mean=sleep_features['short_len_mean'],
            short_sleep_len_std=sleep_features['short_len_std'],
            short_sleep_len_zscore=sleep_features['short_len_zscore'],
            short_sleep_st_mean=sleep_features['short_st_mean'],
            short_sleep_st_std=sleep_features['short_st_std'],
            short_sleep_st_zscore=sleep_features['short_st_zscore'],
            short_sleep_wt_mean=sleep_features['short_wt_mean'],
            short_sleep_wt_std=sleep_features['short_wt_std'],
            short_sleep_wt_zscore=sleep_features['short_wt_zscore'],
            # Circadian
            circadian_amplitude_mean=circadian_features['circadian_amplitude_mean'],
            circadian_amplitude_std=circadian_features['circadian_amplitude_std'],
            circadian_amplitude_zscore=circadian_features['circadian_amplitude_zscore'],
            circadian_phase_mean=circadian_features['circadian_phase_mean'],
            circadian_phase_std=circadian_features['circadian_phase_std'],
            circadian_phase_zscore=circadian_features['circadian_phase_zscore'],
        )