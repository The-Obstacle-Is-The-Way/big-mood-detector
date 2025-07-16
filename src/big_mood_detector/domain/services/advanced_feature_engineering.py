"""
Advanced Feature Engineering Service
Implements research-based features for clinical mood prediction.
Based on Seoul National, Harvard/Fitbit, and XGBoost studies.
"""

import numpy as np
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary
from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary


@dataclass
class AdvancedFeatures:
    """Complete feature set for mood prediction based on research."""
    
    # Basic features (already implemented)
    date: date
    sleep_duration_hours: float
    sleep_efficiency: float
    total_steps: int
    avg_resting_hr: float
    hrv_sdnn: float
    
    # Advanced Sleep Features (Seoul National Study)
    sleep_regularity_index: float  # 0-100, higher = more regular
    interdaily_stability: float  # 0-1, circadian rhythm stability
    intradaily_variability: float  # 0-2, fragmentation of rhythm
    relative_amplitude: float  # 0-1, strength of rhythm
    l5_value: float  # Least active 5 hours (activity)
    m10_value: float  # Most active 10 hours (activity)
    l5_onset: Optional[datetime]  # When L5 period starts
    m10_onset: Optional[datetime]  # When M10 period starts
    
    # Circadian Phase Features
    circadian_phase_advance: float  # Hours ahead of baseline
    circadian_phase_delay: float  # Hours behind baseline
    dlmo_estimate: Optional[datetime]  # Dim Light Melatonin Onset
    core_body_temp_nadir: Optional[datetime]  # Lowest temp time
    
    # Sleep Window Features
    short_sleep_window_pct: float  # % of nights < 6 hours
    long_sleep_window_pct: float  # % of nights > 10 hours
    sleep_onset_variance: float  # Variance in bedtime (hours²)
    wake_time_variance: float  # Variance in wake time (hours²)
    
    # Activity Pattern Features
    activity_fragmentation: float  # Transitions between active/inactive
    sedentary_bout_mean: float  # Average sedentary period length
    sedentary_bout_max: float  # Longest sedentary period
    activity_intensity_ratio: float  # High/low activity ratio
    
    # Individual Normalization (Z-scores)
    sleep_duration_zscore: float
    activity_zscore: float
    hr_zscore: float
    hrv_zscore: float
    
    # Temporal Features (rolling windows)
    sleep_7day_mean: float
    sleep_7day_std: float
    activity_7day_mean: float
    activity_7day_std: float
    hr_7day_mean: float
    hr_7day_std: float
    
    # Clinical Indicators
    is_hypersomnia_pattern: bool  # >10 hours consistently
    is_insomnia_pattern: bool  # <6 hours consistently
    is_phase_advanced: bool  # Early sleep/wake
    is_phase_delayed: bool  # Late sleep/wake
    is_irregular_pattern: bool  # High variability
    mood_risk_score: float  # 0-1, composite risk
    
    def to_ml_features(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        # 36 features as required by Seoul study
        return np.array([
            self.sleep_duration_hours,
            self.sleep_efficiency,
            self.sleep_regularity_index,
            self.interdaily_stability,
            self.intradaily_variability,
            self.relative_amplitude,
            self.l5_value,
            self.m10_value,
            self.circadian_phase_advance,
            self.circadian_phase_delay,
            self.short_sleep_window_pct,
            self.long_sleep_window_pct,
            self.sleep_onset_variance,
            self.wake_time_variance,
            self.total_steps,
            self.activity_fragmentation,
            self.sedentary_bout_mean,
            self.activity_intensity_ratio,
            self.avg_resting_hr,
            self.hrv_sdnn,
            self.sleep_duration_zscore,
            self.activity_zscore,
            self.hr_zscore,
            self.hrv_zscore,
            self.sleep_7day_mean,
            self.sleep_7day_std,
            self.activity_7day_mean,
            self.activity_7day_std,
            self.hr_7day_mean,
            self.hr_7day_std,
            float(self.is_hypersomnia_pattern),
            float(self.is_insomnia_pattern),
            float(self.is_phase_advanced),
            float(self.is_phase_delayed),
            float(self.is_irregular_pattern),
            self.mood_risk_score,
        ])


class AdvancedFeatureEngineer:
    """
    Implements advanced feature engineering for mood prediction.
    Based on peer-reviewed research methodologies.
    """
    
    def __init__(self):
        """Initialize with baseline statistics tracking."""
        self.individual_baselines: Dict[str, Dict[str, float]] = {}
        self.population_baselines: Dict[str, float] = {}
        
    def extract_advanced_features(
        self,
        current_date: date,
        historical_sleep: List[DailySleepSummary],
        historical_activity: List[DailyActivitySummary],
        historical_heart: List[DailyHeartSummary],
        lookback_days: int = 30,
    ) -> AdvancedFeatures:
        """
        Extract comprehensive feature set for a given date.
        
        Args:
            current_date: Date to extract features for
            historical_sleep: Past sleep summaries (including current)
            historical_activity: Past activity summaries
            historical_heart: Past heart summaries
            lookback_days: Days of history to consider
            
        Returns:
            AdvancedFeatures object with all calculated features
        """
        # Get recent data within lookback window
        recent_sleep = self._filter_recent(historical_sleep, current_date, lookback_days)
        recent_activity = self._filter_recent(historical_activity, current_date, lookback_days)
        recent_heart = self._filter_recent(historical_heart, current_date, lookback_days)
        
        # Get current day data
        current_sleep = self._get_current_day(recent_sleep, current_date)
        current_activity = self._get_current_day(recent_activity, current_date)
        current_heart = self._get_current_day(recent_heart, current_date)
        
        # Calculate advanced sleep features
        sleep_features = self._calculate_sleep_features(recent_sleep, current_sleep)
        
        # Calculate circadian features
        circadian_features = self._calculate_circadian_features(
            recent_sleep, recent_activity
        )
        
        # Calculate activity pattern features
        activity_features = self._calculate_activity_features(
            recent_activity, current_activity
        )
        
        # Calculate individual normalization
        norm_features = self._calculate_normalized_features(
            current_date, current_sleep, current_activity, current_heart
        )
        
        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(
            recent_sleep, recent_activity, recent_heart
        )
        
        # Calculate clinical indicators
        clinical_indicators = self._calculate_clinical_indicators(
            sleep_features, circadian_features, activity_features
        )
        
        # Combine all features
        return AdvancedFeatures(
            date=current_date,
            sleep_duration_hours=current_sleep.total_sleep_hours if current_sleep else 0,
            sleep_efficiency=current_sleep.sleep_efficiency if current_sleep else 0,
            total_steps=current_activity.total_steps if current_activity else 0,
            avg_resting_hr=current_heart.avg_resting_hr if current_heart else 0,
            hrv_sdnn=current_heart.hrv_sdnn if current_heart else 0,
            **sleep_features,
            **circadian_features,
            **activity_features,
            **norm_features,
            **temporal_features,
            **clinical_indicators,
        )
    
    def _calculate_sleep_features(
        self, recent_sleep: List[DailySleepSummary], current: Optional[DailySleepSummary]
    ) -> Dict[str, float]:
        """Calculate advanced sleep features based on Seoul study."""
        if not recent_sleep:
            return self._empty_sleep_features()
        
        # Sleep regularity index (0-100)
        sleep_times = [s.sleep_onset.hour + s.sleep_onset.minute/60 for s in recent_sleep]
        wake_times = [s.wake_time.hour + s.wake_time.minute/60 for s in recent_sleep]
        
        sleep_regularity = 100 - (np.std(sleep_times) + np.std(wake_times)) * 10
        sleep_regularity = max(0, min(100, sleep_regularity))
        
        # Interdaily stability (IS) - consistency across days
        # Uses non-parametric circadian rhythm analysis
        is_value = self._calculate_interdaily_stability(recent_sleep)
        
        # Intradaily variability (IV) - fragmentation within days
        iv_value = self._calculate_intradaily_variability(recent_sleep)
        
        # Relative amplitude (RA) - strength of rhythm
        ra_value = self._calculate_relative_amplitude(recent_sleep)
        
        # Sleep window analysis
        durations = [s.total_sleep_hours for s in recent_sleep]
        short_sleep_pct = sum(1 for d in durations if d < 6) / len(durations) * 100
        long_sleep_pct = sum(1 for d in durations if d > 10) / len(durations) * 100
        
        # Variance in timing
        sleep_onset_var = np.var(sleep_times)
        wake_time_var = np.var(wake_times)
        
        return {
            "sleep_regularity_index": sleep_regularity,
            "interdaily_stability": is_value,
            "intradaily_variability": iv_value,
            "relative_amplitude": ra_value,
            "short_sleep_window_pct": short_sleep_pct,
            "long_sleep_window_pct": long_sleep_pct,
            "sleep_onset_variance": sleep_onset_var,
            "wake_time_variance": wake_time_var,
        }
    
    def _calculate_circadian_features(
        self, recent_sleep: List[DailySleepSummary], recent_activity: List[DailyActivitySummary]
    ) -> Dict[str, float]:
        """Calculate circadian rhythm features."""
        if not recent_sleep:
            return self._empty_circadian_features()
        
        # L5 and M10 calculation (requires hourly activity data)
        # For now, estimate from daily patterns
        l5_value = min([a.sedentary_minutes for a in recent_activity] or [0])
        m10_value = max([a.active_minutes for a in recent_activity] or [0])
        
        # Circadian phase calculation
        # Compare to population average or individual baseline
        avg_sleep_time = np.mean([s.sleep_onset.hour + s.sleep_onset.minute/60 
                                  for s in recent_sleep])
        population_avg = 23.0  # 11 PM typical
        
        phase_shift = avg_sleep_time - population_avg
        phase_advance = max(0, -phase_shift)  # Earlier than normal
        phase_delay = max(0, phase_shift)  # Later than normal
        
        # DLMO estimation (2 hours before habitual sleep onset)
        dlmo_hour = (avg_sleep_time - 2) % 24
        dlmo_estimate = datetime.now().replace(
            hour=int(dlmo_hour), 
            minute=int((dlmo_hour % 1) * 60)
        )
        
        # Core body temperature nadir (2 hours before wake)
        avg_wake_time = np.mean([s.wake_time.hour + s.wake_time.minute/60 
                                for s in recent_sleep])
        temp_nadir_hour = (avg_wake_time - 2) % 24
        temp_nadir = datetime.now().replace(
            hour=int(temp_nadir_hour),
            minute=int((temp_nadir_hour % 1) * 60)
        )
        
        return {
            "l5_value": l5_value,
            "m10_value": m10_value,
            "l5_onset": None,  # Would need hourly data
            "m10_onset": None,  # Would need hourly data
            "circadian_phase_advance": phase_advance,
            "circadian_phase_delay": phase_delay,
            "dlmo_estimate": dlmo_estimate,
            "core_body_temp_nadir": temp_nadir,
        }
    
    def _calculate_activity_features(
        self, recent_activity: List[DailyActivitySummary], current: Optional[DailyActivitySummary]
    ) -> Dict[str, float]:
        """Calculate activity pattern features."""
        if not recent_activity:
            return self._empty_activity_features()
        
        # Activity fragmentation (transitions between active/sedentary)
        # Estimate from variance in daily patterns
        step_variance = np.var([a.total_steps for a in recent_activity])
        fragmentation = min(1.0, step_variance / 1000000)  # Normalize
        
        # Sedentary bout analysis
        sedentary_mins = [a.sedentary_minutes for a in recent_activity]
        bout_mean = np.mean(sedentary_mins)
        bout_max = max(sedentary_mins)
        
        # Activity intensity ratio
        # Estimate from step count distribution
        high_activity_days = sum(1 for a in recent_activity if a.total_steps > 10000)
        low_activity_days = sum(1 for a in recent_activity if a.total_steps < 5000)
        intensity_ratio = high_activity_days / (low_activity_days + 1)  # Avoid division by zero
        
        return {
            "activity_fragmentation": fragmentation,
            "sedentary_bout_mean": bout_mean,
            "sedentary_bout_max": bout_max,
            "activity_intensity_ratio": intensity_ratio,
        }
    
    def _calculate_normalized_features(
        self, current_date: date, sleep: Optional[DailySleepSummary],
        activity: Optional[DailyActivitySummary], heart: Optional[DailyHeartSummary]
    ) -> Dict[str, float]:
        """Calculate individual normalized features (Z-scores)."""
        # Update baselines
        self._update_individual_baseline("sleep", sleep.total_sleep_hours if sleep else 0)
        self._update_individual_baseline("activity", activity.total_steps if activity else 0)
        self._update_individual_baseline("hr", heart.avg_resting_hr if heart else 0)
        self._update_individual_baseline("hrv", heart.hrv_sdnn if heart else 0)
        
        # Calculate Z-scores
        sleep_z = self._calculate_zscore("sleep", sleep.total_sleep_hours if sleep else 0)
        activity_z = self._calculate_zscore("activity", activity.total_steps if activity else 0)
        hr_z = self._calculate_zscore("hr", heart.avg_resting_hr if heart else 0)
        hrv_z = self._calculate_zscore("hrv", heart.hrv_sdnn if heart else 0)
        
        return {
            "sleep_duration_zscore": sleep_z,
            "activity_zscore": activity_z,
            "hr_zscore": hr_z,
            "hrv_zscore": hrv_z,
        }
    
    def _calculate_temporal_features(
        self, recent_sleep: List[DailySleepSummary],
        recent_activity: List[DailyActivitySummary],
        recent_heart: List[DailyHeartSummary]
    ) -> Dict[str, float]:
        """Calculate rolling window temporal features."""
        # 7-day windows
        week_sleep = recent_sleep[-7:] if len(recent_sleep) >= 7 else recent_sleep
        week_activity = recent_activity[-7:] if len(recent_activity) >= 7 else recent_activity
        week_heart = recent_heart[-7:] if len(recent_heart) >= 7 else recent_heart
        
        # Sleep temporal features
        sleep_durations = [s.total_sleep_hours for s in week_sleep]
        sleep_7day_mean = np.mean(sleep_durations) if sleep_durations else 0
        sleep_7day_std = np.std(sleep_durations) if len(sleep_durations) > 1 else 0
        
        # Activity temporal features
        activity_steps = [a.total_steps for a in week_activity]
        activity_7day_mean = np.mean(activity_steps) if activity_steps else 0
        activity_7day_std = np.std(activity_steps) if len(activity_steps) > 1 else 0
        
        # Heart rate temporal features
        hr_values = [h.avg_resting_hr for h in week_heart]
        hr_7day_mean = np.mean(hr_values) if hr_values else 0
        hr_7day_std = np.std(hr_values) if len(hr_values) > 1 else 0
        
        return {
            "sleep_7day_mean": sleep_7day_mean,
            "sleep_7day_std": sleep_7day_std,
            "activity_7day_mean": activity_7day_mean,
            "activity_7day_std": activity_7day_std,
            "hr_7day_mean": hr_7day_mean,
            "hr_7day_std": hr_7day_std,
        }
    
    def _calculate_clinical_indicators(
        self, sleep_features: Dict, circadian_features: Dict, activity_features: Dict
    ) -> Dict[str, any]:
        """Calculate clinical risk indicators."""
        # Hypersomnia pattern (>10 hours consistently)
        is_hypersomnia = sleep_features.get("long_sleep_window_pct", 0) > 50
        
        # Insomnia pattern (<6 hours consistently)
        is_insomnia = sleep_features.get("short_sleep_window_pct", 0) > 50
        
        # Phase advanced (early sleep/wake)
        is_phase_advanced = circadian_features.get("circadian_phase_advance", 0) > 2
        
        # Phase delayed (late sleep/wake)
        is_phase_delayed = circadian_features.get("circadian_phase_delay", 0) > 2
        
        # Irregular pattern (high variability)
        is_irregular = sleep_features.get("sleep_regularity_index", 100) < 70
        
        # Composite mood risk score (0-1)
        risk_factors = [
            is_hypersomnia * 0.2,
            is_insomnia * 0.2,
            is_phase_advanced * 0.15,
            is_phase_delayed * 0.15,
            is_irregular * 0.3,
        ]
        mood_risk_score = sum(risk_factors)
        
        return {
            "is_hypersomnia_pattern": is_hypersomnia,
            "is_insomnia_pattern": is_insomnia,
            "is_phase_advanced": is_phase_advanced,
            "is_phase_delayed": is_phase_delayed,
            "is_irregular_pattern": is_irregular,
            "mood_risk_score": mood_risk_score,
        }
    
    # Helper methods
    def _filter_recent(self, summaries: List[any], current_date: date, days: int) -> List[any]:
        """Filter summaries to recent window."""
        cutoff = current_date - timedelta(days=days)
        return [s for s in summaries if s.date >= cutoff and s.date <= current_date]
    
    def _get_current_day(self, summaries: List[any], current_date: date) -> Optional[any]:
        """Get summary for specific date."""
        for s in summaries:
            if s.date == current_date:
                return s
        return None
    
    def _calculate_interdaily_stability(self, sleep_summaries: List[DailySleepSummary]) -> float:
        """Calculate IS using non-parametric circadian rhythm analysis."""
        # Simplified version - full implementation would use hourly data
        sleep_times = [s.sleep_onset.hour + s.sleep_onset.minute/60 for s in sleep_summaries]
        if len(sleep_times) < 3:
            return 0.0
        
        # Calculate variance ratio
        hourly_means = defaultdict(list)
        for i, time in enumerate(sleep_times):
            hour = i % 24
            hourly_means[hour].append(time)
        
        between_var = np.var([np.mean(times) for times in hourly_means.values() if times])
        total_var = np.var(sleep_times)
        
        return between_var / (total_var + 0.001)  # Avoid division by zero
    
    def _calculate_intradaily_variability(self, sleep_summaries: List[DailySleepSummary]) -> float:
        """Calculate IV - fragmentation of rhythm."""
        # Simplified - measures variability in sleep duration
        durations = [s.total_sleep_hours for s in sleep_summaries]
        if len(durations) < 2:
            return 0.0
        
        # First derivative squared
        diffs = np.diff(durations)
        iv = np.mean(diffs ** 2) / np.var(durations)
        
        return min(2.0, iv)  # Cap at 2.0
    
    def _calculate_relative_amplitude(self, sleep_summaries: List[DailySleepSummary]) -> float:
        """Calculate RA - strength of circadian rhythm."""
        if not sleep_summaries:
            return 0.0
        
        # Use sleep efficiency as proxy for rhythm strength
        efficiencies = [s.sleep_efficiency for s in sleep_summaries]
        
        # Most restful vs least restful periods
        sorted_eff = sorted(efficiencies)
        m10 = np.mean(sorted_eff[-10:]) if len(sorted_eff) >= 10 else np.mean(sorted_eff)
        l5 = np.mean(sorted_eff[:5]) if len(sorted_eff) >= 5 else np.mean(sorted_eff)
        
        return (m10 - l5) / (m10 + l5 + 0.001)  # Normalized difference
    
    def _update_individual_baseline(self, metric: str, value: float):
        """Update individual baseline statistics."""
        if metric not in self.individual_baselines:
            self.individual_baselines[metric] = {"values": [], "mean": 0, "std": 0}
        
        baseline = self.individual_baselines[metric]
        baseline["values"].append(value)
        
        # Keep last 30 days
        if len(baseline["values"]) > 30:
            baseline["values"].pop(0)
        
        # Update statistics
        if len(baseline["values"]) >= 3:
            baseline["mean"] = np.mean(baseline["values"])
            baseline["std"] = np.std(baseline["values"])
    
    def _calculate_zscore(self, metric: str, value: float) -> float:
        """Calculate Z-score relative to individual baseline."""
        if metric not in self.individual_baselines:
            return 0.0
        
        baseline = self.individual_baselines[metric]
        if baseline["std"] == 0:
            return 0.0
        
        return (value - baseline["mean"]) / baseline["std"]
    
    def _empty_sleep_features(self) -> Dict[str, float]:
        """Return empty sleep features."""
        return {
            "sleep_regularity_index": 0,
            "interdaily_stability": 0,
            "intradaily_variability": 0,
            "relative_amplitude": 0,
            "short_sleep_window_pct": 0,
            "long_sleep_window_pct": 0,
            "sleep_onset_variance": 0,
            "wake_time_variance": 0,
        }
    
    def _empty_circadian_features(self) -> Dict[str, any]:
        """Return empty circadian features."""
        return {
            "l5_value": 0,
            "m10_value": 0,
            "l5_onset": None,
            "m10_onset": None,
            "circadian_phase_advance": 0,
            "circadian_phase_delay": 0,
            "dlmo_estimate": None,
            "core_body_temp_nadir": None,
        }
    
    def _empty_activity_features(self) -> Dict[str, float]:
        """Return empty activity features."""
        return {
            "activity_fragmentation": 0,
            "sedentary_bout_mean": 0,
            "sedentary_bout_max": 0,
            "activity_intensity_ratio": 0,
        }