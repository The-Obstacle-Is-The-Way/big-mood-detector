"""
Advanced Feature Engineering Service
Implements research-based features for clinical mood prediction.
Based on Seoul National, Harvard/Fitbit, and XGBoost studies.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
from structlog import get_logger

logger = get_logger()

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.activity_feature_calculator import (
    ActivityFeatureCalculator,
)
from big_mood_detector.domain.services.circadian_feature_calculator import (
    CircadianFeatureCalculator,
)
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary
from big_mood_detector.domain.services.sleep_feature_calculator import (
    SleepFeatureCalculator,
)
from big_mood_detector.domain.services.temporal_feature_calculator import (
    TemporalFeatureCalculator,
)


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
    l5_onset: datetime | None  # When L5 period starts
    m10_onset: datetime | None  # When M10 period starts

    # Circadian Phase Features
    circadian_phase_advance: float  # Hours ahead of baseline
    circadian_phase_delay: float  # Hours behind baseline
    dlmo_estimate: datetime | None  # Dim Light Melatonin Onset
    core_body_temp_nadir: datetime | None  # Lowest temp time

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
        return np.array(
            [
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
            ]
        )


class AdvancedFeatureEngineer:
    """
    Implements advanced feature engineering for mood prediction.
    Based on peer-reviewed research methodologies.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        baseline_repository: Any = None,  # BaselineRepositoryInterface
        user_id: str | None = None,
    ) -> None:
        """Initialize with baseline statistics tracking.

        Args:
            config: Optional configuration dictionary
            baseline_repository: Optional repository for baseline persistence
            user_id: Optional user ID for loading/saving baselines
        """
        self.individual_baselines: dict[str, dict[str, Any]] = {}
        self.population_baselines: dict[str, float] = {}
        self.baseline_repository = baseline_repository
        self.user_id = user_id
        self._loaded_baseline = None  # Track loaded baseline for data_points accumulation

        # Initialize specialized calculators with config
        self.sleep_calculator = SleepFeatureCalculator(config)
        self.circadian_calculator = CircadianFeatureCalculator(config)
        self.activity_calculator = ActivityFeatureCalculator(config)
        self.temporal_calculator = TemporalFeatureCalculator(config)

        # Load existing baselines if repository and user_id provided
        if self.baseline_repository and self.user_id:
            self._load_baselines_from_repository()

    def extract_advanced_features(
        self,
        current_date: date,
        historical_sleep: list[DailySleepSummary],
        historical_activity: list[DailyActivitySummary],
        historical_heart: list[DailyHeartSummary],
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
        recent_sleep = self._filter_recent(
            historical_sleep, current_date, lookback_days
        )
        recent_activity = self._filter_recent(
            historical_activity, current_date, lookback_days
        )
        recent_heart = self._filter_recent(
            historical_heart, current_date, lookback_days
        )

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
            sleep_duration_hours=(
                current_sleep.total_sleep_hours if current_sleep else 0
            ),
            sleep_efficiency=current_sleep.sleep_efficiency if current_sleep else 0,
            total_steps=current_activity.total_steps if current_activity else 0,
            avg_resting_hr=current_heart.avg_resting_hr if current_heart else 0,
            hrv_sdnn=current_heart.avg_hrv_sdnn if current_heart else 0,
            # Sleep features
            sleep_regularity_index=sleep_features["sleep_regularity_index"],
            interdaily_stability=sleep_features["interdaily_stability"],
            intradaily_variability=sleep_features["intradaily_variability"],
            relative_amplitude=sleep_features["relative_amplitude"],
            short_sleep_window_pct=sleep_features["short_sleep_window_pct"],
            long_sleep_window_pct=sleep_features["long_sleep_window_pct"],
            sleep_onset_variance=sleep_features["sleep_onset_variance"],
            wake_time_variance=sleep_features["wake_time_variance"],
            # Circadian features
            l5_value=circadian_features["l5_value"],
            m10_value=circadian_features["m10_value"],
            l5_onset=circadian_features["l5_onset"],
            m10_onset=circadian_features["m10_onset"],
            circadian_phase_advance=circadian_features["circadian_phase_advance"],
            circadian_phase_delay=circadian_features["circadian_phase_delay"],
            dlmo_estimate=circadian_features["dlmo_estimate"],
            core_body_temp_nadir=circadian_features["core_body_temp_nadir"],
            # Activity features
            activity_fragmentation=activity_features["activity_fragmentation"],
            sedentary_bout_mean=activity_features["sedentary_bout_mean"],
            sedentary_bout_max=activity_features["sedentary_bout_max"],
            activity_intensity_ratio=activity_features["activity_intensity_ratio"],
            # Normalization features
            sleep_duration_zscore=norm_features["sleep_duration_zscore"],
            activity_zscore=norm_features["activity_zscore"],
            hr_zscore=norm_features["hr_zscore"],
            hrv_zscore=norm_features["hrv_zscore"],
            # Temporal features
            sleep_7day_mean=temporal_features["sleep_7day_mean"],
            sleep_7day_std=temporal_features["sleep_7day_std"],
            activity_7day_mean=temporal_features["activity_7day_mean"],
            activity_7day_std=temporal_features["activity_7day_std"],
            hr_7day_mean=temporal_features["hr_7day_mean"],
            hr_7day_std=temporal_features["hr_7day_std"],
            # Clinical indicators
            is_hypersomnia_pattern=clinical_indicators["is_hypersomnia_pattern"],
            is_insomnia_pattern=clinical_indicators["is_insomnia_pattern"],
            is_phase_advanced=clinical_indicators["is_phase_advanced"],
            is_phase_delayed=clinical_indicators["is_phase_delayed"],
            is_irregular_pattern=clinical_indicators["is_irregular_pattern"],
            mood_risk_score=clinical_indicators["mood_risk_score"],
        )

    def _calculate_sleep_features(
        self, recent_sleep: list[DailySleepSummary], current: DailySleepSummary | None
    ) -> dict[str, float]:
        """Calculate advanced sleep features based on Seoul study."""
        if not recent_sleep:
            return self._empty_sleep_features()

        # Delegate to specialized calculator
        sleep_regularity = self.sleep_calculator.calculate_regularity_index(
            recent_sleep
        )
        is_value = self.sleep_calculator.calculate_interdaily_stability(recent_sleep)
        iv_value = self.sleep_calculator.calculate_intradaily_variability(recent_sleep)
        ra_value = self.sleep_calculator.calculate_relative_amplitude(recent_sleep)
        short_sleep_pct, long_sleep_pct = (
            self.sleep_calculator.calculate_sleep_window_percentages(recent_sleep)
        )
        sleep_onset_var, wake_time_var = (
            self.sleep_calculator.calculate_timing_variances(recent_sleep)
        )

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
        self,
        recent_sleep: list[DailySleepSummary],
        recent_activity: list[DailyActivitySummary],
    ) -> dict[str, Any]:
        """Calculate circadian rhythm features."""
        if not recent_sleep:
            return self._empty_circadian_features()

        # Delegate to specialized calculator
        l5_m10_result = self.circadian_calculator.calculate_l5_m10_metrics(
            recent_activity
        )
        phase_result = self.circadian_calculator.calculate_phase_shifts(recent_sleep)
        dlmo_estimate = self.circadian_calculator.estimate_dlmo(recent_sleep)
        temp_nadir = self.circadian_calculator.estimate_core_temp_nadir(recent_sleep)

        return {
            "l5_value": l5_m10_result.l5_value,
            "m10_value": l5_m10_result.m10_value,
            "l5_onset": l5_m10_result.l5_onset,
            "m10_onset": l5_m10_result.m10_onset,
            "circadian_phase_advance": phase_result.phase_advance_hours,
            "circadian_phase_delay": phase_result.phase_delay_hours,
            "dlmo_estimate": dlmo_estimate,
            "core_body_temp_nadir": temp_nadir,
        }

    def _calculate_activity_features(
        self,
        recent_activity: list[DailyActivitySummary],
        current: DailyActivitySummary | None,
    ) -> dict[str, float]:
        """Calculate activity pattern features."""
        if not recent_activity:
            return self._empty_activity_features()

        # Delegate to specialized calculator
        fragmentation = self.activity_calculator.calculate_activity_fragmentation(
            recent_activity
        )
        bout_mean, bout_max, _ = self.activity_calculator.calculate_sedentary_bouts(
            recent_activity
        )
        intensity_metrics = (
            self.activity_calculator.calculate_activity_intensity_metrics(
                recent_activity
            )
        )

        return {
            "activity_fragmentation": fragmentation,
            "sedentary_bout_mean": bout_mean,
            "sedentary_bout_max": bout_max,
            "activity_intensity_ratio": intensity_metrics.intensity_ratio,
        }

    def _calculate_normalized_features(
        self,
        current_date: date,
        sleep: DailySleepSummary | None,
        activity: DailyActivitySummary | None,
        heart: DailyHeartSummary | None,
    ) -> dict[str, float]:
        """Calculate individual normalized features (Z-scores)."""
        # Update baselines
        sleep_hours = sleep.total_sleep_hours if sleep else 0
        logger.debug(
            "baseline_update_called",
            sleep_hours=sleep_hours,
            has_sleep=sleep is not None,
            user_id=self.user_id,
            date=str(current_date),
        )
        self._update_individual_baseline("sleep", sleep_hours)
        self._update_individual_baseline(
            "activity", activity.total_steps if activity else 0
        )
        self._update_individual_baseline("hr", heart.avg_resting_hr if heart else 0)
        self._update_individual_baseline("hrv", heart.avg_hrv_sdnn if heart else 0)

        # Calculate Z-scores
        sleep_z = self._calculate_zscore(
            "sleep", sleep.total_sleep_hours if sleep else 0
        )
        activity_z = self._calculate_zscore(
            "activity", activity.total_steps if activity else 0
        )
        hr_z = self._calculate_zscore("hr", heart.avg_resting_hr if heart else 0)
        hrv_z = self._calculate_zscore("hrv", heart.avg_hrv_sdnn if heart else 0)

        return {
            "sleep_duration_zscore": sleep_z,
            "activity_zscore": activity_z,
            "hr_zscore": hr_z,
            "hrv_zscore": hrv_z,
        }

    def _calculate_temporal_features(
        self,
        recent_sleep: list[DailySleepSummary],
        recent_activity: list[DailyActivitySummary],
        recent_heart: list[DailyHeartSummary],
    ) -> dict[str, float]:
        """Calculate rolling window temporal features."""
        # Delegate to specialized calculator

        # Sleep temporal features (7-day window)
        sleep_stats = self.temporal_calculator.calculate_rolling_statistics(
            recent_sleep, window_days=7, metric_extractor=lambda s: s.total_sleep_hours
        )

        # Activity temporal features (7-day window)
        activity_stats = self.temporal_calculator.calculate_rolling_statistics(
            recent_activity, window_days=7, metric_extractor=lambda a: a.total_steps
        )

        # Heart rate temporal features (7-day window)
        hr_stats = self.temporal_calculator.calculate_rolling_statistics(
            recent_heart, window_days=7, metric_extractor=lambda h: h.avg_resting_hr
        )

        return {
            "sleep_7day_mean": sleep_stats.mean,
            "sleep_7day_std": sleep_stats.std,
            "activity_7day_mean": activity_stats.mean,
            "activity_7day_std": activity_stats.std,
            "hr_7day_mean": hr_stats.mean,
            "hr_7day_std": hr_stats.std,
        }

    def _calculate_clinical_indicators(
        self, sleep_features: dict, circadian_features: dict, activity_features: dict
    ) -> dict[str, Any]:
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
    def _filter_recent(
        self, summaries: list[Any], current_date: date, days: int
    ) -> list[Any]:
        """Filter summaries to recent window."""
        cutoff = current_date - timedelta(days=days)
        return [s for s in summaries if s.date >= cutoff and s.date <= current_date]

    def _get_current_day(self, summaries: list[Any], current_date: date) -> Any | None:
        """Get summary for specific date."""
        for s in summaries:
            if s.date == current_date:
                return s
        return None

    # TODO: Remove these stubs after Q1 2025 - moved to SleepFeatureCalculator
    # NOTE: These methods have been moved to SleepFeatureCalculator
    # Keeping stub for backward compatibility if needed

    # def _calculate_interdaily_stability(
    #     self, sleep_summaries: list[DailySleepSummary]
    # ) -> float:
    #     """Moved to SleepFeatureCalculator"""
    #     return self.sleep_calculator.calculate_interdaily_stability(sleep_summaries)

    # def _calculate_intradaily_variability(
    #     self, sleep_summaries: list[DailySleepSummary]
    # ) -> float:
    #     """Moved to SleepFeatureCalculator"""
    #     return self.sleep_calculator.calculate_intradaily_variability(sleep_summaries)

    # def _calculate_relative_amplitude(
    #     self, sleep_summaries: list[DailySleepSummary]
    # ) -> float:
    #     """Moved to SleepFeatureCalculator"""
    #     return self.sleep_calculator.calculate_relative_amplitude(sleep_summaries)

    def _update_individual_baseline(self, metric: str, value: float) -> None:
        """Update individual baseline statistics using incremental update."""
        # Debug logging for sleep baseline
        if metric == "sleep":
            logger.debug(
                "updating_sleep_baseline",
                value=value,
                metric=metric,
                user_id=self.user_id,
            )
            
        if metric not in self.individual_baselines:
            self.individual_baselines[metric] = {
                "values": [], 
                "mean": 0.0, 
                "std": 0.0,
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0
            }

        baseline = self.individual_baselines[metric]
        
        # If we have a loaded baseline with existing statistics but no values,
        # initialize the incremental stats from the loaded mean/std
        if baseline.get("count", 0) == 0 and baseline["mean"] != 0:
            # Estimate count from loaded baseline (assuming 7 days of data minimum)
            if hasattr(self, "_loaded_baseline") and self._loaded_baseline:
                estimated_count = max(7, self._loaded_baseline.data_points)
            else:
                estimated_count = 7
            baseline["count"] = estimated_count
            baseline["sum"] = baseline["mean"] * estimated_count
            # Variance = std^2, and sum_sq can be derived from variance formula
            variance = baseline["std"] ** 2
            baseline["sum_sq"] = (variance * estimated_count) + (baseline["sum"] ** 2 / estimated_count)
        
        # Add new value to baseline
        values_list = baseline["values"]
        if not isinstance(values_list, list):
            values_list = []
            baseline["values"] = values_list
        values_list.append(value)

        # Update incremental statistics
        baseline["count"] = baseline.get("count", 0) + 1
        baseline["sum"] = baseline.get("sum", 0.0) + value
        baseline["sum_sq"] = baseline.get("sum_sq", 0.0) + (value ** 2)
        
        # Calculate new mean and std using incremental formulas
        count = baseline["count"]
        if count > 0:
            baseline["mean"] = baseline["sum"] / count
            if count > 1:
                # Variance = (sum_sq / n) - (mean^2)
                variance = (baseline["sum_sq"] / count) - (baseline["mean"] ** 2)
                # Handle numerical issues
                variance = max(0, variance)
                baseline["std"] = float(np.sqrt(variance))
            else:
                baseline["std"] = 0.0
        
        # Keep last 30 values for inspection (but don't use for calculation)
        if len(values_list) > 30:
            values_list.pop(0)

    def _calculate_zscore(self, metric: str, value: float) -> float:
        """Calculate Z-score relative to individual baseline."""
        if metric not in self.individual_baselines:
            return 0.0

        baseline = self.individual_baselines[metric]
        if baseline["std"] == 0:
            return 0.0

        return float((value - baseline["mean"]) / baseline["std"])

    def _empty_sleep_features(self) -> dict[str, float]:
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

    def _empty_circadian_features(self) -> dict[str, Any]:
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

    def _empty_activity_features(self) -> dict[str, float]:
        """Return empty activity features."""
        return {
            "activity_fragmentation": 0,
            "sedentary_bout_mean": 1440,  # 24 hours in minutes when no activity data
            "sedentary_bout_max": 1440,  # 24 hours in minutes when no activity data
            "activity_intensity_ratio": 0,
        }

    def _load_baselines_from_repository(self) -> None:
        """Load existing baselines from repository."""
        if not self.baseline_repository or not self.user_id:
            return

        baseline = self.baseline_repository.get_baseline(self.user_id)
        if not baseline:
            return

        # Store loaded baseline for data_points accumulation
        self._loaded_baseline = baseline

        # Convert UserBaseline to internal format
        self.individual_baselines["sleep"] = {
            "values": [],  # Historical values not stored
            "mean": baseline.sleep_mean,
            "std": baseline.sleep_std,
        }
        self.individual_baselines["activity"] = {
            "values": [],
            "mean": baseline.activity_mean,
            "std": baseline.activity_std,
        }
        # Note: We'd need to extend UserBaseline to include HR/HRV baselines
        # For now, these will be calculated fresh

    def persist_baselines(self) -> None:
        """Persist current baselines to repository."""
        if not self.baseline_repository or not self.user_id:
            return

        # Extract current baseline statistics
        sleep_baseline = self.individual_baselines.get("sleep", {})
        activity_baseline = self.individual_baselines.get("activity", {})

        # Calculate circadian phase from recent data if available
        circadian_phase = 0.0  # Would calculate from recent sleep patterns

        # Count data points - accumulate from previous baseline if exists
        current_data_points = max(
            len(sleep_baseline.get("values", [])),
            len(activity_baseline.get("values", [])),
        )
        
        # Add to previous data points if we loaded a baseline
        previous_data_points = 0
        if hasattr(self, "_loaded_baseline") and self._loaded_baseline:
            previous_data_points = self._loaded_baseline.data_points
        
        data_points = previous_data_points + current_data_points

        # Create UserBaseline object
        from big_mood_detector.domain.repositories.baseline_repository_interface import (
            UserBaseline,
        )

        baseline = UserBaseline(
            user_id=self.user_id,
            baseline_date=date.today(),
            sleep_mean=sleep_baseline.get("mean", 0.0),
            sleep_std=sleep_baseline.get("std", 0.0),
            activity_mean=activity_baseline.get("mean", 0.0),
            activity_std=activity_baseline.get("std", 0.0),
            circadian_phase=circadian_phase,
            last_updated=datetime.now(),
            data_points=data_points,
        )

        # Save to repository
        self.baseline_repository.save_baseline(baseline)
