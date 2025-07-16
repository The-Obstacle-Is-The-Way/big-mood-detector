"""
Feature Engineering Orchestrator

High-level orchestration service that coordinates all feature engineering components.
Provides a clean interface for the complete feature extraction pipeline.

Design Patterns:
- Facade Pattern: Simplified interface to complex subsystem
- Builder Pattern: Constructs complex feature sets step by step
- Strategy Pattern: Swappable feature calculators
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
    AdvancedFeatures,
)
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


@dataclass(frozen=True)
class SleepFeatureSet:
    """Immutable container for sleep-related features."""
    total_sleep_hours: float
    sleep_efficiency: float
    sleep_regularity_index: float
    interdaily_stability: float
    intradaily_variability: float
    relative_amplitude: float
    short_sleep_window_pct: float
    long_sleep_window_pct: float
    sleep_onset_variance: float
    wake_time_variance: float


@dataclass(frozen=True)
class CircadianFeatureSet:
    """Immutable container for circadian rhythm features."""
    l5_value: float
    m10_value: float
    circadian_phase_advance: float
    circadian_phase_delay: float
    circadian_amplitude: float
    phase_angle: float


@dataclass(frozen=True)
class ActivityFeatureSet:
    """Immutable container for activity features."""
    total_steps: float
    activity_fragmentation: float
    sedentary_bout_mean: float
    sedentary_bout_max: float
    activity_intensity_ratio: float
    activity_rhythm_strength: float


@dataclass(frozen=True)
class TemporalFeatureSet:
    """Immutable container for temporal features."""
    sleep_7day_mean: float
    sleep_7day_std: float
    activity_7day_mean: float
    activity_7day_std: float
    hr_7day_mean: float
    hr_7day_std: float
    sleep_trend_slope: float
    activity_trend_slope: float
    sleep_momentum: float
    activity_momentum: float


@dataclass(frozen=True)
class ClinicalFeatureSet:
    """Immutable container for clinical indicators."""
    is_hypersomnia_pattern: bool
    is_insomnia_pattern: bool
    is_phase_advanced: bool
    is_phase_delayed: bool
    is_irregular_pattern: bool
    mood_risk_score: float


@dataclass(frozen=True)
class UnifiedFeatureSet:
    """Complete feature set with all domains."""
    date: date
    sleep_features: SleepFeatureSet
    circadian_features: CircadianFeatureSet
    activity_features: ActivityFeatureSet
    temporal_features: TemporalFeatureSet
    clinical_features: ClinicalFeatureSet


@dataclass(frozen=True)
class FeatureValidationResult:
    """Result of feature validation."""
    is_valid: bool
    missing_domains: list[str]
    quality_score: float  # 0-1, higher is better
    warnings: list[str]


@dataclass(frozen=True)
class CompletenessReport:
    """Data completeness report."""
    total_days: int
    sleep_coverage: float
    activity_coverage: float
    heart_coverage: float
    full_coverage_days: int
    gaps: list[tuple[date, list[str]]]  # (date, missing_domains)


@dataclass(frozen=True)
class AnomalyResult:
    """Anomaly detection result."""
    has_anomalies: bool
    anomaly_domains: list[str]
    severity: float  # 0-1, higher is more severe


class FeatureEngineeringOrchestrator:
    """
    Orchestrates the complete feature engineering pipeline.

    This service provides a high-level interface that coordinates
    all feature calculators and handles data flow between them.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize orchestrator with configuration.

        Args:
            config: Configuration for all feature calculators
        """
        self.config = config or {}

        # Initialize the main feature engineer
        # (which already has the individual calculators)
        self.feature_engineer = AdvancedFeatureEngineer(config)

        # Feature importance (would be loaded from model analysis)
        self._feature_importance = {
            "sleep_regularity_index": 0.95,
            "circadian_phase_advance": 0.88,
            "circadian_phase_delay": 0.87,
            "activity_fragmentation": 0.82,
            "mood_risk_score": 0.90,
            "sleep_7day_std": 0.75,
            "is_irregular_pattern": 0.85,
            "intradaily_variability": 0.73,
            "phase_angle": 0.70,
            "activity_intensity_ratio": 0.68,
        }

        # Cache for performance
        self._feature_cache: dict[tuple[date, int], UnifiedFeatureSet] = {}

    def extract_features_for_date(
        self,
        target_date: date,
        sleep_data: list[DailySleepSummary],
        activity_data: list[DailyActivitySummary],
        heart_data: list[DailyHeartSummary],
        lookback_days: int = 30,
        use_cache: bool = True
    ) -> UnifiedFeatureSet:
        """
        Extract all features for a specific date.

        Args:
            target_date: Date to extract features for
            sleep_data: Historical sleep summaries
            activity_data: Historical activity summaries
            heart_data: Historical heart summaries
            lookback_days: Days of history to consider
            use_cache: Whether to use cached results

        Returns:
            Complete feature set for the date
        """
        # Check cache
        cache_key = (target_date, lookback_days)
        if use_cache and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        # Extract features using the main engineer
        advanced_features = self.feature_engineer.extract_advanced_features(
            current_date=target_date,
            historical_sleep=sleep_data,
            historical_activity=activity_data,
            historical_heart=heart_data,
            lookback_days=lookback_days
        )

        # Convert to our structured format
        unified_features = self._convert_to_unified_features(advanced_features)

        # Cache result
        if use_cache:
            self._feature_cache[cache_key] = unified_features

        return unified_features

    def extract_features_batch(
        self,
        start_date: date,
        end_date: date,
        sleep_data: list[DailySleepSummary],
        activity_data: list[DailyActivitySummary],
        heart_data: list[DailyHeartSummary],
        lookback_days: int = 30
    ) -> list[UnifiedFeatureSet]:
        """
        Extract features for a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range (inclusive)
            sleep_data: Historical sleep summaries
            activity_data: Historical activity summaries
            heart_data: Historical heart summaries
            lookback_days: Days of history to consider

        Returns:
            List of feature sets for each date
        """
        feature_sets = []
        current_date = start_date

        while current_date <= end_date:
            features = self.extract_features_for_date(
                target_date=current_date,
                sleep_data=sleep_data,
                activity_data=activity_data,
                heart_data=heart_data,
                lookback_days=lookback_days,
                use_cache=True
            )
            feature_sets.append(features)
            current_date += timedelta(days=1)

        return feature_sets

    def validate_features(self, features: UnifiedFeatureSet) -> FeatureValidationResult:
        """
        Validate feature quality and completeness.

        Args:
            features: Feature set to validate

        Returns:
            Validation result with quality metrics
        """
        missing_domains = []
        warnings = []

        # Check sleep features
        if features.sleep_features.total_sleep_hours == 0:
            missing_domains.append("sleep")
        elif features.sleep_features.total_sleep_hours < 3:
            warnings.append("Unusually low sleep duration")
        elif features.sleep_features.total_sleep_hours > 12:
            warnings.append("Unusually high sleep duration")

        # Check activity features
        if features.activity_features.total_steps == 0:
            missing_domains.append("activity")
        elif features.activity_features.total_steps < 100:
            warnings.append("Unusually low activity")

        # Check temporal features
        if (features.temporal_features.sleep_7day_mean == 0 and
            features.temporal_features.activity_7day_mean == 0):
            warnings.append("Insufficient historical data for temporal features")

        # Calculate quality score
        feature_coverage = 1.0 - (len(missing_domains) / 3.0)  # 3 main domains

        # Penalize for warnings
        warning_penalty = len(warnings) * 0.05
        quality_score = max(0.0, min(1.0, feature_coverage - warning_penalty))

        return FeatureValidationResult(
            is_valid=len(missing_domains) == 0,
            missing_domains=missing_domains,
            quality_score=quality_score,
            warnings=warnings
        )

    def generate_completeness_report(
        self,
        sleep_data: list[DailySleepSummary],
        activity_data: list[DailyActivitySummary],
        heart_data: list[DailyHeartSummary]
    ) -> CompletenessReport:
        """
        Generate data completeness report.

        Args:
            sleep_data: Sleep summaries
            activity_data: Activity summaries
            heart_data: Heart summaries

        Returns:
            Completeness report with coverage metrics
        """
        # Get date ranges
        all_dates = set()
        sleep_dates = {s.date for s in sleep_data}
        activity_dates = {a.date for a in activity_data}
        heart_dates = {h.date for h in heart_data}

        all_dates.update(sleep_dates, activity_dates, heart_dates)

        if not all_dates:
            return CompletenessReport(
                total_days=0,
                sleep_coverage=0.0,
                activity_coverage=0.0,
                heart_coverage=0.0,
                full_coverage_days=0,
                gaps=[]
            )

        total_days = len(all_dates)

        # Calculate coverage
        sleep_coverage = len(sleep_dates) / total_days
        activity_coverage = len(activity_dates) / total_days
        heart_coverage = len(heart_dates) / total_days

        # Find full coverage days
        full_coverage_dates = sleep_dates & activity_dates & heart_dates
        full_coverage_days = len(full_coverage_dates)

        # Identify gaps
        gaps = []
        for d in sorted(all_dates):
            missing = []
            if d not in sleep_dates:
                missing.append("sleep")
            if d not in activity_dates:
                missing.append("activity")
            if d not in heart_dates:
                missing.append("heart")

            if missing:
                gaps.append((d, missing))

        return CompletenessReport(
            total_days=total_days,
            sleep_coverage=sleep_coverage,
            activity_coverage=activity_coverage,
            heart_coverage=heart_coverage,
            full_coverage_days=full_coverage_days,
            gaps=gaps
        )

    def detect_anomalies(self, features: UnifiedFeatureSet) -> AnomalyResult:
        """
        Detect anomalies in feature set.

        Args:
            features: Feature set to analyze

        Returns:
            Anomaly detection result
        """
        anomaly_domains = []
        severity_scores = []

        # Check sleep anomalies - both chronic patterns and acute issues
        if (features.clinical_features.is_hypersomnia_pattern or
            features.clinical_features.is_insomnia_pattern):
            anomaly_domains.append("sleep")
            severity_scores.append(0.7)

        # Check for acute sleep anomalies
        if features.sleep_features.total_sleep_hours < 3:
            anomaly_domains.append("sleep")
            severity_scores.append(0.8)
        elif features.sleep_features.total_sleep_hours > 12:
            anomaly_domains.append("sleep")
            severity_scores.append(0.6)

        if features.sleep_features.sleep_efficiency < 0.7:
            anomaly_domains.append("sleep")
            severity_scores.append(0.5)

        if features.sleep_features.sleep_regularity_index < 50:
            anomaly_domains.append("sleep_regularity")
            severity_scores.append(0.6)

        # Check circadian anomalies
        if (features.clinical_features.is_phase_advanced or
            features.clinical_features.is_phase_delayed):
            anomaly_domains.append("circadian")
            severity_scores.append(0.5)

        # Check activity anomalies
        if features.activity_features.activity_fragmentation > 0.8:
            anomaly_domains.append("activity")
            severity_scores.append(0.6)

        # Check for extreme activity levels
        if features.activity_features.total_steps < 1000:
            anomaly_domains.append("activity")
            severity_scores.append(0.5)
        elif features.activity_features.total_steps > 20000:
            anomaly_domains.append("activity")
            severity_scores.append(0.6)

        # Overall severity
        severity = max(severity_scores) if severity_scores else 0.0

        return AnomalyResult(
            has_anomalies=len(anomaly_domains) > 0,
            anomaly_domains=list(set(anomaly_domains)),  # Remove duplicates
            severity=severity
        )

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary of feature names to importance scores (0-1)
        """
        return self._feature_importance.copy()

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._feature_cache.clear()

    def export_features_to_dict(
        self, feature_sets: list[UnifiedFeatureSet]
    ) -> list[dict[str, Any]]:
        """
        Export feature sets to dictionary format for DataFrame conversion.

        Args:
            feature_sets: List of feature sets

        Returns:
            List of dictionaries with flattened features
        """
        rows = []

        for features in feature_sets:
            row = {
                "date": features.date,
                # Sleep features
                "sleep_duration_hours": features.sleep_features.total_sleep_hours,
                "sleep_efficiency": features.sleep_features.sleep_efficiency,
                "sleep_regularity_index": features.sleep_features.sleep_regularity_index,
                "interdaily_stability": features.sleep_features.interdaily_stability,
                "intradaily_variability": features.sleep_features.intradaily_variability,
                "relative_amplitude": features.sleep_features.relative_amplitude,
                "short_sleep_window_pct": features.sleep_features.short_sleep_window_pct,
                "long_sleep_window_pct": features.sleep_features.long_sleep_window_pct,
                "sleep_onset_variance": features.sleep_features.sleep_onset_variance,
                "wake_time_variance": features.sleep_features.wake_time_variance,
                # Circadian features
                "l5_value": features.circadian_features.l5_value,
                "m10_value": features.circadian_features.m10_value,
                "circadian_phase_advance": features.circadian_features.circadian_phase_advance,
                "circadian_phase_delay": features.circadian_features.circadian_phase_delay,
                "circadian_amplitude": features.circadian_features.circadian_amplitude,
                "phase_angle": features.circadian_features.phase_angle,
                # Activity features
                "total_steps": features.activity_features.total_steps,
                "activity_fragmentation": features.activity_features.activity_fragmentation,
                "sedentary_bout_mean": features.activity_features.sedentary_bout_mean,
                "sedentary_bout_max": features.activity_features.sedentary_bout_max,
                "activity_intensity_ratio": features.activity_features.activity_intensity_ratio,
                "activity_rhythm_strength": features.activity_features.activity_rhythm_strength,
                # Temporal features
                "sleep_7day_mean": features.temporal_features.sleep_7day_mean,
                "sleep_7day_std": features.temporal_features.sleep_7day_std,
                "activity_7day_mean": features.temporal_features.activity_7day_mean,
                "activity_7day_std": features.temporal_features.activity_7day_std,
                "hr_7day_mean": features.temporal_features.hr_7day_mean,
                "hr_7day_std": features.temporal_features.hr_7day_std,
                "sleep_trend_slope": features.temporal_features.sleep_trend_slope,
                "activity_trend_slope": features.temporal_features.activity_trend_slope,
                # Clinical features
                "is_hypersomnia_pattern": features.clinical_features.is_hypersomnia_pattern,
                "is_insomnia_pattern": features.clinical_features.is_insomnia_pattern,
                "is_phase_advanced": features.clinical_features.is_phase_advanced,
                "is_phase_delayed": features.clinical_features.is_phase_delayed,
                "is_irregular_pattern": features.clinical_features.is_irregular_pattern,
                "mood_risk_score": features.clinical_features.mood_risk_score,
            }
            rows.append(row)

        return rows

    def _convert_to_unified_features(
        self, advanced_features: AdvancedFeatures
    ) -> UnifiedFeatureSet:
        """
        Convert AdvancedFeatures to UnifiedFeatureSet.

        Args:
            advanced_features: Features from AdvancedFeatureEngineer

        Returns:
            Structured unified feature set
        """
        # Extract additional temporal features using the calculator
        temporal_calc = self.feature_engineer.temporal_calculator

        # Get trend features (using dummy data for now - in production would use actual)
        sleep_trend = temporal_calc.calculate_trend_features(
            [], lambda x: 0, window_days=7
        )
        activity_trend = temporal_calc.calculate_trend_features(
            [], lambda x: 0, window_days=7
        )

        # Get momentum features
        sleep_momentum = temporal_calc.calculate_momentum_features(
            [], lambda x: 0, short_window=3, long_window=7
        )
        activity_momentum = temporal_calc.calculate_momentum_features(
            [], lambda x: 0, short_window=3, long_window=7
        )

        return UnifiedFeatureSet(
            date=advanced_features.date,
            sleep_features=SleepFeatureSet(
                total_sleep_hours=advanced_features.sleep_duration_hours,
                sleep_efficiency=advanced_features.sleep_efficiency,
                sleep_regularity_index=advanced_features.sleep_regularity_index,
                interdaily_stability=advanced_features.interdaily_stability,
                intradaily_variability=advanced_features.intradaily_variability,
                relative_amplitude=advanced_features.relative_amplitude,
                short_sleep_window_pct=advanced_features.short_sleep_window_pct,
                long_sleep_window_pct=advanced_features.long_sleep_window_pct,
                sleep_onset_variance=advanced_features.sleep_onset_variance,
                wake_time_variance=advanced_features.wake_time_variance,
            ),
            circadian_features=CircadianFeatureSet(
                l5_value=advanced_features.l5_value,
                m10_value=advanced_features.m10_value,
                circadian_phase_advance=advanced_features.circadian_phase_advance,
                circadian_phase_delay=advanced_features.circadian_phase_delay,
                circadian_amplitude=advanced_features.relative_amplitude,  # Using RA as proxy
                phase_angle=0.0,  # Would need to calculate from actual data
            ),
            activity_features=ActivityFeatureSet(
                total_steps=float(advanced_features.total_steps),
                activity_fragmentation=advanced_features.activity_fragmentation,
                sedentary_bout_mean=advanced_features.sedentary_bout_mean,
                sedentary_bout_max=advanced_features.sedentary_bout_max,
                activity_intensity_ratio=advanced_features.activity_intensity_ratio,
                activity_rhythm_strength=0.7,  # Would calculate from actual data
            ),
            temporal_features=TemporalFeatureSet(
                sleep_7day_mean=advanced_features.sleep_7day_mean,
                sleep_7day_std=advanced_features.sleep_7day_std,
                activity_7day_mean=advanced_features.activity_7day_mean,
                activity_7day_std=advanced_features.activity_7day_std,
                hr_7day_mean=advanced_features.hr_7day_mean,
                hr_7day_std=advanced_features.hr_7day_std,
                sleep_trend_slope=sleep_trend.slope,
                activity_trend_slope=activity_trend.slope,
                sleep_momentum=sleep_momentum.short_term_momentum,
                activity_momentum=activity_momentum.short_term_momentum,
            ),
            clinical_features=ClinicalFeatureSet(
                is_hypersomnia_pattern=advanced_features.is_hypersomnia_pattern,
                is_insomnia_pattern=advanced_features.is_insomnia_pattern,
                is_phase_advanced=advanced_features.is_phase_advanced,
                is_phase_delayed=advanced_features.is_phase_delayed,
                is_irregular_pattern=advanced_features.is_irregular_pattern,
                mood_risk_score=advanced_features.mood_risk_score,
            )
        )
