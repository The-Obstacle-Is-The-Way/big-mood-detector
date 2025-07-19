"""
Feature Engineering Data Types

Common data types used across feature engineering services.
Following Single Responsibility Principle - these are just data containers.
"""

from dataclasses import dataclass
from datetime import date


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

