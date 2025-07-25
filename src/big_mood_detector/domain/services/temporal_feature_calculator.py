"""
Temporal Feature Calculator Service

Calculates time-based features from historical data patterns.
Extracted from AdvancedFeatureEngineer following Single Responsibility Principle.

Design Patterns:
- Strategy Pattern: Different calculation strategies for temporal analysis
- Value Objects: Immutable result objects
- Pure Functions: All calculations are side-effect free
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from big_mood_detector.domain.utils.math_helpers import safe_std, safe_var

T = TypeVar("T")  # Generic type for domain summaries


@dataclass(frozen=True)
class RollingStatistics:
    """Rolling window statistics."""

    mean: float
    std: float
    min: float
    max: float
    trend: float  # Simple linear trend


@dataclass(frozen=True)
class TrendFeatures:
    """Trend analysis features."""

    slope: float  # Rate of change
    r_squared: float  # Goodness of fit
    acceleration: float  # Second derivative
    is_increasing: bool
    is_stable: bool  # Low variance


@dataclass(frozen=True)
class VariabilityFeatures:
    """Variability metrics."""

    coefficient_of_variation: float
    range: float
    iqr: float  # Interquartile range
    mad: float  # Median absolute deviation


@dataclass(frozen=True)
class PeriodicityFeatures:
    """Periodicity detection results."""

    dominant_period_days: int
    period_strength: float  # 0-1, strength of periodicity
    phase_shift: float  # Days offset from expected phase


@dataclass(frozen=True)
class ChangePoint:
    """Detected change point in time series."""

    index: int
    date: Any  # Using Any to avoid circular imports
    magnitude: float
    direction: str  # "increase" or "decrease"


@dataclass(frozen=True)
class CrossDomainCorrelation:
    """Correlation between different health domains."""

    pearson_r: float
    spearman_rho: float
    p_value: float
    is_significant: bool


@dataclass(frozen=True)
class MomentumFeatures:
    """Momentum (rate of change) features."""

    short_term_momentum: float
    long_term_momentum: float
    momentum_divergence: float
    is_accelerating: bool


class TemporalFeatureCalculator:
    """
    Calculates temporal features from time series health data.

    This service focuses on time-based patterns and changes
    that are crucial for mood episode detection.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize with optional configuration.

        Args:
            config: Configuration dict with temporal settings
        """
        if config is None:
            # Default values
            self.significance_level = 0.05
            self.stability_threshold = 0.1  # CV threshold for stability
            self.min_period_strength = 0.3
        else:
            temporal_config = config.get("temporal", {})
            self.significance_level = temporal_config.get("significance_level", 0.05)
            self.stability_threshold = temporal_config.get("stability_threshold", 0.1)
            self.min_period_strength = temporal_config.get("min_period_strength", 0.3)

    def calculate_rolling_statistics(
        self, data: list[T], window_days: int, metric_extractor: Callable[[T], float]
    ) -> RollingStatistics:
        """
        Calculate rolling window statistics.

        Args:
            data: List of daily summaries
            window_days: Size of rolling window
            metric_extractor: Function to extract metric from summary

        Returns:
            Rolling statistics for the window
        """
        if not data:
            return RollingStatistics(mean=0, std=0, min=0, max=0, trend=0)

        # Extract metrics
        metrics = [metric_extractor(d) for d in data]

        # Use last window_days of data
        window_data = metrics[-window_days:] if len(metrics) > window_days else metrics

        if not window_data:
            return RollingStatistics(mean=0, std=0, min=0, max=0, trend=0)

        mean = float(np.mean(window_data))
        std = float(safe_std(window_data))
        min_val = float(min(window_data))
        max_val = float(max(window_data))

        # Simple trend: slope of linear fit
        if len(window_data) > 1:
            x = np.arange(len(window_data))
            slope, _ = np.polyfit(x, window_data, 1)
            trend = float(slope)
        else:
            trend = 0.0

        return RollingStatistics(
            mean=mean, std=std, min=min_val, max=max_val, trend=trend
        )

    def calculate_trend_features(
        self, data: list[T], metric_extractor: Callable[[T], float], window_days: int
    ) -> TrendFeatures:
        """
        Calculate trend features using linear regression.

        Args:
            data: List of daily summaries
            metric_extractor: Function to extract metric
            window_days: Window for trend calculation

        Returns:
            Trend analysis features
        """
        if not data:
            return TrendFeatures(
                slope=0,
                r_squared=0,
                acceleration=0,
                is_increasing=False,
                is_stable=True,
            )

        # Extract metrics from window
        metrics = [metric_extractor(d) for d in data]
        window_data = metrics[-window_days:] if len(metrics) > window_days else metrics

        if len(window_data) < 2:
            return TrendFeatures(
                slope=0,
                r_squared=0,
                acceleration=0,
                is_increasing=False,
                is_stable=True,
            )

        # Linear regression
        x = np.arange(len(window_data))
        y = np.array(window_data)

        # Calculate slope and r-squared
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Acceleration (second derivative estimate)
        if len(window_data) > 2:
            # Calculate daily changes
            changes = np.diff(window_data)
            # Acceleration is change in changes
            if len(changes) > 1:
                acceleration = float(np.mean(np.diff(changes)))
            else:
                acceleration = 0.0
        else:
            acceleration = 0.0

        # Determine if stable (low CV)
        cv = (
            safe_std(window_data) / np.mean(window_data)
            if np.mean(window_data) != 0
            else 0
        )
        is_stable = cv < self.stability_threshold

        return TrendFeatures(
            slope=float(slope),
            r_squared=float(r_squared),
            acceleration=acceleration,
            is_increasing=slope > 0,
            is_stable=bool(is_stable),
        )

    def calculate_variability_features(
        self, data: list[T], metric_extractor: Callable[[T], float], window_days: int
    ) -> VariabilityFeatures:
        """
        Calculate variability metrics.

        Args:
            data: List of daily summaries
            metric_extractor: Function to extract metric
            window_days: Window for calculation

        Returns:
            Variability features
        """
        if not data:
            return VariabilityFeatures(
                coefficient_of_variation=0, range=0, iqr=0, mad=0
            )

        # Extract metrics from window
        metrics = [metric_extractor(d) for d in data]
        window_data = metrics[-window_days:] if len(metrics) > window_days else metrics

        if not window_data:
            return VariabilityFeatures(
                coefficient_of_variation=0, range=0, iqr=0, mad=0
            )

        # Coefficient of variation
        mean_val = np.mean(window_data)
        cv = safe_std(window_data) / mean_val if mean_val != 0 else 0

        # Range
        data_range = max(window_data) - min(window_data)

        # Interquartile range
        q1 = np.percentile(window_data, 25)
        q3 = np.percentile(window_data, 75)
        iqr = q3 - q1

        # Median absolute deviation
        median = np.median(window_data)
        mad = np.median([abs(x - median) for x in window_data])

        return VariabilityFeatures(
            coefficient_of_variation=float(cv),
            range=float(data_range),
            iqr=float(iqr),
            mad=float(mad),
        )

    def calculate_periodicity_features(
        self,
        data: list[T],
        metric_extractor: Callable[[T], float],
        max_period_days: int = 7,
    ) -> PeriodicityFeatures:
        """
        Detect periodic patterns (e.g., weekly cycles).

        Args:
            data: List of daily summaries
            metric_extractor: Function to extract metric
            max_period_days: Maximum period to check

        Returns:
            Periodicity features
        """
        if len(data) < max_period_days * 2:
            return PeriodicityFeatures(
                dominant_period_days=0, period_strength=0, phase_shift=0
            )

        metrics = [metric_extractor(d) for d in data]

        # Try different periods and find strongest
        best_period = 0
        best_strength = 0
        best_phase = 0

        for period in range(2, min(max_period_days + 1, len(metrics) // 2)):
            # Calculate autocorrelation at this lag
            if len(metrics) > period:
                # Normalize data
                mean_val = float(np.mean(metrics))
                std_val = safe_std(metrics)
                normalized = [(m - mean_val) / (std_val + 1e-10) for m in metrics]

                # Calculate correlation with shifted version
                normalized_array = np.array(normalized)
                correlation = np.correlate(
                    normalized_array[:-period], normalized_array[period:], mode="valid"
                )[0]
                correlation /= len(normalized_array) - period

                if abs(correlation) > best_strength:
                    best_strength = abs(correlation)
                    best_period = period

                    # Estimate phase shift
                    # Find which day of period has highest average
                    period_averages = []
                    for day in range(period):
                        day_values = [
                            metrics[i] for i in range(day, len(metrics), period)
                        ]
                        period_averages.append(np.mean(day_values))

                    best_phase = int(np.argmax(period_averages))

        return PeriodicityFeatures(
            dominant_period_days=best_period,
            period_strength=float(best_strength),
            phase_shift=float(best_phase),
        )

    def calculate_anomaly_scores(
        self, data: list[T], metric_extractor: Callable[[T], float], window_days: int
    ) -> list[float]:
        """
        Calculate anomaly scores for each day.

        Uses z-score method within rolling window.

        Args:
            data: List of daily summaries
            metric_extractor: Function to extract metric
            window_days: Window for baseline calculation

        Returns:
            List of anomaly scores (z-scores)
        """
        if not data:
            return []

        metrics = [metric_extractor(d) for d in data]
        anomaly_scores = []

        for i in range(len(metrics)):
            # Get window of data before current point
            if i < window_days:
                # Use all previous data if not enough
                window = metrics[:i] if i > 0 else [metrics[0]]
            else:
                # Use previous window_days
                window = metrics[i - window_days : i]

            if len(window) > 1:
                mean = np.mean(window)
                std = safe_std(window)

                if std > 0:
                    z_score = float(abs(metrics[i] - mean) / std)
                else:
                    z_score = 0.0
            else:
                z_score = 0.0

            anomaly_scores.append(float(z_score))

        return anomaly_scores

    def detect_change_points(
        self,
        data: list[T],
        metric_extractor: Callable[[T], float],
        min_segment_days: int = 5,
    ) -> list[ChangePoint]:
        """
        Detect significant changes in time series.

        Uses simple threshold method for change detection.

        Args:
            data: List of daily summaries
            metric_extractor: Function to extract metric
            min_segment_days: Minimum days between change points

        Returns:
            List of detected change points
        """
        if len(data) < min_segment_days * 2:
            return []

        metrics = [metric_extractor(d) for d in data]
        change_points = []

        # Simple method: compare running averages
        for i in range(min_segment_days, len(metrics) - min_segment_days):
            # Compare before and after segments
            before = metrics[i - min_segment_days : i]
            after = metrics[i : i + min_segment_days]

            before_mean = np.mean(before)
            after_mean = np.mean(after)

            # Calculate pooled standard deviation
            pooled_std = np.sqrt((safe_var(before) + safe_var(after)) / 2)

            if pooled_std > 0:
                # Cohen's d effect size
                effect_size = abs(after_mean - before_mean) / pooled_std

                # Significant change if effect size > 0.8 (large effect)
                if effect_size > 0.8:
                    change_points.append(
                        ChangePoint(
                            index=i,
                            date=data[i].date,  # type: ignore[attr-defined]
                            magnitude=float(after_mean - before_mean),
                            direction=(
                                "increase" if after_mean > before_mean else "decrease"
                            ),
                        )
                    )

                    # Skip ahead to avoid detecting overlapping changes
                    i += min_segment_days

        return change_points

    def calculate_cross_domain_correlation(
        self,
        data1: list[T],
        data2: list[T],
        metric1_extractor: Callable[[T], float],
        metric2_extractor: Callable[[T], float],
        lag_days: int = 0,
    ) -> CrossDomainCorrelation:
        """
        Calculate correlation between two health domains.

        Args:
            data1: First domain data
            data2: Second domain data
            metric1_extractor: Extract metric from domain 1
            metric2_extractor: Extract metric from domain 2
            lag_days: Time lag for correlation

        Returns:
            Cross-domain correlation metrics
        """
        if not data1 or not data2:
            return CrossDomainCorrelation(
                pearson_r=0, spearman_rho=0, p_value=1.0, is_significant=False
            )

        # Align data by dates
        dates1 = {d.date: metric1_extractor(d) for d in data1}  # type: ignore[attr-defined]
        dates2 = {d.date: metric2_extractor(d) for d in data2}  # type: ignore[attr-defined]

        # Find common dates
        common_dates = sorted(set(dates1.keys()) & set(dates2.keys()))

        if len(common_dates) < 3:
            return CrossDomainCorrelation(
                pearson_r=0, spearman_rho=0, p_value=1.0, is_significant=False
            )

        # Extract aligned metrics
        metrics1 = [dates1[d] for d in common_dates]
        metrics2 = [dates2[d] for d in common_dates]

        # Apply lag if specified
        if lag_days > 0 and lag_days < len(metrics1):
            metrics1 = metrics1[:-lag_days]
            metrics2 = metrics2[lag_days:]
        elif lag_days < 0 and abs(lag_days) < len(metrics1):
            metrics1 = metrics1[abs(lag_days) :]
            metrics2 = metrics2[:lag_days]

        # Calculate correlations
        if len(metrics1) >= 3:
            pearson_r, pearson_p = stats.pearsonr(metrics1, metrics2)
            spearman_rho, spearman_p = stats.spearmanr(metrics1, metrics2)

            # Use more conservative p-value
            p_value = max(pearson_p, spearman_p)
        else:
            pearson_r = spearman_rho = 0
            p_value = 1.0

        return CrossDomainCorrelation(
            pearson_r=float(pearson_r),
            spearman_rho=float(spearman_rho),
            p_value=float(p_value),
            is_significant=p_value < self.significance_level,
        )

    def calculate_momentum_features(
        self,
        data: list[T],
        metric_extractor: Callable[[T], float],
        short_window: int = 3,
        long_window: int = 7,
    ) -> MomentumFeatures:
        """
        Calculate momentum (rate of change) features.

        Args:
            data: List of daily summaries
            metric_extractor: Function to extract metric
            short_window: Days for short-term momentum
            long_window: Days for long-term momentum

        Returns:
            Momentum features
        """
        if not data:
            return MomentumFeatures(
                short_term_momentum=0,
                long_term_momentum=0,
                momentum_divergence=0,
                is_accelerating=False,
            )

        metrics = [metric_extractor(d) for d in data]

        # Short-term momentum
        if len(metrics) >= short_window:
            short_data = metrics[-short_window:]
            x = np.arange(len(short_data))
            short_slope, _ = np.polyfit(x, short_data, 1)
            short_momentum = float(short_slope)
        else:
            short_momentum = 0.0

        # Long-term momentum
        if len(metrics) >= long_window:
            long_data = metrics[-long_window:]
            x = np.arange(len(long_data))
            long_slope, _ = np.polyfit(x, long_data, 1)
            long_momentum = float(long_slope)
        else:
            long_momentum = short_momentum

        # Momentum divergence
        divergence = short_momentum - long_momentum

        # Is accelerating if short > long and both positive
        is_accelerating = short_momentum > long_momentum and short_momentum > 0

        return MomentumFeatures(
            short_term_momentum=short_momentum,
            long_term_momentum=long_momentum,
            momentum_divergence=float(divergence),
            is_accelerating=is_accelerating,
        )
