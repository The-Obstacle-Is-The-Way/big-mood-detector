"""
Sparse Data Handler

Handles sparse temporal health data with missing days, sensor misalignment,
and variable data density. Implements 2025 best practices for robust analysis.

Based on research in docs/sparse_temporal_data_research_2025.md
"""

from dataclasses import dataclass
from datetime import date
from typing import Any
from enum import Enum, auto

import numpy as np
import pandas as pd

from big_mood_detector.domain.services.interpolation_strategies import (
    InterpolationMethod,
    InterpolationStrategyFactory,
)


class DataDensity(Enum):
    """Data density classification."""

    DENSE = auto()  # >80% coverage, <3 day gaps
    MODERATE = auto()  # 50-80% coverage, <7 day gaps
    SPARSE = auto()  # 20-50% coverage, <14 day gaps
    VERY_SPARSE = auto()  # <20% coverage or >14 day gaps

    def __eq__(self, other: object) -> bool:
        return self.value == other.value if isinstance(other, DataDensity) else False


# InterpolationMethod moved to interpolation_strategies module


class AlignmentStrategy(Enum):
    """Multi-sensor alignment strategies."""

    INTERSECTION = auto()  # Only overlapping times
    UNION = auto()  # All times (with NaN)
    AGGREGATE_TO_DAILY = auto()  # Aggregate to daily level
    INTERPOLATE_ALIGN = auto()  # Interpolate to common grid


@dataclass
class DensityMetrics:
    """Metrics describing data density and quality."""

    coverage_ratio: float  # Proportion of days with data
    max_gap_days: int  # Largest gap in days
    consecutive_days: int  # Longest consecutive run
    total_days: int  # Total days in range
    missing_patterns: dict[str, Any]  # Patterns in missingness
    density_class: DataDensity

    @property
    def requires_special_handling(self) -> bool:
        """Whether data needs special sparse handling."""
        return self.density_class in [DataDensity.SPARSE, DataDensity.VERY_SPARSE]


@dataclass
class ProcessingStrategy:
    """Selected processing strategy based on data characteristics."""

    algorithm: str
    confidence: float
    interpolation_method: InterpolationMethod | None
    features_available: list[str]
    warnings: list[str]


class SparseDataHandler:
    """
    Handles sparse temporal health data with adaptive strategies.

    Implements recommendations from 2025 research on sparse health data.
    """

    def __init__(
        self,
        max_interpolation_gap_hours: int = 24,
        min_confidence_threshold: float = 0.5,
    ):
        """
        Initialize sparse data handler.

        Args:
            max_interpolation_gap_hours: Maximum gap to interpolate
            min_confidence_threshold: Minimum confidence for predictions
        """
        self.max_interpolation_gap_hours = max_interpolation_gap_hours
        self.min_confidence_threshold = min_confidence_threshold

    def assess_density(self, dates: list[date] | pd.DatetimeIndex) -> DensityMetrics:
        """
        Assess the density and quality of temporal data.

        Args:
            dates: List of dates with data

        Returns:
            DensityMetrics with detailed assessment
        """
        # Handle DatetimeIndex
        if isinstance(dates, pd.DatetimeIndex):
            dates = [d.date() for d in dates]

        if not dates:
            return DensityMetrics(
                coverage_ratio=0.0,
                max_gap_days=0,
                consecutive_days=0,
                total_days=0,
                missing_patterns={},
                density_class=DataDensity.VERY_SPARSE,
            )

        dates = sorted(dates)
        set(dates)

        # Calculate basic metrics
        start_date = dates[0]
        end_date = dates[-1]
        total_days = (end_date - start_date).days + 1
        coverage_ratio = len(dates) / total_days if total_days > 0 else 0

        # Find gaps
        gaps = []
        consecutive_runs = []
        current_run = 1

        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days - 1
            if gap > 0:
                gaps.append(gap)
                consecutive_runs.append(current_run)
                current_run = 1
            else:
                current_run += 1
        consecutive_runs.append(current_run)

        max_gap_days = max(gaps) if gaps else 0
        consecutive_days = max(consecutive_runs) if consecutive_runs else len(dates)

        # Analyze missing patterns
        missing_patterns = self._analyze_missing_patterns(dates, start_date, end_date)

        # Classify density
        if coverage_ratio > 0.8 and max_gap_days < 3:
            density_class = DataDensity.DENSE
        elif coverage_ratio > 0.5 and max_gap_days < 7:
            density_class = DataDensity.MODERATE
        elif coverage_ratio > 0.1 and max_gap_days < 14:  # Lowered threshold
            density_class = DataDensity.SPARSE
        else:
            density_class = DataDensity.VERY_SPARSE

        return DensityMetrics(
            coverage_ratio=coverage_ratio,
            max_gap_days=max_gap_days,
            consecutive_days=consecutive_days,
            total_days=total_days,
            missing_patterns=missing_patterns,
            density_class=density_class,
        )

    def _analyze_missing_patterns(
        self, dates: list[date], start: date, end: date
    ) -> dict[str, Any]:
        """Analyze patterns in missing data."""
        date_set = set(dates)
        all_dates = pd.date_range(start, end, freq="D")

        missing_dates = [d.date() for d in all_dates if d.date() not in date_set]

        if not missing_dates:
            return {"weekend_bias": 0.0, "weekday_bias": 0.0, "periodic": False}

        # Check for weekend/weekday bias
        missing_df = pd.DataFrame({"date": missing_dates})
        missing_df["weekday"] = pd.to_datetime(missing_df["date"]).dt.weekday

        weekend_missing = len(missing_df[missing_df["weekday"].isin([5, 6])])
        weekday_missing = len(missing_df[~missing_df["weekday"].isin([5, 6])])

        total_missing = len(missing_dates)

        return {
            "weekend_bias": weekend_missing / total_missing if total_missing > 0 else 0,
            "weekday_bias": weekday_missing / total_missing if total_missing > 0 else 0,
            "periodic": self._check_periodic_pattern(missing_dates),
        }

    def _check_periodic_pattern(self, missing_dates: list[date]) -> bool:
        """Check if missing dates follow a periodic pattern."""
        if len(missing_dates) < 3:
            return False

        # Convert to ordinal days and check for periodicity
        ordinals = [d.toordinal() for d in missing_dates]
        gaps = np.diff(ordinals)

        # Simple periodicity check - are gaps consistent?
        if len(set(gaps)) == 1:
            return True

        # Check for weekly pattern
        weekdays = [d.weekday() for d in missing_dates]
        unique_weekdays = set(weekdays)
        if len(unique_weekdays) <= 2:  # Missing same days of week
            return True

        return False

    def align_sensors(
        self,
        sensor_data: dict[str, pd.DataFrame],
        strategy: AlignmentStrategy = AlignmentStrategy.INTERSECTION,
    ) -> pd.DataFrame:
        """
        Align data from multiple sensors with different coverage.

        Args:
            sensor_data: Dict of sensor_name -> DataFrame with date/timestamp index
            strategy: How to handle non-overlapping periods

        Returns:
            Aligned DataFrame with all sensors
        """
        if not sensor_data:
            return pd.DataFrame()

        # Standardize to daily if using daily aggregation
        if strategy == AlignmentStrategy.AGGREGATE_TO_DAILY:
            aligned_data = {}
            for name, df in sensor_data.items():
                # Set proper datetime index if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "timestamp" in df.columns:
                        df = df.set_index("timestamp")
                    elif "date" in df.columns:
                        df = df.set_index("date")

                # Check if index has time component (more than daily resolution)
                if (
                    len(df) > 0
                    and hasattr(df.index, "hour")
                    and any(df.index.hour != 0)
                ):
                    # Aggregate to daily
                    daily = df.resample("D").agg(
                        {
                            col: (
                                "mean"
                                if df[col].dtype in ["float64", "int64"]
                                else "first"
                            )
                            for col in df.columns
                        }
                    )
                    # Rename columns to indicate aggregation
                    daily.columns = [
                        (
                            f"{name}_{col}_mean"
                            if df[col].dtype in ["float64", "int64"]
                            else f"{name}_{col}"
                        )
                        for col in df.columns
                    ]
                    aligned_data[name] = daily
                else:
                    # Already daily
                    df_copy = df.copy()
                    df_copy.columns = [f"{name}_{col}" for col in df.columns]
                    aligned_data[name] = df_copy

            # Merge all sensors
            result = None
            for _, df in aligned_data.items():
                if result is None:
                    result = df
                else:
                    # For AGGREGATE_TO_DAILY, default to inner join to get overlapping days
                    result = result.join(df, how="inner")

            return result

        # For non-aggregating strategies
        elif strategy == AlignmentStrategy.INTERSECTION:
            # Find common dates
            common_index = None
            renamed_dfs = []

            for name, df in sensor_data.items():
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "date" in df.columns:
                        df = df.set_index("date")
                    elif "timestamp" in df.columns:
                        df = df.set_index("timestamp")

                # Rename columns to avoid conflicts
                df_renamed = df.copy()
                df_renamed.columns = [f"{name}_{col}" for col in df.columns]
                renamed_dfs.append(df_renamed)

                if common_index is None:
                    common_index = df_renamed.index
                else:
                    common_index = common_index.intersection(df_renamed.index)

            # Combine on common index
            result = pd.DataFrame(index=common_index)
            for df in renamed_dfs:
                result = result.join(df, how="left")

            return result

        elif strategy == AlignmentStrategy.UNION:
            # Find union of all dates
            union_index = None
            renamed_dfs = []

            for name, df in sensor_data.items():
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "date" in df.columns:
                        df = df.set_index("date")
                    elif "timestamp" in df.columns:
                        df = df.set_index("timestamp")

                # Rename columns to avoid conflicts
                df_renamed = df.copy()
                df_renamed.columns = [f"{name}_{col}" for col in df.columns]
                renamed_dfs.append(df_renamed)

                if union_index is None:
                    union_index = df_renamed.index
                else:
                    union_index = union_index.union(df_renamed.index)

            # Create result with union of all timestamps
            if union_index is None:
                return pd.DataFrame()

            result = pd.DataFrame(index=union_index.sort_values())

            # Join all sensor data with outer join (preserves all timestamps)
            for df in renamed_dfs:
                result = result.join(df, how="left")

            return result

        elif strategy == AlignmentStrategy.INTERPOLATE_ALIGN:
            # Create a common time grid and interpolate all sensors to it
            if not sensor_data:
                return pd.DataFrame()

            # Step 1: Determine the common time grid (daily resolution)
            all_dates = set()
            processed_sensors = {}

            for name, df in sensor_data.items():
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "date" in df.columns:
                        df = df.set_index("date")
                    elif "timestamp" in df.columns:
                        df = df.set_index("timestamp")

                # Convert to daily resolution
                if (
                    len(df) > 0
                    and hasattr(df.index, "hour")
                    and any(df.index.hour != 0)
                ):
                    # High-frequency data: aggregate to daily
                    daily_agg = df.resample("D").agg(
                        {
                            col: (
                                "mean"
                                if df[col].dtype in ["float64", "int64"]
                                else "first"
                            )
                            for col in df.columns
                        }
                    )
                    processed_sensors[name] = daily_agg
                else:
                    # Already daily or lower frequency
                    processed_sensors[name] = df.copy()

                # Collect all dates
                all_dates.update(processed_sensors[name].index.normalize())

            # Step 2: Create common daily grid
            if not all_dates:
                return pd.DataFrame()

            common_grid = pd.date_range(
                start=min(all_dates), end=max(all_dates), freq="D"
            )

            # Step 3: Interpolate each sensor to the common grid
            result = pd.DataFrame(index=common_grid)

            for name, sensor_df in processed_sensors.items():
                # Rename columns to avoid conflicts
                sensor_df = sensor_df.copy()
                sensor_df.columns = [f"{name}_{col}" for col in sensor_df.columns]

                # Reindex to common grid with interpolation
                aligned_sensor = sensor_df.reindex(common_grid)

                # Apply interpolation using the existing interpolation method
                aligned_sensor = self.interpolate(
                    aligned_sensor,
                    method=InterpolationMethod.LINEAR,
                    max_gap_hours=72,  # Allow up to 3 days gaps
                )

                # Join to result
                result = result.join(aligned_sensor, how="left")

            return result

        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented")

    def interpolate(
        self,
        df: pd.DataFrame,
        method: InterpolationMethod = InterpolationMethod.LINEAR,
        max_gap_hours: int | None = None,
    ) -> pd.DataFrame:
        """
        Interpolate missing values using Strategy Pattern.

        Args:
            df: DataFrame with missing values
            method: Interpolation method to use
            max_gap_hours: Maximum gap to interpolate (default: instance setting)

        Returns:
            DataFrame with interpolated values
        """
        if max_gap_hours is None:
            max_gap_hours = self.max_interpolation_gap_hours

        # Use strategy pattern for clean separation of concerns
        strategy = InterpolationStrategyFactory.create(method, max_gap_hours)
        return strategy.interpolate(df)

    def extract_missingness_features(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Extract features from missing data patterns.

        Args:
            df: DataFrame with potential missing values

        Returns:
            Dictionary of missingness features
        """
        features = {}

        # Overall missing ratio
        features["missing_ratio"] = df.isna().sum().sum() / (df.shape[0] * df.shape[1])

        # Per-column missing
        for col in df.columns:
            col_missing = df[col].isna()
            features[f"{col}_missing_ratio"] = col_missing.mean()

            # Consecutive missing
            if col_missing.any():
                consecutive = []
                current = 0
                for is_missing in col_missing:
                    if is_missing:
                        current += 1
                    else:
                        if current > 0:
                            consecutive.append(current)
                        current = 0
                if current > 0:
                    consecutive.append(current)

                features[f"{col}_max_consecutive_missing"] = (
                    max(consecutive) if consecutive else 0
                )

        # Weekend/weekday patterns
        if isinstance(df.index, pd.DatetimeIndex):
            weekday_mask = df.index.weekday < 5
            weekend_mask = ~weekday_mask

            if weekday_mask.any():
                features["missing_weekday_ratio"] = df[
                    weekday_mask
                ].isna().sum().sum() / (weekday_mask.sum() * df.shape[1])

            if weekend_mask.any():
                features["missing_weekend_ratio"] = df[
                    weekend_mask
                ].isna().sum().sum() / (weekend_mask.sum() * df.shape[1])

        # Always include max_consecutive_missing for backward compatibility
        max_consecutive = 0
        for col in df.columns:
            col_max = features.get(f"{col}_max_consecutive_missing", 0)
            max_consecutive = max(max_consecutive, col_max)
        features["max_consecutive_missing"] = max_consecutive

        return features

    def compute_confidence(self, dates: list[date] | pd.DatetimeIndex) -> float:
        """
        Compute confidence score based on data density.

        Args:
            dates: Available dates

        Returns:
            Confidence score between 0 and 1
        """
        if isinstance(dates, pd.DatetimeIndex):
            dates = [d.date() for d in dates]

        if not dates:
            return 0.0

        density = self.assess_density(dates)

        # Base confidence on coverage and gaps
        coverage_score = density.coverage_ratio
        gap_penalty = min(density.max_gap_days / 7.0, 1.0)  # Penalty for gaps > 7 days
        consecutive_bonus = min(
            density.consecutive_days / 7.0, 1.0
        )  # Bonus for consecutive days

        confidence = (
            coverage_score * 0.5 + (1 - gap_penalty) * 0.3 + consecutive_bonus * 0.2
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def select_processing_strategy(self, dates: list[date]) -> ProcessingStrategy:
        """
        Select appropriate processing strategy based on data characteristics.

        Args:
            dates: Available dates

        Returns:
            ProcessingStrategy with algorithm selection
        """
        density = self.assess_density(dates)
        confidence = self.compute_confidence(dates)

        if density.density_class == DataDensity.DENSE:
            return ProcessingStrategy(
                algorithm="full_circadian_analysis",
                confidence=confidence,
                interpolation_method=None,
                features_available=[
                    "circadian_phase",
                    "circadian_amplitude",
                    "weekly_patterns",
                    "all_sleep_metrics",
                ],
                warnings=[],
            )

        elif density.density_class == DataDensity.SPARSE:
            return ProcessingStrategy(
                algorithm="interpolation_with_uncertainty",
                confidence=confidence,
                interpolation_method=InterpolationMethod.CIRCADIAN_SPLINE,
                features_available=["basic_sleep_metrics", "activity_summary"],
                warnings=["Sparse data - confidence intervals widened"],
            )

        else:  # VERY_SPARSE
            return ProcessingStrategy(
                algorithm="minimal_analysis",
                confidence=confidence,
                interpolation_method=InterpolationMethod.NONE,
                features_available=["data_available_flag", "basic_statistics"],
                warnings=["Very sparse data - results may be unreliable"],
            )

    def get_available_features(
        self, dates: list[date] | pd.DatetimeIndex
    ) -> dict[str, bool]:
        """
        Determine which features can be calculated given data density.

        Args:
            dates: Available dates

        Returns:
            Dictionary of feature_name -> availability
        """
        if isinstance(dates, pd.DatetimeIndex):
            dates = [d.date() for d in dates]

        density = self.assess_density(dates)
        self.compute_confidence(dates)

        features = {
            "confidence": True,  # Always available
            "basic_statistics": True,
            "daily_summary": density.consecutive_days >= 1,
            "weekly_patterns": density.consecutive_days >= 7,
            "circadian_phase": density.consecutive_days >= 7
            and density.coverage_ratio > 0.7,
            "circadian_amplitude": density.consecutive_days >= 7
            and density.coverage_ratio > 0.7,
            "monthly_trends": density.total_days >= 28,
            "seasonal_patterns": density.total_days >= 90,
        }

        return features

    def find_analysis_windows(
        self,
        sleep_dates: list[date],
        activity_dates: list[date],
        min_overlap_days: int = 3,
    ) -> list[tuple[date, date]]:
        """
        Find windows where multiple sensors have sufficient overlap.

        Args:
            sleep_dates: Dates with sleep data
            activity_dates: Dates with activity data
            min_overlap_days: Minimum days of overlap required

        Returns:
            List of (start_date, end_date) tuples for analysis windows
        """
        sleep_set = set(sleep_dates)
        activity_set = set(activity_dates)

        overlap = sorted(sleep_set.intersection(activity_set))

        if len(overlap) < min_overlap_days:
            return []

        # Find consecutive windows
        windows = []
        current_start = overlap[0]
        current_end = overlap[0]

        for i in range(1, len(overlap)):
            if (overlap[i] - overlap[i - 1]).days == 1:
                current_end = overlap[i]
            else:
                if (current_end - current_start).days + 1 >= min_overlap_days:
                    windows.append((current_start, current_end))
                current_start = overlap[i]
                current_end = overlap[i]

        if (current_end - current_start).days + 1 >= min_overlap_days:
            windows.append((current_start, current_end))

        return windows

    def get_recommendation(self, windows: list[tuple[date, date]]) -> str:
        """Get recommendation based on available windows."""
        if not windows:
            return "Use sensor-specific analysis"
        elif len(windows) == 1 and (windows[0][1] - windows[0][0]).days < 7:
            return "Limited overlap - use with caution"
        else:
            return "Sufficient overlap for integrated analysis"

    def find_consecutive_windows(
        self, dates: list[date], min_consecutive_days: int = 3
    ) -> list[list[date]]:
        """
        Find windows of consecutive days.

        Args:
            dates: List of dates
            min_consecutive_days: Minimum consecutive days required

        Returns:
            List of consecutive date windows
        """
        if not dates:
            return []

        dates = sorted(dates)
        windows = []
        current_window = [dates[0]]

        for i in range(1, len(dates)):
            if (dates[i] - dates[i - 1]).days == 1:
                current_window.append(dates[i])
            else:
                if len(current_window) >= min_consecutive_days:
                    windows.append(current_window)
                current_window = [dates[i]]

        if len(current_window) >= min_consecutive_days:
            windows.append(current_window)

        return windows
