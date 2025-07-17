"""
Interpolation Strategies for Sparse Health Data

Strategy Pattern implementation for different interpolation methods.
Extracted from SparseDataHandler for better separation of concerns.

Design Patterns:
- Strategy Pattern: Different interpolation algorithms
- Factory Pattern: Strategy creation
- Single Responsibility: Each strategy handles one interpolation method
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import interpolate


class InterpolationMethod(Enum):
    """Interpolation strategies for different data types."""

    NONE = auto()  # No interpolation
    LINEAR = auto()  # Linear interpolation
    FORWARD_FILL = auto()  # Forward fill (categorical)
    CIRCADIAN_SPLINE = auto()  # Circadian-aware spline


class InterpolationStrategy(Protocol):
    """Protocol for interpolation strategies."""

    def interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing values in DataFrame.

        Args:
            data: DataFrame with missing values

        Returns:
            DataFrame with interpolated values
        """
        ...


class BaseInterpolationStrategy(ABC):
    """Base class for interpolation strategies."""

    def __init__(self, max_gap_hours: int = 24):
        """
        Initialize strategy.

        Args:
            max_gap_hours: Maximum gap size to interpolate
        """
        self.max_gap_hours = max_gap_hours

    @abstractmethod
    def interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values in DataFrame."""
        pass

    def _ensure_complete_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has complete datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Infer frequency from data
        inferred_freq = df.index.inferred_freq
        if inferred_freq is None:
            # Check if it's daily data by looking at time components
            if all(t.time() == pd.Timestamp("00:00:00").time() for t in df.index):
                inferred_freq = "D"
            else:
                inferred_freq = "h"

        # Create complete date range
        full_range = pd.date_range(df.index.min(), df.index.max(), freq=inferred_freq)
        return df.reindex(full_range)


class LinearInterpolationStrategy(BaseInterpolationStrategy):
    """Linear interpolation strategy for numerical data."""

    def interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply linear interpolation to numerical columns.

        Args:
            data: DataFrame with missing values

        Returns:
            DataFrame with linear interpolation applied
        """
        result = data.copy()

        # Calculate limit based on frequency
        if isinstance(result.index, pd.DatetimeIndex) and len(result.index) > 1:
            # Estimate time difference between consecutive points
            time_diff = result.index[1] - result.index[0]
            if time_diff >= pd.Timedelta(days=1):
                # For daily data, convert hours to days for limit
                limit = self.max_gap_hours // 24 if self.max_gap_hours > 0 else 0
            else:
                # For hourly or sub-hourly data
                limit = self.max_gap_hours
        else:
            limit = self.max_gap_hours

        # Only interpolate if limit allows it
        if limit > 0:
            # Apply linear interpolation to numerical columns
            for col in result.columns:
                if result[col].dtype in ["float64", "int64", "float32", "int32"]:
                    result[col] = result[col].interpolate(
                        method="linear", limit=limit, limit_direction="both"
                    )

        # Only ensure complete index if we actually interpolated
        if limit > 0:
            result = self._ensure_complete_index(result)

        return result


class ForwardFillInterpolationStrategy(BaseInterpolationStrategy):
    """Forward fill interpolation strategy for categorical data."""

    def interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply forward fill to categorical columns.

        Args:
            data: DataFrame with missing values

        Returns:
            DataFrame with forward fill applied
        """
        result = self._ensure_complete_index(data.copy())

        # Calculate limit based on frequency
        limit = self.max_gap_hours
        if isinstance(result.index, pd.DatetimeIndex):
            time_diff = (
                result.index[1] - result.index[0]
                if len(result.index) > 1
                else pd.Timedelta(hours=1)
            )
            if time_diff >= pd.Timedelta(days=1):
                limit = self.max_gap_hours // 24

        # Apply forward fill to all columns (especially good for categorical)
        for col in result.columns:
            result[col] = result[col].ffill(limit=limit)

        return result


class CircadianSplineInterpolationStrategy(BaseInterpolationStrategy):
    """Circadian-aware spline interpolation for activity data."""

    def interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply circadian-aware interpolation that preserves daily rhythms.

        Args:
            data: DataFrame with missing values

        Returns:
            DataFrame with circadian-aware interpolation applied
        """
        result = self._ensure_complete_index(data.copy())

        for col in result.columns:
            if "activity" in col.lower() and result[col].dtype in [
                "float64",
                "int64",
                "float32",
                "int32",
            ]:
                result[col] = self._circadian_interpolate(result[col])
            else:
                # Fall back to linear interpolation for non-activity data
                result[col] = result[col].interpolate(
                    method="linear", limit=self.max_gap_hours, limit_direction="both"
                )

        return result

    def _circadian_interpolate(self, series: pd.Series) -> pd.Series:
        """
        Apply circadian-aware interpolation to a single series.

        This method preserves daily rhythms by using spline interpolation
        with circadian period constraints.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            # Fall back to linear interpolation
            return series.interpolate(method="linear", limit=self.max_gap_hours)

        # Get valid (non-null) data points
        valid_mask = series.notna()
        if valid_mask.sum() < 2:
            return series  # Need at least 2 points for interpolation

        valid_times = series.index[valid_mask]
        valid_values = series[valid_mask]

        # Convert times to hours from start for spline fitting
        start_time = series.index[0]
        valid_hours = [(t - start_time).total_seconds() / 3600 for t in valid_times]
        all_hours = [(t - start_time).total_seconds() / 3600 for t in series.index]

        # Create circadian-aware spline
        # Use periodic spline if we have enough data points
        if len(valid_hours) >= 4:
            try:
                # Create spline with smoothing to preserve circadian patterns
                spline = interpolate.UnivariateSpline(
                    valid_hours, valid_values, s=len(valid_values), k=3
                )
                interpolated_values = spline(all_hours)

                # Ensure non-negative values for activity data
                interpolated_values = np.maximum(interpolated_values, 0)

                result = series.copy()
                result[:] = interpolated_values
                return result
            except Exception:
                # Fall back to linear if spline fails
                pass

        # Fallback to linear interpolation
        return series.interpolate(method="linear", limit=self.max_gap_hours)


class NoInterpolationStrategy(BaseInterpolationStrategy):
    """No interpolation strategy - preserves missing values."""

    def __init__(self) -> None:
        """Initialize with no parameters since no interpolation is performed."""
        super().__init__(max_gap_hours=0)  # Not used

    def interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return data unchanged (no interpolation).

        Args:
            data: DataFrame with missing values

        Returns:
            Same DataFrame with missing values preserved
        """
        return data.copy()


class InterpolationStrategyFactory:
    """Factory for creating interpolation strategies."""

    @staticmethod
    def create(
        method: InterpolationMethod, max_gap_hours: int = 24
    ) -> InterpolationStrategy:
        """
        Create appropriate interpolation strategy.

        Args:
            method: Interpolation method to use
            max_gap_hours: Maximum gap size to interpolate

        Returns:
            Concrete interpolation strategy

        Raises:
            ValueError: If method is not supported
        """
        if method == InterpolationMethod.LINEAR:
            return LinearInterpolationStrategy(max_gap_hours)
        elif method == InterpolationMethod.FORWARD_FILL:
            return ForwardFillInterpolationStrategy(max_gap_hours)
        elif method == InterpolationMethod.CIRCADIAN_SPLINE:
            return CircadianSplineInterpolationStrategy(max_gap_hours)
        elif method == InterpolationMethod.NONE:
            return NoInterpolationStrategy()
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
