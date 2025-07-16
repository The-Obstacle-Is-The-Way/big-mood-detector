"""
PAT Sequence Builder Service

Builds 7-day activity sequences for the Pretrained Actigraphy Transformer (PAT).
Based on the Dartmouth study: "AI Foundation Models for Wearable Movement Data"

Key requirements:
- 10,080 minute sequences (7 days × 24 hours × 60 minutes)
- Patch size of 18 minutes (560 patches for PAT-L)
- Z-score normalization
- Handles missing data with zero-filling

Design Principles:
- Builder Pattern for sequence construction
- Immutable value objects
- Clear separation of concerns
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
    MinuteLevelSequence,
)


@dataclass(frozen=True)
class PATSequence:
    """
    Immutable 7-day activity sequence ready for PAT model.

    Contains 10,080 activity values (one per minute for 7 days).
    """

    end_date: date  # Last day of the sequence
    activity_values: np.ndarray  # Shape: (10080,)
    missing_days: List[date]  # Days with no data
    data_quality_score: float  # 0-1, based on completeness

    @property
    def start_date(self) -> date:
        """First day of the sequence."""
        return self.end_date - timedelta(days=6)

    @property
    def num_patches(self) -> int:
        """Number of patches based on patch size."""
        # PAT-L uses patch size 9, PAT-M/S use 18
        return len(self.activity_values) // 18  # Default to 18

    @property
    def is_complete(self) -> bool:
        """Check if all 7 days have data."""
        return len(self.missing_days) == 0

    def to_patches(self, patch_size: int = 18) -> np.ndarray:
        """
        Convert to patches for transformer input.

        Args:
            patch_size: Minutes per patch (9 for PAT-L, 18 for PAT-M/S)

        Returns:
            Array of shape (num_patches, patch_size)
        """
        num_patches = len(self.activity_values) // patch_size
        return self.activity_values[: num_patches * patch_size].reshape(
            num_patches, patch_size
        )

    def get_normalized(self) -> np.ndarray:
        """
        Get z-score normalized sequence.

        Returns:
            Normalized activity values
        """
        # Avoid division by zero
        mean = np.mean(self.activity_values)
        std = np.std(self.activity_values)

        if std > 0:
            return (self.activity_values - mean) / std
        else:
            return self.activity_values - mean


class PATSequenceBuilder:
    """
    Builds 7-day sequences for PAT model input.

    Handles:
    - Combining daily minute sequences
    - Missing data interpolation
    - Quality assessment
    - Normalization
    """

    SEQUENCE_DAYS = 7
    MINUTES_PER_DAY = 1440
    TOTAL_MINUTES = SEQUENCE_DAYS * MINUTES_PER_DAY  # 10,080

    def __init__(self, sequence_extractor: Optional[ActivitySequenceExtractor] = None):
        """
        Initialize builder with optional custom extractor.

        Args:
            sequence_extractor: Custom extractor or use default
        """
        self.extractor = sequence_extractor or ActivitySequenceExtractor()

    def build_sequence(
        self,
        activity_records: List[ActivityRecord],
        end_date: date,
        interpolate_missing: bool = True,
    ) -> PATSequence:
        """
        Build a 7-day sequence ending on the specified date.

        Args:
            activity_records: All available activity records
            end_date: Last day of the sequence
            interpolate_missing: Whether to interpolate missing days

        Returns:
            PATSequence ready for model input
        """
        # Calculate date range
        start_date = end_date - timedelta(days=self.SEQUENCE_DAYS - 1)

        # Extract daily sequences
        daily_sequences = []
        missing_days = []

        for day_offset in range(self.SEQUENCE_DAYS):
            current_date = start_date + timedelta(days=day_offset)

            # Extract sequence for this day
            day_sequence = self.extractor.extract_daily_sequence(
                activity_records, current_date
            )

            # Check if day has data
            if day_sequence.total_activity == 0:
                missing_days.append(current_date)

            daily_sequences.append(day_sequence)

        # Combine into single array
        combined_values = self._combine_sequences(daily_sequences)

        # Interpolate if requested and needed
        if interpolate_missing and missing_days:
            combined_values = self._interpolate_missing_days(
                combined_values, missing_days, start_date
            )

        # Calculate quality score
        quality_score = 1.0 - (len(missing_days) / self.SEQUENCE_DAYS)

        return PATSequence(
            end_date=end_date,
            activity_values=combined_values,
            missing_days=missing_days,
            data_quality_score=quality_score,
        )

    def build_multiple_sequences(
        self,
        activity_records: List[ActivityRecord],
        start_date: date,
        end_date: date,
        stride_days: int = 1,
    ) -> List[PATSequence]:
        """
        Build multiple overlapping sequences for a date range.

        Args:
            activity_records: All available activity records
            start_date: First possible sequence end date
            end_date: Last possible sequence end date
            stride_days: Days between sequence end dates

        Returns:
            List of PATSequences
        """
        sequences = []

        # Ensure we have at least 7 days before start_date
        current_end = max(start_date, start_date + timedelta(days=6))

        while current_end <= end_date:
            sequence = self.build_sequence(activity_records, current_end)
            sequences.append(sequence)

            current_end += timedelta(days=stride_days)

        return sequences

    def _combine_sequences(
        self, daily_sequences: List[MinuteLevelSequence]
    ) -> np.ndarray:
        """
        Combine daily sequences into a single 7-day array.

        Args:
            daily_sequences: List of 7 daily sequences

        Returns:
            Combined array of shape (10080,)
        """
        # Stack all activity values
        all_values = []
        for seq in daily_sequences:
            all_values.extend(seq.activity_values)

        return np.array(all_values, dtype=np.float32)

    def _interpolate_missing_days(
        self, values: np.ndarray, missing_days: List[date], start_date: date
    ) -> np.ndarray:
        """
        Interpolate missing days using neighboring data.

        Simple strategy: Use average of adjacent days or zeros.

        Args:
            values: Combined activity values
            missing_days: Days with no data
            start_date: First day of sequence

        Returns:
            Interpolated values
        """
        # For now, keep zeros for missing days
        # In production, could use more sophisticated interpolation
        # (e.g., copy pattern from same day of week, or use average profile)

        return values

    def calculate_pat_features(self, sequence: PATSequence) -> dict:
        """
        Calculate PAT-specific features from sequence.

        These can be used alongside or instead of raw sequences.

        Args:
            sequence: 7-day activity sequence

        Returns:
            Dictionary of PAT features
        """
        values = sequence.activity_values

        # Calculate circadian features
        hourly_totals = values.reshape(7 * 24, 60).sum(axis=1)

        # Principal Activity Time (center of mass)
        hours = np.arange(7 * 24)
        if hourly_totals.sum() > 0:
            pat_hour = np.average(hours, weights=hourly_totals) % 24
        else:
            pat_hour = 0.0

        # Activity fragmentation
        transitions = np.sum(np.diff(values > 0))
        fragmentation = transitions / len(values)

        # Peak activity periods
        sorted_values = np.sort(values)[::-1]
        peak_10pct = sorted_values[: int(0.1 * len(values))].mean()

        return {
            "pat_hour": pat_hour,
            "fragmentation": fragmentation,
            "peak_activity": peak_10pct,
            "total_activity": values.sum(),
            "active_minutes": np.sum(values > 0),
            "quality_score": sequence.data_quality_score,
        }
