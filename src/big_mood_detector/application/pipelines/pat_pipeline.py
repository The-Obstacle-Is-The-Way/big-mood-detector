"""
PAT (Pretrained Actigraphy Transformer) Pipeline.

Independent pipeline for current depression state assessment
using 7 consecutive days of activity data.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from big_mood_detector.application.validators.pipeline_validators import (
    PATValidator,
    ValidationResult,
)
from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class PATResult:
    """Result from PAT depression assessment."""

    depression_risk_score: float  # 0.0 to 1.0
    confidence: float  # Model confidence 0.0 to 1.0
    assessment_window_days: int  # Should be 7
    model_version: str  # PAT-S, PAT-M, or PAT-L
    clinical_interpretation: str
    window_start_date: date
    window_end_date: date


class PatPipeline:
    """
    Independent pipeline for PAT depression screening.

    This pipeline:
    1. Validates that 7 consecutive days of activity data are available
    2. Finds the best 7-day window (closest to target date)
    3. Extracts minute-level activity sequences (10,080 minutes)
    4. Runs PAT model for current depression assessment
    """

    def __init__(
        self,
        pat_loader,  # PAT model loader (avoiding circular imports)
        validator: PATValidator,
        model_size: str = "L",  # Default to largest model
    ):
        """
        Initialize PAT pipeline.

        Args:
            pat_loader: PAT model loader instance
            validator: PAT data validator
            model_size: Model size (S, M, or L)
        """
        self.pat_loader = pat_loader
        self.validator = validator
        self.model_size = model_size
        self.sequence_extractor = ActivitySequenceExtractor()

    def can_run(
        self,
        activity_records: list[ActivityRecord],
        start_date: date,
        end_date: date,
    ) -> ValidationResult:
        """
        Check if PAT can run with available data.

        Args:
            activity_records: Available activity records
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            ValidationResult with details about data sufficiency
        """
        return self.validator.validate(
            activity_records=activity_records,
            start_date=start_date,
            end_date=end_date,
        )

    def process(
        self,
        activity_records: list[ActivityRecord],
        target_date: date,
    ) -> Optional[PATResult]:
        """
        Process activity data through PAT pipeline.

        Args:
            activity_records: All available activity records
            target_date: Date to assess (finds best 7-day window ending near this date)

        Returns:
            PATResult if sufficient data, None otherwise
        """
        # Find best 7-day consecutive window
        window = self._find_best_window(activity_records, target_date)
        if not window:
            logger.info("No valid 7-day consecutive window found for PAT")
            return None

        window_start, window_end = window
        logger.info(
            f"PAT using window: {window_start} to {window_end} "
            f"(target was {target_date})"
        )

        # Filter records to window
        window_records = [
            r for r in activity_records
            if window_start <= r.start_date.date() <= window_end
        ]

        # Extract minute-level sequence (10,080 minutes for 7 days)
        sequences = []
        for day_offset in range(7):
            current_date = window_start + timedelta(days=day_offset)
            day_records = [
                r for r in window_records
                if r.start_date.date() == current_date
            ]
            
            if day_records:
                # Extract 1440-minute sequence for this day
                day_sequence = self.sequence_extractor.extract_daily_sequence(
                    day_records, current_date
                )
                sequences.append(day_sequence)
            else:
                # No data for this day - use zeros
                sequences.append([0.0] * 1440)

        # Flatten to single 10,080-minute sequence
        full_sequence = []
        for seq in sequences:
            if isinstance(seq, list):
                full_sequence.extend(seq)
            else:
                # seq is MinuteLevelSequence object
                full_sequence.extend(seq.activity_values)

        # Run PAT model
        try:
            result = self.pat_loader.extract_embeddings([full_sequence])
            
            # Extract predictions
            depression_score = result.get("depression_score", 0.0)
            confidence = result.get("confidence", 0.0)
            
            # Clinical interpretation
            if depression_score < 0.3:
                interpretation = "Low risk for current depression"
            elif depression_score < 0.5:
                interpretation = "Moderate risk for current depression - monitor closely"
            elif depression_score < 0.7:
                interpretation = "Elevated risk for current depression - consider clinical evaluation"
            else:
                interpretation = "High risk for current depression - clinical evaluation recommended"

            return PATResult(
                depression_risk_score=depression_score,
                confidence=confidence,
                assessment_window_days=7,
                model_version=f"PAT-{self.model_size}",
                clinical_interpretation=f"Current depression risk: {interpretation}",
                window_start_date=window_start,
                window_end_date=window_end,
            )

        except Exception as e:
            logger.error(f"PAT model error: {e}")
            return None

    def _find_best_window(
        self,
        activity_records: list[ActivityRecord],
        target_date: date,
    ) -> Optional[tuple[date, date]]:
        """
        Find the best 7-day consecutive window closest to target date.

        Args:
            activity_records: All activity records
            target_date: Preferred end date

        Returns:
            Tuple of (start_date, end_date) or None if no valid window
        """
        if not activity_records:
            return None

        # Get unique dates with data
        dates_with_data = sorted({r.start_date.date() for r in activity_records})

        # Find all possible 7-day consecutive windows
        valid_windows = []
        
        for i in range(len(dates_with_data) - 6):
            # Check if we have 7 consecutive days
            window_start = dates_with_data[i]
            expected_end = window_start + timedelta(days=6)
            actual_end = dates_with_data[i + 6]
            
            if actual_end == expected_end:
                # Valid 7-day window
                valid_windows.append((window_start, actual_end))

        if not valid_windows:
            return None

        # Find window closest to target date
        best_window = min(
            valid_windows,
            key=lambda w: abs((w[1] - target_date).days)
        )

        return best_window