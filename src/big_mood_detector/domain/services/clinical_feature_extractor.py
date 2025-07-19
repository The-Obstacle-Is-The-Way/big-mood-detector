"""
Clinical Feature Extractor Service

Integrates all domain services to extract comprehensive clinical features
for mood disorder detection following the Seoul XGBoost study approach.

Key Features:
1. Sleep window analysis with 3.75h merging
2. DLMO calculation with activity-based prediction
3. Circadian rhythm metrics (IS, IV, RA, L5/M10)
4. PAT sequence generation for transformer models
5. Z-score normalization with individual baselines
6. Clinical significance detection

Design Principles:
- Facade pattern for service orchestration
- Immutable value objects for feature sets
- Single Responsibility: Feature extraction only
- Clean Architecture: Domain logic only
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
)
from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
)
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator
from big_mood_detector.domain.services.pat_sequence_builder import (
    PATSequence,
    PATSequenceBuilder,
)
from big_mood_detector.domain.services.sleep_window_analyzer import SleepWindowAnalyzer


@dataclass(frozen=True)
class SeoulXGBoostFeatures:
    """
    Complete 36-feature set from Seoul National University study.

    These features achieved AUC 0.80-0.98 for mood episode prediction.
    """

    date: date

    # Basic Sleep Features (1-5)
    sleep_duration_hours: float
    sleep_efficiency: float
    sleep_onset_hour: float  # 24h format
    wake_time_hour: float
    sleep_fragmentation: float

    # Advanced Sleep Features (6-10)
    sleep_regularity_index: float  # 0-100
    short_sleep_window_pct: float  # % < 6 hours
    long_sleep_window_pct: float  # % > 10 hours
    sleep_onset_variance: float
    wake_time_variance: float

    # Circadian Rhythm Features (11-18)
    interdaily_stability: float  # IS: 0-1
    intradaily_variability: float  # IV: 0-2
    relative_amplitude: float  # RA: 0-1
    l5_value: float  # Least active 5 hours
    m10_value: float  # Most active 10 hours
    l5_onset_hour: float  # When L5 starts
    m10_onset_hour: float  # When M10 starts
    dlmo_hour: float  # Dim Light Melatonin Onset

    # Activity Features (19-24)
    total_steps: int
    activity_variance: float
    sedentary_hours: float
    activity_fragmentation: float
    sedentary_bout_mean: float
    activity_intensity_ratio: float

    # Heart Rate Features (25-28)
    avg_resting_hr: float
    hrv_sdnn: float
    hr_circadian_range: float
    hr_minimum_hour: float

    # Phase Features (29-32)
    circadian_phase_advance: float  # Hours ahead
    circadian_phase_delay: float  # Hours behind
    dlmo_confidence: float  # DLMO calculation confidence
    pat_hour: float  # Principal activity time

    # Z-Score Features (33-36)
    sleep_duration_zscore: float
    activity_zscore: float
    hr_zscore: float
    hrv_zscore: float

    # Metadata
    data_completeness: float  # 0-1, fraction of required data available
    is_hypersomnia_pattern: bool = False
    is_insomnia_pattern: bool = False
    is_phase_advanced: bool = False
    is_phase_delayed: bool = False
    is_irregular_pattern: bool = False

    def to_xgboost_features(self) -> list[float]:
        """
        Convert to 36-element feature vector for XGBoost model.

        Order must match the Seoul study's trained model exactly.
        """
        return [
            # Basic sleep (1-5)
            self.sleep_duration_hours,
            self.sleep_efficiency,
            self.sleep_onset_hour,
            self.wake_time_hour,
            self.sleep_fragmentation,
            # Advanced sleep (6-10)
            self.sleep_regularity_index,
            self.short_sleep_window_pct,
            self.long_sleep_window_pct,
            self.sleep_onset_variance,
            self.wake_time_variance,
            # Circadian (11-18)
            self.interdaily_stability,
            self.intradaily_variability,
            self.relative_amplitude,
            self.l5_value,
            self.m10_value,
            self.l5_onset_hour,
            self.m10_onset_hour,
            self.dlmo_hour,
            # Activity (19-24)
            float(self.total_steps),
            self.activity_variance,
            self.sedentary_hours,
            self.activity_fragmentation,
            self.sedentary_bout_mean,
            self.activity_intensity_ratio,
            # Heart rate (25-28)
            self.avg_resting_hr,
            self.hrv_sdnn,
            self.hr_circadian_range,
            self.hr_minimum_hour,
            # Phase (29-32)
            self.circadian_phase_advance,
            self.circadian_phase_delay,
            self.dlmo_confidence,
            self.pat_hour,
            # Z-scores (33-36)
            self.sleep_duration_zscore,
            self.activity_zscore,
            self.hr_zscore,
            self.hrv_zscore,
        ]


@dataclass(frozen=True)
class ClinicalFeatureSet:
    """
    Complete clinical feature set for mood disorder detection.

    Includes both traditional features and deep learning inputs.
    """

    date: date
    seoul_features: SeoulXGBoostFeatures
    pat_sequence: PATSequence | None = None

    # Activity features - DIRECT ATTRIBUTES for API exposure
    # These extend the original Seoul study features
    total_steps: float = 0.0
    activity_variance: float = 0.0
    sedentary_hours: float = 24.0
    activity_fragmentation: float = 0.0
    sedentary_bout_mean: float = 24.0
    activity_intensity_ratio: float = 0.0

    # Clinical indicators
    is_clinically_significant: bool = False
    clinical_notes: list[str] = field(default_factory=list)

    # Risk scores (populated after model inference)
    depression_risk_score: float | None = None
    mania_risk_score: float | None = None
    hypomania_risk_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""

        result = {
            "date": self.date,
            # Direct activity features - use the API naming convention
            "daily_steps": self.total_steps,  # Map to API expected name
            "activity_variance": self.activity_variance,
            "sedentary_hours": self.sedentary_hours,
            "activity_fragmentation": self.activity_fragmentation,
            "sedentary_bout_mean": self.sedentary_bout_mean,
            "activity_intensity_ratio": self.activity_intensity_ratio,
        }

        # Add Seoul features if available
        if self.seoul_features:
            # For now, just export the activity-related features from Seoul
            # The full 36-feature mapping needs to be done properly
            seoul_dict = {
                "sleep_duration_hours": self.seoul_features.sleep_duration_hours,
                "sleep_efficiency": self.seoul_features.sleep_efficiency,
                "sleep_onset_hour": self.seoul_features.sleep_onset_hour,
                "wake_time_hour": self.seoul_features.wake_time_hour,
                # Note: Activity features already exported as direct attributes
                # Other Seoul features need proper mapping to XGBoost names
            }
            result.update(seoul_dict)

        return result


class ClinicalFeatureExtractor:
    """
    Orchestrates all domain services to extract clinical features.

    This is the main entry point for feature extraction in the
    mood disorder detection pipeline.
    """

    def __init__(self) -> None:
        """Initialize with all required domain services."""
        self.sleep_window_analyzer = SleepWindowAnalyzer()
        self.activity_sequence_extractor = ActivitySequenceExtractor()
        self.dlmo_calculator = DLMOCalculator()
        self.advanced_feature_engineer = AdvancedFeatureEngineer()
        self.pat_sequence_builder = PATSequenceBuilder(self.activity_sequence_extractor)

    def extract_seoul_features(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
        target_date: date,
        min_days_required: int = 7,
    ) -> SeoulXGBoostFeatures:
        """
        Extract all 36 features required by Seoul XGBoost model.

        Args:
            sleep_records: Historical sleep data
            activity_records: Historical activity data
            heart_records: Historical heart rate data
            target_date: Date to extract features for
            min_days_required: Minimum days of data needed

        Returns:
            Complete Seoul feature set
        """
        # First, we need to create daily summaries using the aggregators
        from big_mood_detector.domain.services.activity_aggregator import (
            ActivityAggregator,
        )
        from big_mood_detector.domain.services.heart_rate_aggregator import (
            HeartRateAggregator,
        )
        from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator

        sleep_aggregator = SleepAggregator()
        activity_aggregator = ActivityAggregator()
        heart_aggregator = HeartRateAggregator()

        # Get daily summaries for the past 30 days
        daily_sleep_summaries = sleep_aggregator.aggregate_daily(sleep_records)
        daily_activity_summaries = activity_aggregator.aggregate_daily(activity_records)
        daily_heart_summaries = heart_aggregator.aggregate_daily(heart_records)

        # Convert to lists for advanced feature engineer
        historical_sleep = list(daily_sleep_summaries.values())
        historical_activity = list(daily_activity_summaries.values())
        historical_heart = list(daily_heart_summaries.values())

        # Get advanced features from feature engineer
        advanced_features = self.advanced_feature_engineer.extract_advanced_features(
            current_date=target_date,
            historical_sleep=historical_sleep,
            historical_activity=historical_activity,
            historical_heart=historical_heart,
            lookback_days=30,
        )

        # Calculate DLMO
        dlmo_result = self.dlmo_calculator.calculate_dlmo(
            sleep_records=sleep_records,
            activity_records=activity_records,
            target_date=target_date,
            days_to_model=14,
            use_activity=bool(activity_records),
        )

        # Extract PAT (Principal Activity Time) from activity sequence
        pat_analysis = (
            self.activity_sequence_extractor.calculate_pat(
                records=activity_records, target_date=target_date
            )
            if activity_records
            else None
        )

        # Calculate data completeness
        days_with_sleep = len({s.start_date.date() for s in sleep_records})
        days_with_activity = len({a.start_date.date() for a in activity_records})
        data_completeness = min(days_with_sleep, days_with_activity) / max(
            14, min_days_required
        )

        # Build Seoul features
        return SeoulXGBoostFeatures(
            date=target_date,
            # Basic sleep
            sleep_duration_hours=advanced_features.sleep_duration_hours,
            sleep_efficiency=advanced_features.sleep_efficiency,
            sleep_onset_hour=self._extract_sleep_onset_hour(sleep_records, target_date),
            wake_time_hour=self._extract_wake_time_hour(sleep_records, target_date),
            sleep_fragmentation=self._calculate_sleep_fragmentation(
                sleep_records, target_date
            ),
            # Advanced sleep
            sleep_regularity_index=advanced_features.sleep_regularity_index,
            short_sleep_window_pct=advanced_features.short_sleep_window_pct,
            long_sleep_window_pct=advanced_features.long_sleep_window_pct,
            sleep_onset_variance=advanced_features.sleep_onset_variance,
            wake_time_variance=advanced_features.wake_time_variance,
            # Circadian
            interdaily_stability=advanced_features.interdaily_stability,
            intradaily_variability=advanced_features.intradaily_variability,
            relative_amplitude=advanced_features.relative_amplitude,
            l5_value=advanced_features.l5_value,
            m10_value=advanced_features.m10_value,
            l5_onset_hour=(
                advanced_features.l5_onset.hour if advanced_features.l5_onset else 2.0
            ),
            m10_onset_hour=(
                advanced_features.m10_onset.hour
                if advanced_features.m10_onset
                else 14.0
            ),
            dlmo_hour=dlmo_result.dlmo_hour if dlmo_result else 21.0,
            # Activity
            total_steps=advanced_features.total_steps,
            activity_variance=self._calculate_activity_variance(activity_records),
            sedentary_hours=advanced_features.sedentary_bout_mean
            / 60.0,  # Convert minutes to hours
            activity_fragmentation=advanced_features.activity_fragmentation,
            sedentary_bout_mean=advanced_features.sedentary_bout_mean,
            activity_intensity_ratio=advanced_features.activity_intensity_ratio,
            # Heart rate
            avg_resting_hr=advanced_features.avg_resting_hr,
            hrv_sdnn=advanced_features.hrv_sdnn,
            hr_circadian_range=self._calculate_hr_circadian_range(heart_records),
            hr_minimum_hour=self._find_hr_minimum_hour(heart_records, target_date),
            # Phase
            circadian_phase_advance=advanced_features.circadian_phase_advance,
            circadian_phase_delay=advanced_features.circadian_phase_delay,
            dlmo_confidence=dlmo_result.confidence if dlmo_result else 0.0,
            pat_hour=pat_analysis.pat_hour if pat_analysis else 14.0,
            # Z-scores
            sleep_duration_zscore=advanced_features.sleep_duration_zscore,
            activity_zscore=advanced_features.activity_zscore,
            hr_zscore=advanced_features.hr_zscore,
            hrv_zscore=advanced_features.hrv_zscore,
            # Metadata
            data_completeness=data_completeness,
            is_hypersomnia_pattern=advanced_features.is_hypersomnia_pattern,
            is_insomnia_pattern=advanced_features.is_insomnia_pattern,
            is_phase_advanced=advanced_features.is_phase_advanced,
            is_phase_delayed=advanced_features.is_phase_delayed,
            is_irregular_pattern=advanced_features.is_irregular_pattern,
        )

    def extract_pat_sequence(
        self, activity_records: list[ActivityRecord], end_date: date
    ) -> PATSequence | None:
        """
        Extract 7-day PAT sequence for transformer model.

        Args:
            activity_records: Activity data
            end_date: Last day of sequence

        Returns:
            PAT sequence or None if insufficient data
        """
        if not activity_records:
            return None

        return self.pat_sequence_builder.build_sequence(
            activity_records=activity_records,
            end_date=end_date,
            interpolate_missing=True,
        )

    def extract_clinical_features(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
        target_date: date,
        include_pat_sequence: bool = False,
    ) -> ClinicalFeatureSet:
        """
        Extract complete clinical feature set.

        Args:
            sleep_records: Historical sleep data
            activity_records: Historical activity data
            heart_records: Historical heart rate data
            target_date: Date to extract features for
            include_pat_sequence: Whether to include PAT sequence

        Returns:
            Complete clinical feature set
        """
        # Extract Seoul features
        seoul_features = self.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
        )

        # Extract PAT sequence if requested
        pat_sequence = None
        if include_pat_sequence:
            pat_sequence = self.extract_pat_sequence(
                activity_records=activity_records, end_date=target_date
            )

        # Detect clinical significance
        is_significant = False
        clinical_notes = []

        if seoul_features.is_insomnia_pattern:
            is_significant = True
            clinical_notes.append(
                "Insomnia pattern detected: consistently short sleep duration"
            )

        if seoul_features.is_hypersomnia_pattern:
            is_significant = True
            clinical_notes.append(
                "Hypersomnia pattern detected: consistently long sleep duration"
            )

        if seoul_features.is_irregular_pattern:
            is_significant = True
            clinical_notes.append(
                "Irregular sleep pattern: high variability in sleep timing"
            )

        if seoul_features.interdaily_stability < 0.3:
            is_significant = True
            clinical_notes.append("Very low circadian stability detected")

        return ClinicalFeatureSet(
            date=target_date,
            seoul_features=seoul_features,
            pat_sequence=pat_sequence,
            is_clinically_significant=is_significant,
            clinical_notes=clinical_notes,
        )

    def _extract_sleep_onset_hour(
        self, sleep_records: list[SleepRecord], target_date: date
    ) -> float:
        """Extract sleep onset hour for target date."""
        for record in sleep_records:
            if record.start_date.date() == target_date:
                return record.start_date.hour + record.start_date.minute / 60.0
        return 23.0  # Default

    def _extract_wake_time_hour(
        self, sleep_records: list[SleepRecord], target_date: date
    ) -> float:
        """Extract wake time hour for target date."""
        for record in sleep_records:
            if record.end_date.date() == target_date + timedelta(days=1):
                return record.end_date.hour + record.end_date.minute / 60.0
        return 7.0  # Default

    def _calculate_sleep_fragmentation(
        self, sleep_records: list[SleepRecord], target_date: date
    ) -> float:
        """Calculate sleep fragmentation index."""
        # Count number of sleep episodes on target date
        episodes = [r for r in sleep_records if r.start_date.date() == target_date]
        if not episodes:
            return 0.0

        # More episodes = more fragmentation
        return min(1.0, (len(episodes) - 1) / 3.0)

    def _calculate_activity_variance(
        self, activity_records: list[ActivityRecord]
    ) -> float:
        """Calculate variance in activity levels."""
        if not activity_records:
            return 0.0

        values = [r.value for r in activity_records]
        return float(np.var(values))

    def _calculate_hr_circadian_range(
        self, heart_records: list[HeartRateRecord]
    ) -> float:
        """Calculate circadian range in heart rate."""
        if not heart_records:
            return 0.0

        values = [r.value for r in heart_records]
        return max(values) - min(values)

    def _find_hr_minimum_hour(
        self, heart_records: list[HeartRateRecord], target_date: date
    ) -> float:
        """Find hour of minimum heart rate."""
        if not heart_records:
            return 4.0  # Default to typical minimum

        # Get records for target date
        date_records = [r for r in heart_records if r.timestamp.date() == target_date]
        if not date_records:
            return 4.0

        # Find minimum
        min_record = min(date_records, key=lambda r: r.value)
        return min_record.timestamp.hour + min_record.timestamp.minute / 60.0
