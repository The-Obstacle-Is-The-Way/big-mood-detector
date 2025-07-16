"""
Feature Extraction Service

Combines data from all sources to extract clinical features for mood prediction.
Following Domain-Driven Design and Clean Architecture principles.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.activity_aggregator import ActivityAggregator
from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
    AdvancedFeatures,
)
from big_mood_detector.domain.services.heart_rate_aggregator import (
    HeartRateAggregator,
)
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator


@dataclass(frozen=True)
class ClinicalFeatures:
    """
    Immutable clinical features for a single day.

    Represents the complete feature set for mood prediction.
    """

    date: date
    # Sleep features
    sleep_duration_hours: float = 0.0
    sleep_efficiency: float = 0.0
    sleep_fragmentation: float = 0.0
    sleep_onset_hour: float = 0.0  # 24-hour format (e.g., 23.5 = 11:30 PM)
    wake_time_hour: float = 0.0
    # Activity features
    total_steps: float = 0.0
    activity_variance: float = 0.0
    sedentary_hours: float = 0.0
    peak_activity_hour: int = 0
    # Heart features
    avg_resting_hr: float = 0.0
    hrv_sdnn: float = 0.0
    hr_circadian_range: float = 0.0
    # Circadian features
    circadian_alignment_score: float = 0.0
    # Clinical indicators
    is_clinically_significant: bool = False
    clinical_notes: list[str] = field(default_factory=list)

    def to_feature_vector(self) -> list[float]:
        """
        Convert to ML-ready feature vector.

        Returns numeric features in consistent order.
        """
        return [
            self.sleep_duration_hours,
            self.sleep_efficiency,
            self.sleep_fragmentation,
            self.sleep_onset_hour,
            self.wake_time_hour,
            self.total_steps,
            self.activity_variance,
            self.sedentary_hours,
            float(self.peak_activity_hour),
            self.avg_resting_hr,
            self.hrv_sdnn,
            self.hr_circadian_range,
            self.circadian_alignment_score,
        ]


class FeatureExtractionService:
    """
    Domain service for extracting clinical features.

    Orchestrates all aggregators to produce comprehensive features
    for mood disorder detection.
    """

    def __init__(self) -> None:
        """Initialize with required aggregators."""
        self.sleep_aggregator = SleepAggregator()
        self.activity_aggregator = ActivityAggregator()
        self.heart_aggregator = HeartRateAggregator()
        self.advanced_engineer = AdvancedFeatureEngineer()

    def extract_features(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
    ) -> dict[date, ClinicalFeatures]:
        """
        Extract clinical features from all data sources.

        Args:
            sleep_records: Raw sleep records
            activity_records: Raw activity records
            heart_records: Raw heart rate records

        Returns:
            Dictionary mapping dates to clinical features
        """
        # Aggregate each data type
        sleep_summaries = self.sleep_aggregator.aggregate_daily(sleep_records)
        activity_summaries = self.activity_aggregator.aggregate_daily(activity_records)
        heart_summaries = self.heart_aggregator.aggregate_daily(heart_records)

        # Get all unique dates
        all_dates: set[date] = set()
        all_dates.update(sleep_summaries.keys())
        all_dates.update(activity_summaries.keys())
        all_dates.update(heart_summaries.keys())

        if not all_dates:
            return {}

        # Extract features for each day
        features = {}
        for feature_date in sorted(all_dates):
            features[feature_date] = self._extract_day_features(
                feature_date,
                sleep_summaries.get(feature_date),
                activity_summaries.get(feature_date),
                heart_summaries.get(feature_date),
            )

        return features

    def _extract_day_features(
        self,
        feature_date: date,
        sleep_summary: Any,
        activity_summary: Any,
        heart_summary: Any,
    ) -> ClinicalFeatures:
        """Extract features for a single day."""
        clinical_notes = []
        is_significant = False

        # Extract sleep features
        sleep_duration = 0.0
        sleep_efficiency = 0.0
        sleep_fragmentation = 0.0
        sleep_onset = 0.0
        wake_time = 0.0

        if sleep_summary:
            sleep_duration = sleep_summary.total_sleep_hours
            sleep_efficiency = sleep_summary.sleep_efficiency
            sleep_fragmentation = sleep_summary.sleep_fragmentation_index

            if sleep_summary.earliest_bedtime:
                sleep_onset = (
                    sleep_summary.earliest_bedtime.hour
                    + sleep_summary.earliest_bedtime.minute / 60.0
                )
            if sleep_summary.latest_wake_time:
                wake_time = (
                    sleep_summary.latest_wake_time.hour
                    + sleep_summary.latest_wake_time.minute / 60.0
                )

            if sleep_summary.is_clinically_significant:
                is_significant = True
                if sleep_duration < 4:
                    clinical_notes.append("Severe sleep deprivation")
                elif sleep_duration > 10:
                    clinical_notes.append("Hypersomnia pattern")
                if sleep_efficiency < 0.7:
                    clinical_notes.append("Poor sleep efficiency")

        # Extract activity features
        total_steps = 0.0
        activity_variance = 0.0
        sedentary_hours = 0.0
        peak_activity_hour = 0

        if activity_summary:
            total_steps = activity_summary.total_steps
            activity_variance = activity_summary.activity_variance
            sedentary_hours = activity_summary.sedentary_hours
            peak_activity_hour = activity_summary.peak_activity_hour or 0

            if activity_summary.is_clinically_significant:
                is_significant = True
                if activity_summary.is_high_activity:
                    clinical_notes.append("Hyperactivity pattern")
                elif activity_summary.is_low_activity:
                    clinical_notes.append("Hypoactivity pattern")
                if activity_summary.is_erratic_pattern:
                    clinical_notes.append("Erratic activity pattern")

        # Extract heart features
        avg_resting_hr = 0.0
        hrv_sdnn = 0.0
        hr_circadian_range = 0.0

        if heart_summary:
            avg_resting_hr = heart_summary.avg_resting_hr
            hrv_sdnn = heart_summary.avg_hrv_sdnn
            hr_circadian_range = heart_summary.circadian_hr_range

            if heart_summary.is_clinically_significant:
                is_significant = True
                if heart_summary.has_high_resting_hr:
                    clinical_notes.append("Elevated resting heart rate")
                if heart_summary.has_low_hrv:
                    clinical_notes.append("Low HRV indicates poor recovery")
                if heart_summary.has_abnormal_circadian_rhythm:
                    clinical_notes.append("Abnormal heart rate circadian rhythm")

        # Calculate circadian alignment score
        circadian_score = self._calculate_circadian_alignment(
            sleep_onset,
            wake_time,
            peak_activity_hour,
            hr_circadian_range,
        )

        return ClinicalFeatures(
            date=feature_date,
            # Sleep
            sleep_duration_hours=sleep_duration,
            sleep_efficiency=sleep_efficiency,
            sleep_fragmentation=sleep_fragmentation,
            sleep_onset_hour=sleep_onset,
            wake_time_hour=wake_time,
            # Activity
            total_steps=total_steps,
            activity_variance=activity_variance,
            sedentary_hours=sedentary_hours,
            peak_activity_hour=peak_activity_hour,
            # Heart
            avg_resting_hr=avg_resting_hr,
            hrv_sdnn=hrv_sdnn,
            hr_circadian_range=hr_circadian_range,
            # Circadian
            circadian_alignment_score=circadian_score,
            # Clinical
            is_clinically_significant=is_significant,
            clinical_notes=clinical_notes,
        )

    def _calculate_circadian_alignment(
        self,
        sleep_onset: float,
        wake_time: float,
        peak_activity_hour: int,
        hr_circadian_range: float,
    ) -> float:
        """
        Calculate circadian rhythm alignment score.

        Good alignment:
        - Sleep onset: 22:00-00:00 (10 PM - midnight)
        - Wake time: 06:00-08:00 (6-8 AM)
        - Peak activity: 10:00-18:00 (10 AM - 6 PM)
        - HR circadian range: 10-30 bpm

        Returns score 0-1, where 1 is perfect alignment.
        """
        score = 0.0
        components = 0

        # Sleep onset alignment (ideal: 22-24)
        if 22 <= sleep_onset <= 24 or 0 <= sleep_onset <= 1:
            score += 1.0
            components += 1
        elif 20 <= sleep_onset <= 26 or sleep_onset <= 2:
            score += 0.5
            components += 1
        elif sleep_onset > 0:
            components += 1

        # Wake time alignment (ideal: 6-8)
        if 6 <= wake_time <= 8:
            score += 1.0
            components += 1
        elif 5 <= wake_time <= 9:
            score += 0.5
            components += 1
        elif wake_time > 0:
            components += 1

        # Peak activity alignment (ideal: 10-18)
        if 10 <= peak_activity_hour <= 18:
            score += 1.0
            components += 1
        elif 8 <= peak_activity_hour <= 20:
            score += 0.5
            components += 1
        elif peak_activity_hour > 0:
            components += 1

        # HR circadian range (ideal: 10-30 bpm)
        if 10 <= hr_circadian_range <= 30:
            score += 1.0
            components += 1
        elif 5 <= hr_circadian_range <= 40:
            score += 0.5
            components += 1
        elif hr_circadian_range > 0:
            components += 1

        return score / components if components > 0 else 0.0

    def extract_advanced_features(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
        lookback_days: int = 30,
    ) -> dict[date, AdvancedFeatures]:
        """
        Extract research-based advanced features for mood prediction.

        Implements the 36 features required by Seoul National study.

        Args:
            sleep_records: Raw sleep records
            activity_records: Raw activity records
            heart_records: Raw heart rate records
            lookback_days: Days of history for feature calculation

        Returns:
            Dictionary mapping dates to advanced features
        """
        # First aggregate all data
        sleep_summaries = self.sleep_aggregator.aggregate_daily(sleep_records)
        activity_summaries = self.activity_aggregator.aggregate_daily(activity_records)
        heart_summaries = self.heart_aggregator.aggregate_daily(heart_records)

        # Convert to lists for advanced feature engineering
        all_sleep = sorted(sleep_summaries.values(), key=lambda x: x.date)
        all_activity = sorted(activity_summaries.values(), key=lambda x: x.date)
        all_heart = sorted(heart_summaries.values(), key=lambda x: x.date)

        # Get unique dates where we have at least sleep data
        feature_dates = sorted(sleep_summaries.keys())

        # Extract advanced features for each date
        advanced_features = {}
        for feature_date in feature_dates:
            # Skip dates without enough history
            min_date = min(s.date for s in all_sleep) if all_sleep else feature_date
            if (feature_date - min_date).days < 7:  # Need at least 7 days history
                continue

            advanced_features[feature_date] = self.advanced_engineer.extract_advanced_features(
                current_date=feature_date,
                historical_sleep=all_sleep,
                historical_activity=all_activity,
                historical_heart=all_heart,
                lookback_days=lookback_days,
            )

        return advanced_features
