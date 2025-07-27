"""
XGBoost Pipeline for mood prediction.

Independent pipeline for predicting tomorrow's mood episode risk
using 30-60 days of health data.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

from big_mood_detector.application.validators.pipeline_validators import (
    ValidationResult,
    XGBoostValidator,
)
from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.activity_aggregator import ActivityAggregator
from big_mood_detector.domain.services.heart_rate_aggregator import HeartRateAggregator
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator

logger = logging.getLogger(__name__)


@dataclass
class XGBoostResult:
    """Result from XGBoost mood prediction."""

    depression_probability: float  # 0.0 to 1.0
    mania_probability: float  # 0.0 to 1.0
    hypomania_probability: float  # 0.0 to 1.0
    prediction_window: str  # e.g., "next 24 hours"
    data_days_used: int  # Number of days with data
    clinical_interpretation: str
    highest_risk_episode: str  # "depression", "mania", "hypomania", or "stable"
    confidence_level: str  # "high", "medium", "low" based on data quality


class XGBoostPipeline:
    """
    Independent pipeline for XGBoost mood prediction.

    This pipeline:
    1. Validates that 30+ days of data are available (sparse OK)
    2. Extracts Seoul features from available data
    3. Runs XGBoost models for tomorrow's mood prediction
    """

    def __init__(
        self,
        feature_extractor: Any,  # Clinical feature extractor (avoiding circular imports)
        predictor: Any,  # XGBoost predictor (avoiding circular imports)
        validator: XGBoostValidator,
    ):
        """
        Initialize XGBoost pipeline.

        Args:
            feature_extractor: Clinical feature extractor instance
            predictor: XGBoost predictor instance
            validator: XGBoost data validator
        """
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.validator = validator
        
        # Aggregators for daily summaries
        self.sleep_aggregator = SleepAggregator()
        self.activity_aggregator = ActivityAggregator()
        self.heart_rate_aggregator = HeartRateAggregator()

    def can_run(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
        start_date: date,
        end_date: date,
    ) -> ValidationResult:
        """
        Check if XGBoost can run with available data.

        Args:
            sleep_records: Available sleep records
            activity_records: Available activity records
            heart_records: Available heart rate records
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            ValidationResult with details about data sufficiency
        """
        return self.validator.validate(
            sleep_records=sleep_records,
            activity_records=activity_records,
            start_date=start_date,
            end_date=end_date,
        )

    def process(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        heart_records: list[HeartRateRecord],
        target_date: date,
    ) -> Optional[XGBoostResult]:
        """
        Process health data through XGBoost pipeline.

        Args:
            sleep_records: All available sleep records
            activity_records: All available activity records
            heart_records: All available heart rate records
            target_date: Date to predict for (predicts next day)

        Returns:
            XGBoostResult if sufficient data, None otherwise
        """
        # Determine data range (use up to 60 days if available)
        all_dates = set()
        
        for sleep_record in sleep_records:
            all_dates.add(sleep_record.start_date.date())
        for activity_record in activity_records:
            all_dates.add(activity_record.start_date.date())
        for heart_record in heart_records:
            all_dates.add(heart_record.timestamp.date())
        
        if len(all_dates) < 30:
            logger.info(f"Insufficient data for XGBoost: only {len(all_dates)} days")
            return None

        # Use most recent 60 days (or all available if less)
        sorted_dates = sorted(all_dates)
        if len(sorted_dates) > 60:
            # Use most recent 60 days
            start_date = target_date - timedelta(days=59)
            data_dates = [d for d in sorted_dates if d >= start_date]
        else:
            data_dates = sorted_dates
        
        actual_start = min(data_dates)
        actual_end = max(data_dates)
        
        logger.info(
            f"XGBoost using {len(data_dates)} days from {actual_start} to {actual_end}"
        )

        # Aggregate daily summaries
        sleep_summaries = self.sleep_aggregator.aggregate_daily(sleep_records)
        activity_summaries = self.activity_aggregator.aggregate_daily(activity_records)
        heart_summaries = self.heart_rate_aggregator.aggregate_daily(heart_records)

        # Extract clinical features (Seoul features)
        try:
            clinical_features = self.feature_extractor.extract_clinical_features(
                sleep_summaries=sleep_summaries,
                activity_summaries=activity_summaries,
                heart_rate_summaries=heart_summaries,
                start_date=actual_start,
                end_date=actual_end,
            )

            if not clinical_features or not clinical_features.seoul_features:
                logger.error("Failed to extract Seoul features")
                return None

            # Convert to XGBoost input (36 features)
            feature_vector = clinical_features.seoul_features.to_xgboost_input()
            
            # Run prediction
            predictions = self.predictor.predict_mood_episodes(
                features=feature_vector,
                user_id="default",  # Can be customized later
            )

            # Determine highest risk
            risks = {
                "depression": predictions["depression"]["probability"],
                "mania": predictions["mania"]["probability"],
                "hypomania": predictions["hypomania"]["probability"],
            }
            
            highest_risk = max(risks, key=lambda k: risks[k])
            highest_prob = risks[highest_risk]
            
            # Clinical interpretation
            if highest_prob < 0.3:
                interpretation = "Low risk for mood episodes in next 24 hours"
                risk_episode = "stable"
            elif highest_prob < 0.5:
                interpretation = f"Moderate risk for {highest_risk} in next 24 hours - monitor symptoms"
                risk_episode = highest_risk
            elif highest_prob < 0.7:
                interpretation = f"Elevated risk for {highest_risk} in next 24 hours - consider preventive measures"
                risk_episode = highest_risk
            else:
                interpretation = f"High risk for {highest_risk} in next 24 hours - clinical intervention recommended"
                risk_episode = highest_risk

            # Confidence based on data completeness
            data_coverage = len(data_dates) / (actual_end - actual_start).days
            if data_coverage > 0.8:
                confidence = "high"
            elif data_coverage > 0.5:
                confidence = "medium"
            else:
                confidence = "low"

            return XGBoostResult(
                depression_probability=predictions["depression"]["probability"],
                mania_probability=predictions["mania"]["probability"],
                hypomania_probability=predictions["hypomania"]["probability"],
                prediction_window="next 24 hours",
                data_days_used=len(data_dates),
                clinical_interpretation=interpretation,
                highest_risk_episode=risk_episode,
                confidence_level=confidence,
            )

        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return None