"""
Temporal Ensemble Orchestrator

Coordinates PAT (current state) and XGBoost (future risk) predictions
with proper temporal separation. No averaging or mixing of time horizons.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from big_mood_detector.domain.services.pat_predictor import (
    PATPredictorInterface,
)
from big_mood_detector.domain.value_objects.temporal_mood_assessment import (
    CurrentMoodState,
    FutureMoodRisk,
    TemporalMoodAssessment,
)

if TYPE_CHECKING:
    from big_mood_detector.domain.services.pat_encoder import PATEncoderInterface

logger = logging.getLogger(__name__)


class TemporalEnsembleOrchestrator:
    """
    Orchestrates temporal mood assessment by separating current state (PAT)
    from future risk prediction (XGBoost).

    Key principle: NO averaging or mixing of temporal contexts.
    - PAT assesses NOW (past 7 days)
    - XGBoost predicts TOMORROW (next 24 hours)
    """

    def __init__(
        self,
        pat_encoder: PATEncoderInterface,
        pat_predictor: PATPredictorInterface,
        xgboost_predictor: Any,  # MoodPredictor instance
    ):
        """
        Initialize with required models.

        Args:
            pat_encoder: Encodes activity sequences to embeddings
            pat_predictor: Makes current state predictions from PAT embeddings
            xgboost_predictor: Makes future risk predictions from statistical features
        """
        self.pat_encoder = pat_encoder
        self.pat_predictor = pat_predictor
        self.xgboost_predictor = xgboost_predictor

    def predict(
        self,
        pat_sequence: np.ndarray,
        statistical_features: np.ndarray,
        user_id: str | None = None,
    ) -> TemporalMoodAssessment:
        """
        Generate temporal mood assessment with clear time separation.

        Args:
            pat_sequence: 7-day activity sequence for PAT (7x1440 array)
            statistical_features: Statistical features for XGBoost (36 Seoul features)
            user_id: Optional user identifier

        Returns:
            TemporalMoodAssessment with current state and future risk
        """
        # Step 1: Assess current state with PAT
        try:
            # Encode activity sequence to embeddings
            pat_embeddings = self.pat_encoder.encode(pat_sequence)

            # Get current state predictions
            pat_predictions = self.pat_predictor.predict_from_embeddings(pat_embeddings)

            current_state = CurrentMoodState(
                depression_probability=pat_predictions.depression_probability,
                on_benzodiazepine_probability=pat_predictions.benzodiazepine_probability,
                confidence=pat_predictions.confidence,
            )
            logger.debug(
                f"PAT current state: depression={pat_predictions.depression_probability:.3f}, "
                f"benzo={pat_predictions.benzodiazepine_probability:.3f}"
            )
        except Exception as e:
            logger.warning(f"PAT assessment failed: {e}. Using neutral state.")
            current_state = CurrentMoodState(
                depression_probability=0.5,
                on_benzodiazepine_probability=0.5,
                confidence=0.0,
            )

        # Step 2: Predict future risk with XGBoost
        try:
            xgb_predictions = self.xgboost_predictor.predict(statistical_features)

            future_risk = FutureMoodRisk(
                depression_risk=xgb_predictions.depression_risk,
                hypomanic_risk=xgb_predictions.hypomanic_risk,
                manic_risk=xgb_predictions.manic_risk,
                confidence=xgb_predictions.confidence,
            )
            logger.debug(
                f"XGBoost future risk: depression={xgb_predictions.depression_risk:.3f}, "
                f"hypomania={xgb_predictions.hypomanic_risk:.3f}, "
                f"mania={xgb_predictions.manic_risk:.3f}"
            )
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}. Using neutral risk.")
            future_risk = FutureMoodRisk(
                depression_risk=0.33,
                hypomanic_risk=0.33,
                manic_risk=0.34,
                confidence=0.0,
            )

        # Step 3: Create temporal assessment (NO AVERAGING!)
        assessment = TemporalMoodAssessment(
            current_state=current_state,
            future_risk=future_risk,
            assessment_timestamp=datetime.now(),
            user_id=user_id,
        )

        # Log if there's a temporal mismatch (current depression but low future risk)
        if current_state.depression_probability > 0.5 and future_risk.depression_risk < 0.3:
            logger.info(
                f"Temporal mismatch detected for user {user_id}: "
                f"Currently depressed but low future risk"
            )

        return assessment

    def predict_with_alerts(
        self,
        pat_sequence: np.ndarray,
        statistical_features: np.ndarray,
        user_id: str | None = None,
        alert_threshold: float = 0.7,
    ) -> tuple[TemporalMoodAssessment, list[str]]:
        """
        Generate assessment with clinical alerts.

        Args:
            pat_sequence: 7-day activity sequence for PAT
            statistical_features: Statistical features for XGBoost
            user_id: Optional user identifier
            alert_threshold: Threshold for generating alerts (default 0.7)

        Returns:
            Tuple of (assessment, list of alert messages)
        """
        assessment = self.predict(pat_sequence, statistical_features, user_id)
        alerts = []

        # Check current state alerts
        if assessment.current_state.depression_probability > alert_threshold:
            alerts.append(
                f"HIGH CURRENT DEPRESSION: {assessment.current_state.depression_probability:.1%} "
                f"probability based on past 7 days"
            )

        # Check future risk alerts
        if assessment.future_risk.manic_risk > alert_threshold:
            alerts.append(
                f"HIGH MANIA RISK: {assessment.future_risk.manic_risk:.1%} "
                f"probability in next 24 hours"
            )

        if assessment.future_risk.hypomanic_risk > alert_threshold:
            alerts.append(
                f"HIGH HYPOMANIA RISK: {assessment.future_risk.hypomanic_risk:.1%} "
                f"probability in next 24 hours"
            )

        # Check for rapid mood shifts
        if (assessment.current_state.depression_probability > 0.5 and
            (assessment.future_risk.manic_risk +
             assessment.future_risk.hypomanic_risk) > 0.6):
            alerts.append(
                "MOOD SHIFT WARNING: Currently depressed but high risk of "
                "hypomanic/manic episode in next 24 hours"
            )

        return assessment, alerts
