"""
Ensemble Model Orchestrator

Coordinates multiple ML models (PAT + XGBoost) for enhanced mood predictions.
Implements parallel processing, confidence weighting, and fallback strategies.

DEPRECATED: This module implements a flawed ensemble that doesn't actually
combine predictions. Use TemporalEnsembleOrchestrator for proper temporal
separation of current state (PAT) and future risk (XGBoost).
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from big_mood_detector.infrastructure.ml_models.xgboost_models import (
        XGBoostMoodPredictor,
    )

import numpy as np
from numpy.typing import NDArray

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.services.mood_predictor import MoodPrediction
from big_mood_detector.domain.services.pat_model_interface import PATModelInterface
from big_mood_detector.domain.services.pat_sequence_builder import PATSequenceBuilder
# Import just the constant, not the models to avoid module-level loading
PAT_AVAILABLE = True  # Always available with PyTorch
from big_mood_detector.infrastructure.settings.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model orchestration."""

    # Model weights for ensemble averaging
    xgboost_weight: float = 0.6
    pat_weight: float = 0.4

    # Timeout settings (seconds)
    pat_timeout: float = 10.0
    xgboost_timeout: float = 5.0

    # Feature engineering
    use_pat_features: bool = True
    pat_feature_dim: int = 16  # How many PAT features to use

    # Confidence thresholds
    min_confidence_threshold: float = 0.7
    fallback_to_single_model: bool = True

    @classmethod
    def from_settings(cls) -> EnsembleConfig:
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            xgboost_weight=settings.ENSEMBLE_XGBOOST_WEIGHT,
            pat_weight=settings.ENSEMBLE_PAT_WEIGHT,
            pat_timeout=settings.ENSEMBLE_PAT_TIMEOUT,
            xgboost_timeout=settings.ENSEMBLE_XGBOOST_TIMEOUT,
        )


@dataclass
class EnsemblePrediction:
    """Enhanced prediction with model contributions."""

    # Individual model predictions
    xgboost_prediction: MoodPrediction | None
    pat_enhanced_prediction: MoodPrediction | None  # Deprecated - kept for compatibility

    # Ensemble results
    ensemble_prediction: MoodPrediction

    # Metadata
    models_used: list[str]
    confidence_scores: dict[str, float]
    processing_time_ms: dict[str, float]

    # NEW: Separate PAT outputs (with defaults)
    pat_embeddings: NDArray[np.float32] | None = None  # 96-dim embeddings from PAT encoder
    pat_prediction: MoodPrediction | None = None  # Future: PAT classification result

    # NEW: Temporal context for each model
    temporal_context: dict[str, str] | None = None


class EnsembleOrchestrator:
    """
    Orchestrates multiple ML models for robust mood prediction.

    Features:
    - Parallel model execution
    - Graceful degradation on model failures
    - Confidence-based weighting
    - Performance monitoring
    """

    def __init__(
        self,
        xgboost_predictor: "XGBoostMoodPredictor",
        pat_model: PATModelInterface | None = None,
        config: EnsembleConfig | None = None,
        personal_calibrator: Any | None = None,
    ):
        """
        Initialize the orchestrator with models.

        Args:
            xgboost_predictor: Primary XGBoost predictor
            pat_model: Optional PAT model for enhanced features
            config: Ensemble configuration
            personal_calibrator: Optional personal calibrator for user-specific adjustments
        """
        warnings.warn(
            "EnsembleOrchestrator is deprecated and doesn't actually ensemble predictions. "
            "Use TemporalEnsembleOrchestrator for proper temporal separation of "
            "current state (PAT) and future risk (XGBoost).",
            DeprecationWarning,
            stacklevel=2
        )
        self.xgboost_predictor = xgboost_predictor
        self.pat_model = pat_model
        self.pat_builder = PATSequenceBuilder() if pat_model else None
        self.config = config or EnsembleConfig.from_settings()
        self.personal_calibrator = personal_calibrator

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=3)

        logger.info(
            f"Initialized ensemble orchestrator with: "
            f"XGBoost={xgboost_predictor.is_loaded}, "
            f"PAT={pat_model is not None and pat_model.is_loaded}"
        )

    def predict(
        self,
        statistical_features: NDArray[np.float32],
        activity_records: list[ActivityRecord] | None = None,
        prediction_date: np.datetime64 | None = None,
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction using all available models.

        Args:
            statistical_features: 36-feature vector from standard pipeline
            activity_records: Optional activity data for PAT features
            prediction_date: Date for prediction (for PAT sequence)

        Returns:
            EnsemblePrediction with detailed results
        """
        import time

        start_time = time.time()

        # Track timing
        timing: dict[str, float] = {}
        models_used: list[str] = []
        predictions: dict[str, MoodPrediction | None] = {}
        pat_embeddings: NDArray[np.float32] | None = None

        # Submit parallel tasks
        futures: dict[str, Any] = {}

        # 1. Standard XGBoost prediction (only on statistical features)
        futures["xgboost"] = self.executor.submit(
            self._predict_xgboost, statistical_features
        )

        # 2. Extract PAT embeddings (if available)
        if self.config.use_pat_features and self.pat_model and activity_records:
            futures["pat_embeddings"] = self.executor.submit(
                self._extract_pat_embeddings,
                activity_records,
                prediction_date,
            )

        # Collect results with timeouts
        for name, future in futures.items():
            try:
                timeout = (
                    self.config.xgboost_timeout
                    if name == "xgboost"
                    else self.config.pat_timeout
                )

                result_start = time.time()
                if name == "pat_embeddings":
                    pat_embeddings = future.result(timeout=timeout)
                    models_used.append("pat_embeddings")
                else:
                    predictions[name] = future.result(timeout=timeout)
                    models_used.append(name)
                timing[name] = (time.time() - result_start) * 1000  # ms

            except TimeoutError:
                logger.warning(f"{name} timed out after {timeout}s")
                if name == "xgboost":
                    predictions[name] = None

            except Exception as e:
                logger.error(f"{name} failed: {e}")
                if name == "xgboost":
                    predictions[name] = None

        # For now, ensemble is just XGBoost (PAT can't predict yet)
        xgboost_pred = predictions.get("xgboost")
        ensemble_pred = xgboost_pred if xgboost_pred else MoodPrediction(
            depression_risk=0.5, hypomanic_risk=0.5, manic_risk=0.5, confidence=0.0
        )

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(predictions)

        # Total processing time
        timing["total"] = (time.time() - start_time) * 1000

        # Determine temporal context
        temporal_context = {
            "xgboost": "next_24_hours",
            "pat": "embeddings_only" if pat_embeddings is not None else "not_available"
        }

        return EnsemblePrediction(
            xgboost_prediction=xgboost_pred,
            pat_enhanced_prediction=None,  # Deprecated
            pat_embeddings=pat_embeddings,
            pat_prediction=None,  # Not available until we train classification heads
            ensemble_prediction=ensemble_pred,
            models_used=models_used,
            confidence_scores=confidence_scores,
            processing_time_ms=timing,
            temporal_context=temporal_context,
        )

    def _predict_xgboost(self, features: NDArray[np.float32]) -> MoodPrediction:
        """Run standard XGBoost prediction."""
        return self.xgboost_predictor.predict(features.astype(np.float64))

    def _extract_pat_embeddings(
        self,
        activity_records: list[ActivityRecord],
        prediction_date: np.datetime64 | None,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Extract PAT embeddings without making predictions."""
        if self.pat_builder is None or self.pat_model is None:
            raise RuntimeError("PAT model or builder not available")

        # Build PAT sequence
        if prediction_date:
            # Convert numpy datetime64 to date
            from datetime import UTC, datetime

            date_obj = datetime.fromtimestamp(
                prediction_date.astype("datetime64[s]").astype(int), UTC
            ).date()
            sequence = self.pat_builder.build_sequence(
                activity_records, end_date=date_obj
            )
        else:
            # Use last available date
            dates = [r.start_date.date() for r in activity_records]
            if dates:
                sequence = self.pat_builder.build_sequence(
                    activity_records, end_date=max(dates)
                )
            else:
                raise ValueError("No activity records provided")

        # Extract and return PAT embeddings (96-dim)
        return self.pat_model.extract_features(sequence).astype(np.float64)

    def _predict_with_pat(
        self,
        statistical_features: NDArray[np.float32],
        activity_records: list[ActivityRecord],
        prediction_date: np.datetime64 | None,
    ) -> MoodPrediction:
        """Run prediction with PAT-enhanced features."""
        if self.pat_builder is None or self.pat_model is None:
            raise RuntimeError("PAT model or builder not available")

        # Build PAT sequence
        if prediction_date:
            # Convert numpy datetime64 to date
            from datetime import UTC, datetime

            date_obj = datetime.fromtimestamp(
                prediction_date.astype("datetime64[s]").astype(int), UTC
            ).date()
            sequence = self.pat_builder.build_sequence(
                activity_records, end_date=date_obj
            )
        else:
            # Use last available date
            dates = [r.start_date.date() for r in activity_records]
            if dates:
                sequence = self.pat_builder.build_sequence(
                    activity_records, end_date=max(dates)
                )
            else:
                raise ValueError("No activity records provided")

        # Extract PAT features
        pat_features = self.pat_model.extract_features(sequence)

        # Combine with statistical features
        # Take first 20 statistical + 16 PAT features
        enhanced_features = np.concatenate(
            [statistical_features[:20], pat_features[: self.config.pat_feature_dim]]
        )

        # Pad to 36 features if needed
        if len(enhanced_features) < 36:
            enhanced_features = np.pad(
                enhanced_features, (0, 36 - len(enhanced_features)), mode="constant"
            )

        return self.xgboost_predictor.predict(enhanced_features)

    def _calculate_ensemble(
        self, xgboost_pred: MoodPrediction | None, pat_pred: MoodPrediction | None
    ) -> MoodPrediction:
        """Calculate weighted ensemble prediction."""
        # Handle missing predictions
        if xgboost_pred is None and pat_pred is None:
            # Return neutral prediction if both failed
            return MoodPrediction(
                depression_risk=0.5, hypomanic_risk=0.5, manic_risk=0.5, confidence=0.0
            )

        if xgboost_pred is None:
            if pat_pred is None:
                # This should never happen due to the check above
                raise RuntimeError("Both predictions are None")
            return pat_pred

        if pat_pred is None:
            return xgboost_pred

        # Weighted average
        w1, w2 = self.config.xgboost_weight, self.config.pat_weight

        depression = w1 * xgboost_pred.depression_risk + w2 * pat_pred.depression_risk
        hypomanic = w1 * xgboost_pred.hypomanic_risk + w2 * pat_pred.hypomanic_risk
        manic = w1 * xgboost_pred.manic_risk + w2 * pat_pred.manic_risk

        # Average confidence
        confidence = (xgboost_pred.confidence + pat_pred.confidence) / 2

        # Risks are already in the MoodPrediction object

        return MoodPrediction(
            depression_risk=depression,
            hypomanic_risk=hypomanic,
            manic_risk=manic,
            confidence=confidence,
        )

    def _calculate_confidence(
        self, predictions: dict[str, MoodPrediction | None]
    ) -> dict[str, float]:
        """Calculate confidence scores for each model."""
        scores = {}

        for name, pred in predictions.items():
            if pred is not None:
                scores[name] = pred.confidence
            else:
                scores[name] = 0.0

        # Overall ensemble confidence
        valid_scores = [s for s in scores.values() if s > 0]
        scores["ensemble"] = float(np.mean(valid_scores)) if valid_scores else 0.0

        return scores

    def shutdown(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
