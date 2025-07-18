"""
API Dependencies for FastAPI.

Provides singleton instances and dependency injection for performance.
"""

from functools import lru_cache
from pathlib import Path

from fastapi import Request

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    EnsembleConfig,
    EnsembleOrchestrator,
)
from big_mood_detector.core.paths import XGBOOST_PRETRAINED_DIR
from big_mood_detector.domain.services.mood_predictor import MoodPredictor
from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE
from big_mood_detector.infrastructure.ml_models.xgboost_models import (
    XGBoostMoodPredictor,
)


@lru_cache(maxsize=1)
def get_mood_predictor() -> MoodPredictor:
    """
    Get singleton MoodPredictor instance.

    This ensures models are loaded only once at startup,
    not on every request.

    Returns:
        Cached MoodPredictor instance
    """
    return MoodPredictor()


@lru_cache(maxsize=1)
def get_mood_pipeline() -> MoodPredictionPipeline:
    """
    Get singleton MoodPredictionPipeline instance.

    This ensures the pipeline is created only once at startup,
    not on every request.

    Returns:
        Cached MoodPredictionPipeline instance
    """
    return MoodPredictionPipeline()


@lru_cache(maxsize=1)
def get_ensemble_orchestrator() -> EnsembleOrchestrator | None:
    """
    Get singleton EnsembleOrchestrator instance.

    This loads both XGBoost and PAT models for ensemble predictions.
    Returns None if models cannot be loaded.

    Returns:
        Cached EnsembleOrchestrator instance or None
    """
    import logging

    logger = logging.getLogger(__name__)

    # Initialize XGBoost predictor
    xgboost_predictor = XGBoostMoodPredictor()
    
    if not xgboost_predictor.load_models(XGBOOST_PRETRAINED_DIR):
        logger.error("Failed to load XGBoost models")
        return None

    # Initialize PAT model if available
    pat_model = None
    if PAT_AVAILABLE:
        try:
            from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
            
            pat_model = PATModel(model_size="medium")
            if not pat_model.load_pretrained_weights():
                logger.warning("Failed to load PAT model weights")
                pat_model = None
            else:
                logger.info("PAT model loaded successfully for API")
        except Exception as e:
            logger.warning(f"Could not initialize PAT model: {e}")
            pat_model = None
    else:
        logger.warning("PAT not available - TensorFlow not installed")

    # Create ensemble orchestrator
    config = EnsembleConfig()
    orchestrator = EnsembleOrchestrator(
        xgboost_predictor=xgboost_predictor,
        pat_model=pat_model,
        config=config,
    )

    return orchestrator


def get_mood_predictor_with_state(request: Request) -> MoodPredictor:
    """
    Get MoodPredictor from app state if available, otherwise create new.
    
    This is better for multi-worker deployments.
    """
    if hasattr(request.app.state, "predictor") and request.app.state.predictor:
        return request.app.state.predictor
    return get_mood_predictor()


def get_ensemble_orchestrator_with_state(
    request: Request,
) -> EnsembleOrchestrator | None:
    """
    Get EnsembleOrchestrator from app state if available, otherwise create new.
    
    This is better for multi-worker deployments.
    """
    if hasattr(request.app.state, "orchestrator"):
        return request.app.state.orchestrator
    return get_ensemble_orchestrator()
