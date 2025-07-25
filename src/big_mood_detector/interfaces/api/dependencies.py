"""
API Dependencies for FastAPI.

Provides singleton instances and dependency injection for performance.
"""

from functools import lru_cache
from typing import cast

from fastapi import Request

from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    EnsembleConfig,
    EnsembleOrchestrator,
)
from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.services.mood_predictor import MoodPredictor
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

    # Use converted directory where JSON models actually exist
    from big_mood_detector.core.paths import MODEL_WEIGHTS_DIR

    xgboost_converted_dir = MODEL_WEIGHTS_DIR / "xgboost" / "converted"

    if not xgboost_predictor.load_models(xgboost_converted_dir):
        logger.error("Failed to load XGBoost models")
        return None

    # Initialize PAT model through DI
    pat_predictor = None
    try:
        from big_mood_detector.domain.services.pat_predictor import (
            PATPredictorInterface,
        )
        from big_mood_detector.infrastructure.di import get_container

        container = get_container()
        pat_predictor = container.resolve(PATPredictorInterface)  # type: ignore[type-abstract]
        logger.info("PAT production model (0.5929 AUC) loaded successfully for API")
    except Exception as e:
        logger.warning(f"Could not initialize PAT production model: {e}")
        pat_predictor = None

    # Create ensemble orchestrator
    config = EnsembleConfig.from_settings()
    orchestrator = EnsembleOrchestrator(
        xgboost_predictor=xgboost_predictor,
        pat_model=pat_predictor,  # type: ignore[arg-type]  # Deprecated, using new interface
        config=config,
    )

    return orchestrator


def get_mood_predictor_with_state(request: Request) -> MoodPredictor:
    """
    Get MoodPredictor from app state if available, otherwise create new.

    This is better for multi-worker deployments.
    """
    if hasattr(request.app.state, "predictor") and request.app.state.predictor:
        return cast(MoodPredictor, request.app.state.predictor)
    return get_mood_predictor()


def get_ensemble_orchestrator_with_state(
    request: Request,
) -> EnsembleOrchestrator | None:
    """
    Get EnsembleOrchestrator from app state if available, otherwise create new.

    This is better for multi-worker deployments.
    """
    if hasattr(request.app.state, "orchestrator"):
        return cast(EnsembleOrchestrator | None, request.app.state.orchestrator)
    return get_ensemble_orchestrator()
