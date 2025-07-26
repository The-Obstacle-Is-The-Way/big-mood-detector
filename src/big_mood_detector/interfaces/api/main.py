"""
Minimal API for health checks and basic operations.

This is intentionally minimal - full API development comes after
core functionality is proven stable.
"""

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from big_mood_detector.infrastructure.settings.config import get_settings
from big_mood_detector.interfaces.api.clinical_routes import router as clinical_router
from big_mood_detector.interfaces.api.middleware.metrics import (
    setup_metrics,
    update_model_status,
)
from big_mood_detector.interfaces.api.middleware.rate_limit import setup_rate_limiting
from big_mood_detector.interfaces.api.routes.depression import (
    router as depression_router,
)
from big_mood_detector.interfaces.api.routes.features import router as features_router
from big_mood_detector.interfaces.api.routes.labels import router as labels_router
from big_mood_detector.interfaces.api.routes.predictions import (
    router as predictions_router,
)
from big_mood_detector.interfaces.api.routes.upload import router as upload_router

app = FastAPI(
    title="Big Mood Detector",
    description="Clinical-grade mood episode detection from wearable data",
    version="0.4.0",
)

# Set up middleware
setup_rate_limiting(app)
app = setup_metrics(app)


# Ensure directories exist on startup
@app.on_event("startup")
async def startup_event() -> None:
    """Ensure required directories exist and preload models when the API starts."""
    import sys

    from big_mood_detector.core.security import validate_secrets
    from big_mood_detector.infrastructure.logging import get_module_logger
    from big_mood_detector.infrastructure.settings.utils import validate_model_paths
    from big_mood_detector.interfaces.api.dependencies import (
        get_ensemble_orchestrator,
        get_mood_predictor,
    )

    logger = get_module_logger(__name__)

    # Validate security settings first
    validate_secrets()

    settings = get_settings()
    settings.ensure_directories()

    # Validate model weights exist
    validation_error = validate_model_paths(settings)
    if validation_error:
        logger.critical("Model weights validation failed", error=validation_error)
        sys.exit(1)

    # Preload models into memory (singleton cache)
    logger.info("Preloading models...")

    # Preload basic predictor
    predictor = get_mood_predictor()
    logger.info(f"MoodPredictor loaded with {len(predictor.models)} models")

    # Update metrics
    for model_name in predictor.models:
        update_model_status(f"xgboost_{model_name}", True)

    # Preload ensemble orchestrator (includes PAT if available)
    orchestrator = get_ensemble_orchestrator()
    if orchestrator:
        logger.info("Ensemble orchestrator loaded successfully")
        update_model_status("ensemble", True)

        if orchestrator.pat_model:
            logger.info("PAT model loaded and available")
            update_model_status("pat", True)
        else:
            logger.warning("PAT model not available - TensorFlow may not be installed")
            update_model_status("pat", False)
    else:
        logger.warning("Failed to load ensemble orchestrator")
        update_model_status("ensemble", False)

    # Store in app state for multi-worker scenarios
    app.state.predictor = predictor
    app.state.orchestrator = orchestrator

    logger.info("API startup complete", model_path=str(settings.MODEL_WEIGHTS_PATH))


# Include routers
app.include_router(clinical_router)
app.include_router(depression_router)
app.include_router(features_router)
app.include_router(labels_router)
app.include_router(predictions_router)

# Only include upload routes if worker is enabled
if os.environ.get("ENABLE_ASYNC_UPLOAD", "false").lower() == "true":
    app.include_router(upload_router)


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for container orchestration."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "big-mood-detector",
            "version": "0.4.0",
        },
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Kubernetes-style health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with basic info."""
    return {
        "message": "Big Mood Detector API",
        "endpoints": {
            "/health": "Health check",
            "/docs": "API documentation",
            "/api/v1/clinical": "Clinical interpretation endpoints",
        },
    }


# Note: Full API endpoints (predictions, data processing, etc.)
# will be added after core functionality is thoroughly tested
# and the vertical slice is proven to work end-to-end.
