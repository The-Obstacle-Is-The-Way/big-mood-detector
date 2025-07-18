"""
Minimal API for health checks and basic operations.

This is intentionally minimal - full API development comes after
core functionality is proven stable.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from big_mood_detector.interfaces.api.clinical_routes import router as clinical_router
from big_mood_detector.interfaces.api.routes.labels import router as labels_router
from big_mood_detector.interfaces.api.routes.predictions import (
    router as predictions_router,
)
from big_mood_detector.interfaces.api.routes.upload import router as upload_router

app = FastAPI(
    title="Big Mood Detector",
    description="Clinical-grade mood episode detection from wearable data",
    version="0.1.0",
)

# Include routers
app.include_router(clinical_router)
app.include_router(labels_router)
app.include_router(predictions_router)
app.include_router(upload_router)


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for container orchestration."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "big-mood-detector",
            "version": "0.1.0",
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
