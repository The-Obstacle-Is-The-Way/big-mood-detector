"""
Minimal API for health checks and basic operations.

This is intentionally minimal - full API development comes after 
core functionality is proven stable.
"""

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Big Mood Detector",
    description="Clinical-grade mood episode detection from wearable data",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "big-mood-detector",
            "version": "0.1.0"
        }
    )


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Big Mood Detector API",
        "endpoints": {
            "/health": "Health check",
            "/docs": "API documentation (this page)",
        }
    }


# Note: Full API endpoints (predictions, data processing, etc.) 
# will be added after core functionality is thoroughly tested
# and the vertical slice is proven to work end-to-end.