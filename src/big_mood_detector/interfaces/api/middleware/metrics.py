"""
Prometheus metrics for monitoring.

Tracks API performance and model predictions.
"""

import time
from collections.abc import Callable
from typing import Any, cast

from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Metrics definitions
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made", ["model_type", "prediction_type"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds", ["model_type"]
)

MODEL_LOAD_STATUS = Gauge(
    "model_loaded", "Model load status (1=loaded, 0=not loaded)", ["model_name"]
)

ENSEMBLE_REQUESTS = Counter(
    "ensemble_requests_total", "Total ensemble prediction requests", ["pat_available"]
)

RISK_PREDICTIONS = Histogram(
    "risk_prediction_values",
    "Risk prediction values distribution",
    ["risk_type"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to track HTTP metrics.
    """
    # Skip metrics endpoint itself
    if request.url.path == "/metrics":
        response = await call_next(request)
        return cast(Response, response)

    # Start timer
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Record metrics
    duration = time.time() - start_time

    # Get endpoint without parameters
    endpoint = request.url.path

    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status_code=response.status_code
    ).inc()

    REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(duration)

    return cast(Response, response)


def track_prediction(model_type: str, prediction_type: str, latency: float) -> None:
    """
    Track a prediction event.

    Args:
        model_type: "xgboost", "pat", or "ensemble"
        prediction_type: "depression", "hypomanic", or "manic"
        latency: Time taken in seconds
    """
    PREDICTION_COUNT.labels(
        model_type=model_type, prediction_type=prediction_type
    ).inc()

    PREDICTION_LATENCY.labels(model_type=model_type).observe(latency)


def track_ensemble_request(pat_available: bool) -> None:
    """Track ensemble prediction request."""
    ENSEMBLE_REQUESTS.labels(pat_available=str(pat_available).lower()).inc()


def track_risk_values(depression: float, hypomanic: float, manic: float) -> None:
    """Track risk prediction values for distribution analysis."""
    RISK_PREDICTIONS.labels(risk_type="depression").observe(depression)
    RISK_PREDICTIONS.labels(risk_type="hypomanic").observe(hypomanic)
    RISK_PREDICTIONS.labels(risk_type="manic").observe(manic)


def update_model_status(model_name: str, loaded: bool) -> None:
    """Update model load status."""
    MODEL_LOAD_STATUS.labels(model_name=model_name).set(1 if loaded else 0)


async def metrics_endpoint(request: Request) -> PlainTextResponse:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    metrics = generate_latest()
    return PlainTextResponse(
        content=metrics.decode("utf-8"),
        headers={"Content-Type": "text/plain; version=0.0.4"},
    )


def setup_metrics(app: Any) -> Any:
    """
    Set up metrics collection for the FastAPI app.
    """

    # Add middleware
    @app.middleware("http")  # type: ignore[misc]
    async def add_metrics_middleware(request: Request, call_next: Callable) -> Response:
        return await metrics_middleware(request, call_next)

    # Add metrics endpoint
    @app.get("/metrics", include_in_schema=False)  # type: ignore[misc]
    async def get_metrics(request: Request) -> PlainTextResponse:
        return await metrics_endpoint(request)

    return app
