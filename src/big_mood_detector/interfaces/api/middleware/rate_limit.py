"""
Rate limiting middleware for API endpoints.

Protects expensive endpoints like ensemble predictions from abuse.
"""

import os
from functools import wraps
from typing import Callable

from fastapi import HTTPException, Request
from fastapi.responses import Response

# Check if rate limiting is disabled
DISABLE_RATE_LIMIT = os.getenv("DISABLE_RATE_LIMIT", "0") == "1"

if not DISABLE_RATE_LIMIT:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address


# Create limiter with custom key function
def get_real_client_ip(request: Request) -> str:
    """
    Get the real client IP, considering proxies.
    
    Checks X-Forwarded-For and X-Real-IP headers.
    """
    # Check for proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct connection
    return get_remote_address(request)


# Create the limiter
if not DISABLE_RATE_LIMIT:
    limiter = Limiter(key_func=get_real_client_ip)
else:
    limiter = None


# Rate limit configurations for different endpoint types
RATE_LIMITS = {
    # Expensive ensemble predictions
    "ensemble_predict": "10/minute",
    
    # Regular predictions
    "predict": "30/minute",
    
    # File uploads
    "upload": "5/minute",
    
    # Status checks
    "status": "60/minute",
    
    # General API calls
    "default": "100/minute",
}


def rate_limit(limit_key: str = "default") -> Callable:
    """
    Decorator to apply rate limiting to an endpoint.
    
    Args:
        limit_key: Key to look up in RATE_LIMITS dict
        
    Usage:
        @rate_limit("ensemble_predict")
        async def predict_ensemble(...):
            ...
    """
    if DISABLE_RATE_LIMIT:
        # Return a no-op decorator
        def decorator(func: Callable) -> Callable:
            return func
        return decorator
    
    limit = RATE_LIMITS.get(limit_key, RATE_LIMITS["default"])
    
    def decorator(func: Callable) -> Callable:
        # Apply the limiter
        limited_func = limiter.limit(limit)(func)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await limited_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def setup_rate_limiting(app):
    """
    Set up rate limiting for the FastAPI app.
    
    Call this in main.py after creating the app.
    """
    if DISABLE_RATE_LIMIT:
        # No rate limiting in dev mode
        return app
    
    # Add exception handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add middleware to set rate limit headers
    @app.middleware("http")
    async def add_rate_limit_headers(request: Request, call_next):
        response = await call_next(request)
        
        # Add rate limit headers if they exist
        if hasattr(request.state, "view_rate_limit"):
            limit_info = request.state.view_rate_limit
            response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])
        
        return response
    
    return app


# Custom rate limit exceeded response
if not DISABLE_RATE_LIMIT:
    def custom_rate_limit_exceeded_handler(
        request: Request, exc: RateLimitExceeded
    ) -> Response:
        """
        Custom handler for rate limit exceeded errors.
        
        Returns more informative error messages.
        """
        response = {
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded: {exc.detail}",
            "retry_after": request.state.view_rate_limit.get("reset", 60),
        }
        
        return Response(
            content=response,
            status_code=429,
            headers={
                "Retry-After": str(request.state.view_rate_limit.get("reset", 60)),
                "X-RateLimit-Limit": str(request.state.view_rate_limit.get("limit", 0)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(request.state.view_rate_limit.get("reset", 0)),
            },
        )