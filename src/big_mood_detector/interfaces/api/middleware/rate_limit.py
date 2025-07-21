"""
Rate limiting middleware for API endpoints.

Protects expensive endpoints like ensemble predictions from abuse.
"""

import os
from collections.abc import Callable
from functools import wraps
from typing import Any

from fastapi import Request
from fastapi.responses import Response

# Check if rate limiting is disabled
DISABLE_RATE_LIMIT = os.getenv("DISABLE_RATE_LIMIT", "0") == "1"

if not DISABLE_RATE_LIMIT:
    # Production mode - use real rate limiting
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    # Create limiter with custom key function
    def get_real_client_ip(request: Request) -> str:
        """
        Get the real client IP address, handling proxies.

        Checks common proxy headers in order of preference.
        """
        # Check for common proxy headers
        headers_to_check = [
            "X-Real-IP",
            "X-Forwarded-For",
            "CF-Connecting-IP",  # Cloudflare
            "True-Client-IP",    # Akamai
        ]

        for header in headers_to_check:
            if header in request.headers:
                # X-Forwarded-For can contain multiple IPs
                ip = request.headers[header]
                if "," in ip:
                    ip = ip.split(",")[0].strip()
                return ip

        # Fall back to direct connection
        return get_remote_address(request)

    # Create the limiter
    limiter = Limiter(key_func=get_real_client_ip)

    # Rate limit configurations for different endpoint types
    RATE_LIMITS = {
        # Expensive ensemble predictions
        "ensemble_predict": "10/minute",
        # Regular predictions
        "predict": "30/minute",
        # File uploads
        "upload": "5/minute",
        # Feature extraction
        "features": "20/minute",
        # General API calls
        "default": "60/minute",
    }

    def rate_limit(limit_key: str = "default") -> Callable[..., Any]:
        """
        Apply rate limiting to an endpoint.

        Args:
            limit_key: Key to look up in RATE_LIMITS dict

        Returns:
            Decorator function
        """
        limit = RATE_LIMITS.get(limit_key, RATE_LIMITS["default"])

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Apply the limiter decorator
            limited = limiter.limit(limit)(func)

            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await limited(*args, **kwargs)

            return wrapper

        return decorator

    def setup_rate_limiting(app: Any) -> None:
        """
        Set up rate limiting middleware on the FastAPI app.

        Args:
            app: FastAPI application instance
        """
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

else:
    # Test/development mode - no rate limiting

    # Mock implementations
    class RateLimitExceeded(Exception):
        """Mock exception for tests."""
        pass

    limiter = None

    def get_real_client_ip(request: Request) -> str:
        """Mock implementation that returns a fixed IP."""
        return "127.0.0.1"

    def rate_limit(limit_key: str = "default") -> Callable[..., Any]:
        """
        No-op decorator when rate limiting is disabled.

        Args:
            limit_key: Ignored in test mode

        Returns:
            Pass-through decorator
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorator

    def setup_rate_limiting(app: Any) -> None:
        """
        No-op setup when rate limiting is disabled.

        Args:
            app: FastAPI application instance (unused)
        """
        pass

    # Mock for exception handler
    def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
        """Mock handler that should never be called in test mode."""
        return Response(content="Rate limit exceeded", status_code=429)
