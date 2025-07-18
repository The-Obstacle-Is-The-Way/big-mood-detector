"""
Security configuration and validation.

Ensures production deployments don't use default secrets.
"""

import os
import secrets
import sys
from typing import NoReturn

DEFAULT_SECRETS = {
    "CHANGE-ME-USE-STRONG-RANDOM-KEY-IN-PRODUCTION",
    "CHANGE-ME-RANDOM-SALT-STRING",
    "CHANGE-ME-IN-PRODUCTION-USE-SECRETS-MANAGER",
    "CHANGE-ME-RANDOM-STRING",
}


def validate_secrets() -> None:
    """
    Validate that production doesn't use default secrets.
    
    Exits the application if unsafe secrets are detected.
    """
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Only enforce in production
    if environment != "production":
        return
    
    # Check critical secrets
    secret_key = os.getenv("SECRET_KEY", "")
    api_salt = os.getenv("API_KEY_SALT", "")
    
    unsafe_secrets = []
    
    if not secret_key or secret_key in DEFAULT_SECRETS:
        unsafe_secrets.append("SECRET_KEY")
    
    if not api_salt or api_salt in DEFAULT_SECRETS:
        unsafe_secrets.append("API_KEY_SALT")
    
    if unsafe_secrets:
        print(
            f"SECURITY ERROR: Default secrets detected in production for: {', '.join(unsafe_secrets)}",
            file=sys.stderr
        )
        print(
            "Generate secure secrets with: python -c 'import secrets; print(secrets.token_urlsafe(32))'",
            file=sys.stderr
        )
        sys.exit(1)


def generate_secure_key() -> str:
    """Generate a cryptographically secure random key."""
    return secrets.token_urlsafe(32)


def check_cors_origins() -> list[str]:
    """
    Get validated CORS origins.
    
    Returns empty list in production if not explicitly set.
    """
    environment = os.getenv("ENVIRONMENT", "development")
    cors_origins = os.getenv("CORS_ORIGINS", "")
    
    if not cors_origins:
        # Development default
        if environment == "development":
            return ["http://localhost:3000", "http://localhost:8000"]
        # Production requires explicit CORS
        return []
    
    # Parse comma-separated origins
    origins = [origin.strip() for origin in cors_origins.split(",")]
    
    # Warn about wildcards in production
    if environment == "production" and "*" in cors_origins:
        print(
            "WARNING: Wildcard CORS origin detected in production!",
            file=sys.stderr
        )
    
    return origins


def get_secure_headers() -> dict[str, str]:
    """
    Get security headers for API responses.
    
    Based on OWASP recommendations.
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }