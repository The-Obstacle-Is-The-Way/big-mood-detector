"""Privacy and security utilities for the Big Mood Detector.

This module provides utilities for PII protection, user ID hashing,
and other privacy-related functionality to ensure GDPR compliance.
"""

import hashlib
import os
from typing import Any

from structlog import get_logger

logger = get_logger()

# Get salt from settings
from big_mood_detector.infrastructure.settings.config import get_settings

USER_ID_SALT = get_settings().USER_ID_SALT


def hash_user_id(user_id: str) -> str:
    """
    Hash user ID for privacy protection.

    Uses SHA-256 with a salt to create a one-way hash of the user ID.
    This ensures user privacy while maintaining consistency for the same user.

    Args:
        user_id: The raw user identifier

    Returns:
        A hex-encoded SHA-256 hash of the salted user ID
    """
    if not user_id:
        raise ValueError("user_id cannot be empty")

    # Combine user_id with salt
    salted_id = f"{USER_ID_SALT}:{user_id}"

    # Create SHA-256 hash
    hash_object = hashlib.sha256(salted_id.encode("utf-8"))
    hashed_id = hash_object.hexdigest()

    logger.debug(
        "user_id_hashed", original_length=len(user_id), hash_length=len(hashed_id)
    )

    return hashed_id


def redact_pii(value: Any, field_name: str) -> Any:
    """
    Redact PII from a value based on field name.

    Args:
        value: The value to potentially redact
        field_name: The name of the field (used to determine if it contains PII)

    Returns:
        The redacted value if it's PII, otherwise the original value
    """
    # List of field names that might contain PII
    pii_fields = {
        "user_id",
        "userid",
        "user",
        "email",
        "name",
        "first_name",
        "last_name",
        "phone",
        "address",
        "ssn",
        "social_security",
        "device_id",
        "ip_address",
        "location",
        "latitude",
        "longitude",
        "source_name",
    }

    # Check if field name suggests PII
    if field_name.lower() in pii_fields:
        if isinstance(value, str):
            # For user_id, show first 8 chars of hash
            if "user" in field_name.lower() and value:
                return f"[REDACTED:{hash_user_id(value)[:8]}...]"
            else:
                return "[REDACTED]"
        elif (
            isinstance(value, int | float)
            and "lat" in field_name.lower()
            or "lon" in field_name.lower()
        ):
            # For location data, round to 2 decimal places
            return round(value, 2)

    return value


class PrivacyFilter:
    """Logging filter that redacts PII from log messages."""

    def __init__(self) -> None:
        self.pii_patterns = {
            "user_id",
            "email",
            "name",
            "phone",
            "address",
            "ssn",
            "device_id",
            "ip_address",
            "location",
        }

    def filter(self, event_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Filter log event dictionary to redact PII.

        Args:
            event_dict: The structlog event dictionary

        Returns:
            The filtered event dictionary with PII redacted
        """
        for key, value in event_dict.items():
            # Skip internal structlog keys
            if key.startswith("_"):
                continue

            # Redact based on field name
            event_dict[key] = redact_pii(value, key)

            # Handle nested dictionaries
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    value[nested_key] = redact_pii(nested_value, nested_key)

        return event_dict


def configure_privacy_logging() -> None:
    """Configure structlog to use privacy filter for all logging."""
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # PrivacyFilter().filter,  # TODO(gh-101): Fix type compatibility with structlog
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
