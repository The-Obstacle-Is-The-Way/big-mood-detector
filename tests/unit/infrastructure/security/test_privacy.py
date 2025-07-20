"""Unit tests for privacy module."""

import os
from unittest.mock import patch

import pytest

from big_mood_detector.infrastructure.security.privacy import (
    PrivacyFilter,
    hash_user_id,
    redact_pii,
)


class TestHashUserId:
    """Test user ID hashing functionality."""

    def test_hash_user_id_consistent(self):
        """Test that same user ID produces same hash."""
        user_id = "alice@example.com"
        hash1 = hash_user_id(user_id)
        hash2 = hash_user_id(user_id)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

    def test_hash_user_id_different_users(self):
        """Test that different user IDs produce different hashes."""
        hash1 = hash_user_id("alice@example.com")
        hash2 = hash_user_id("bob@example.com")
        
        assert hash1 != hash2

    def test_hash_user_id_with_salt(self):
        """Test that salt affects the hash."""
        user_id = "alice@example.com"
        
        # Hash with default salt
        hash_default = hash_user_id(user_id)
        
        # Hash with custom salt by patching the module-level constant
        with patch("big_mood_detector.infrastructure.security.privacy.USER_ID_SALT", "custom-salt"):
            hash_custom = hash_user_id(user_id)
        
        assert hash_default != hash_custom

    def test_hash_user_id_empty_raises(self):
        """Test that empty user ID raises ValueError."""
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            hash_user_id("")


class TestRedactPII:
    """Test PII redaction functionality."""

    def test_redact_user_id(self):
        """Test user ID redaction shows partial hash."""
        result = redact_pii("alice@example.com", "user_id")
        
        assert result.startswith("[REDACTED:")
        assert result.endswith("...]")
        assert "alice" not in result
        assert "@example.com" not in result

    def test_redact_email(self):
        """Test email redaction."""
        result = redact_pii("alice@example.com", "email")
        
        assert result == "[REDACTED]"
        assert "alice" not in result

    def test_redact_name(self):
        """Test name redaction."""
        result = redact_pii("Alice Smith", "name")
        
        assert result == "[REDACTED]"
        assert "Alice" not in result
        assert "Smith" not in result

    def test_redact_location_coordinates(self):
        """Test location coordinate rounding."""
        # Latitude should be rounded to 2 decimal places
        lat_result = redact_pii(37.7749295, "latitude")
        assert lat_result == 37.77
        
        # Longitude should be rounded to 2 decimal places
        lon_result = redact_pii(-122.4194155, "longitude")
        assert lon_result == -122.42

    def test_no_redaction_for_safe_fields(self):
        """Test that non-PII fields are not redacted."""
        assert redact_pii("some_value", "sleep_duration") == "some_value"
        assert redact_pii(42.5, "heart_rate") == 42.5
        assert redact_pii(True, "is_active") is True

    def test_case_insensitive_field_matching(self):
        """Test that field name matching is case-insensitive."""
        assert redact_pii("alice@example.com", "USER_ID").startswith("[REDACTED:")
        assert redact_pii("alice@example.com", "Email") == "[REDACTED]"
        assert redact_pii("Alice Smith", "NAME") == "[REDACTED]"


class TestPrivacyFilter:
    """Test privacy filter for logging."""

    def test_filter_redacts_pii_fields(self):
        """Test that filter redacts PII fields in log events."""
        filter = PrivacyFilter()
        
        event_dict = {
            "event": "user_login",
            "user_id": "alice@example.com",
            "name": "Alice Smith",  # Changed from user_name to name
            "ip_address": "192.168.1.100",
            "timestamp": "2023-01-01T00:00:00Z",
            "success": True,
        }
        
        filtered = filter.filter(event_dict)
        
        # Check PII fields are redacted
        assert filtered["user_id"].startswith("[REDACTED:")
        assert "alice" not in str(filtered["user_id"])
        assert filtered["name"] == "[REDACTED]"
        assert "Alice" not in str(filtered["name"])
        assert "Smith" not in str(filtered["name"])
        assert filtered["ip_address"] == "[REDACTED]"
        
        # Check non-PII fields are preserved
        assert filtered["event"] == "user_login"
        assert filtered["timestamp"] == "2023-01-01T00:00:00Z"
        assert filtered["success"] is True

    def test_filter_handles_nested_dicts(self):
        """Test that filter handles nested dictionaries."""
        filter = PrivacyFilter()
        
        event_dict = {
            "event": "profile_update",
            "user": {
                "user_id": "alice@example.com",
                "name": "Alice Smith",
                "age": 30,
            },
            "changes": {
                "email": "newalice@example.com",
                "preferences": {"theme": "dark"},
            },
        }
        
        filtered = filter.filter(event_dict)
        
        # Check nested PII is redacted
        assert filtered["user"]["user_id"].startswith("[REDACTED:")
        assert filtered["user"]["name"] == "[REDACTED]"
        assert filtered["changes"]["email"] == "[REDACTED]"
        
        # Check non-PII nested fields are preserved
        assert filtered["user"]["age"] == 30
        assert filtered["changes"]["preferences"]["theme"] == "dark"

    def test_filter_skips_internal_keys(self):
        """Test that filter skips internal structlog keys."""
        filter = PrivacyFilter()
        
        event_dict = {
            "_record": "internal",
            "_from_structlog": True,
            "user_id": "alice@example.com",
            "message": "User logged in",
        }
        
        filtered = filter.filter(event_dict)
        
        # Internal keys should not be modified
        assert filtered["_record"] == "internal"
        assert filtered["_from_structlog"] is True
        
        # But user_id should still be redacted
        assert filtered["user_id"].startswith("[REDACTED:")


class TestPrivacyIntegration:
    """Test privacy features in realistic scenarios."""

    def test_logging_with_pii_redaction(self):
        """Test that PII is redacted in actual log output."""
        # Create a wrapper that adapts our filter to structlog's processor interface
        def privacy_processor(logger, method_name, event_dict):
            """Adapt PrivacyFilter to structlog processor signature."""
            return PrivacyFilter().filter(event_dict)
        
        import structlog
        from io import StringIO
        
        # Capture log output
        log_output = StringIO()
        
        # Configure structlog with our privacy filter
        structlog.configure(
            processors=[
                privacy_processor,
                structlog.dev.ConsoleRenderer(),
            ],
            logger_factory=lambda: structlog.PrintLogger(file=log_output),
            cache_logger_on_first_use=False,
        )
        
        # Log event with PII
        logger = structlog.get_logger()
        logger.info(
            "user_activity",
            user_id="alice@example.com",
            name="Alice Smith",
            action="login",
            session_id="abc123",
        )
        
        # Check output
        output = log_output.getvalue()
        
        # PII should be redacted
        assert "alice@example.com" not in output
        assert "Alice Smith" not in output
        assert "[REDACTED" in output
        
        # Non-PII should be preserved
        assert "user_activity" in output
        assert "login" in output
        assert "abc123" in output