"""Test user ID validation for CLI commands."""

import click
import pytest

class TestUserIdValidation:
    """Test user ID validation logic."""

    def test_valid_user_ids(self):
        """Test that valid user IDs pass validation."""
        from big_mood_detector.interfaces.cli.commands import validate_user_id

        # Should not raise
        validate_user_id("john_doe")
        validate_user_id("user-123")
        validate_user_id("User123")
        validate_user_id("test_user_2024")
        validate_user_id("abc")  # Minimum 3 chars
        validate_user_id("a" * 64)  # Maximum 64 chars
        validate_user_id(None)  # Optional parameter

    def test_user_id_with_spaces(self):
        """Test that user IDs with spaces are rejected."""
        from big_mood_detector.interfaces.cli.commands import validate_user_id

        with pytest.raises(click.BadParameter, match="cannot contain spaces"):
            validate_user_id("john doe")

    def test_user_id_too_short(self):
        """Test that very short user IDs are rejected."""
        from big_mood_detector.interfaces.cli.commands import validate_user_id

        with pytest.raises(click.BadParameter, match="too short"):
            validate_user_id("ab")

    def test_user_id_too_long(self):
        """Test that very long user IDs are rejected."""
        from big_mood_detector.interfaces.cli.commands import validate_user_id

        with pytest.raises(click.BadParameter, match="too long"):
            validate_user_id("a" * 65)

    def test_user_id_with_invalid_characters(self):
        """Test that user IDs with invalid characters are rejected."""
        from big_mood_detector.interfaces.cli.commands import validate_user_id

        with pytest.raises(click.BadParameter, match="Invalid user ID format"):
            validate_user_id("user@#$%")

        with pytest.raises(click.BadParameter, match="Invalid user ID format"):
            validate_user_id("user/test")

        with pytest.raises(click.BadParameter, match="Invalid user ID format"):
            validate_user_id("user.name")

    def test_user_id_with_email_warning(self, capsys):
        """Test that email-like user IDs trigger a warning."""
        from big_mood_detector.interfaces.cli.commands import validate_user_id

        # Should not raise but should warn
        validate_user_id("user@example")

        captured = capsys.readouterr()
        assert "avoid using email addresses" in captured.out
        assert "hashed for privacy protection" in captured.out

    def test_edge_cases(self):
        """Test edge cases for user ID validation."""
        from big_mood_detector.interfaces.cli.commands import validate_user_id

        # Single character repeated
        validate_user_id("aaa")

        # All numbers
        validate_user_id("123456")

        # Mix of valid characters
        validate_user_id("Test_User-123")
