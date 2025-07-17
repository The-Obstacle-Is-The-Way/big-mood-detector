"""
Test Logging Infrastructure

TDD for structured logging with proper configuration.
"""

import json
import logging
from io import StringIO
from unittest.mock import Mock, patch

import pytest


class TestLogging:
    """Test logging configuration and usage."""

    def test_logger_can_be_imported(self):
        """Test that logger module can be imported."""
        from big_mood_detector.core.logging import get_logger, logger
        
        assert get_logger is not None
        assert logger is not None

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a logger instance."""
        from big_mood_detector.core.logging import get_logger
        
        logger = get_logger()
        
        # Should have logging methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_logger_singleton_pattern(self):
        """Test that get_logger uses singleton pattern (cached)."""
        from big_mood_detector.core.logging import get_logger
        
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should be the same instance (cached)
        assert logger1 is logger2

    @patch("sys.stdout", new_callable=StringIO)
    def test_json_logging_format(self, mock_stdout):
        """Test JSON logging format when configured."""
        from big_mood_detector.core.config import Settings
        from big_mood_detector.core.logging import setup_logging
        
        # Configure for JSON
        settings = Settings(LOG_FORMAT="json", LOG_LEVEL="INFO")
        logger = setup_logging(settings)
        
        # Log a message
        logger.info("test_message", user_id=123, action="login")
        
        # Get output
        output = mock_stdout.getvalue()
        
        # Should be valid JSON
        log_entry = json.loads(output.strip())
        assert log_entry["event"] == "test_message"
        assert log_entry["user_id"] == 123
        assert log_entry["action"] == "login"
        assert "timestamp" in log_entry

    @patch("sys.stdout", new_callable=StringIO)
    def test_text_logging_format(self, mock_stdout):
        """Test text logging format when configured."""
        from big_mood_detector.core.config import Settings
        from big_mood_detector.core.logging import setup_logging
        
        # Configure for text
        settings = Settings(LOG_FORMAT="text", LOG_LEVEL="INFO")
        logger = setup_logging(settings)
        
        # Log a message
        logger.info("test_message")
        
        # Get output
        output = mock_stdout.getvalue()
        
        # Should contain the message but not be JSON
        assert "test_message" in output
        with pytest.raises(json.JSONDecodeError):
            json.loads(output.strip())

    def test_log_level_configuration(self):
        """Test that log level is properly configured."""
        from big_mood_detector.core.config import Settings
        from big_mood_detector.core.logging import setup_logging
        
        # Test different log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            settings = Settings(LOG_LEVEL=level)
            logger = setup_logging(settings)
            
            # Get the effective level
            effective_level = logging.getLevelName(
                logging.getLogger().getEffectiveLevel()
            )
            assert effective_level == level

    @patch("sys.stdout", new_callable=StringIO)
    def test_structured_logging_with_context(self, mock_stdout):
        """Test structured logging with context data."""
        from big_mood_detector.core.config import Settings
        from big_mood_detector.core.logging import setup_logging
        
        settings = Settings(LOG_FORMAT="json")
        logger = setup_logging(settings)
        
        # Log with structured data
        logger.info(
            "processing_file",
            file_path="/tmp/test.xml",
            file_size=1024,
            processing_time=0.5,
        )
        
        # Verify structured data
        output = mock_stdout.getvalue()
        log_entry = json.loads(output.strip())
        
        assert log_entry["event"] == "processing_file"
        assert log_entry["file_path"] == "/tmp/test.xml"
        assert log_entry["file_size"] == 1024
        assert log_entry["processing_time"] == 0.5

    def test_logger_adapter_pattern(self):
        """Test logger adapter for adding context."""
        from big_mood_detector.core.logging import LoggerAdapter
        
        base_logger = Mock()
        adapter = LoggerAdapter(base_logger, {"request_id": "123"})
        
        # Log through adapter
        adapter.info("test", extra_field="value")
        
        # Should add context
        base_logger.info.assert_called_once()
        call_args = base_logger.info.call_args
        assert "request_id" in call_args[1]
        assert call_args[1]["request_id"] == "123"
        assert "extra_field" in call_args[1]

    @patch("sys.stdout", new_callable=StringIO)
    def test_exception_logging(self, mock_stdout):
        """Test exception logging with stack trace."""
        from big_mood_detector.core.config import Settings
        from big_mood_detector.core.logging import setup_logging
        
        settings = Settings(LOG_FORMAT="json")
        logger = setup_logging(settings)
        
        # Log an exception
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("error_occurred", error_type="ValueError")
        
        # Verify exception info is logged
        output = mock_stdout.getvalue()
        log_entry = json.loads(output.strip())
        
        assert log_entry["event"] == "error_occurred"
        assert log_entry["error_type"] == "ValueError"
        assert "exception" in log_entry
        assert "ValueError: Test error" in log_entry["exception"]

    def test_performance_logging_decorator(self):
        """Test performance logging decorator."""
        from big_mood_detector.core.logging import log_performance
        
        mock_logger = Mock()
        
        @log_performance(mock_logger)
        def slow_function(x: int) -> int:
            import time
            time.sleep(0.1)
            return x * 2
        
        result = slow_function(5)
        
        assert result == 10
        mock_logger.info.assert_called_once()
        
        # Check logged data
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "function_completed"
        assert "function_name" in call_args[1]
        assert call_args[1]["function_name"] == "slow_function"
        assert "duration" in call_args[1]
        assert call_args[1]["duration"] >= 0.1

    def test_log_context_manager(self):
        """Test context manager for temporary log context."""
        from big_mood_detector.core.logging import log_context
        
        mock_logger = Mock()
        
        # Use context manager
        with log_context(mock_logger, operation="test_op", user_id=123):
            mock_logger.info("inside_context")
        
        # Should log start and end
        assert mock_logger.info.call_count >= 2
        
        # Check context was added
        calls = mock_logger.info.call_args_list
        for call in calls:
            if "operation" in call[1]:
                assert call[1]["operation"] == "test_op"
                assert call[1]["user_id"] == 123

    def test_sanitize_sensitive_data(self):
        """Test that sensitive data is sanitized in logs."""
        from big_mood_detector.core.logging import sanitize_log_data
        
        sensitive_data = {
            "user_id": "123",
            "password": "secret123",
            "api_key": "sk-1234567890",
            "email": "user@example.com",
            "safe_field": "visible",
        }
        
        sanitized = sanitize_log_data(sensitive_data)
        
        assert sanitized["user_id"] == "123"
        assert sanitized["password"] == "***"
        assert sanitized["api_key"] == "sk-****"
        assert sanitized["email"] == "u***@example.com"
        assert sanitized["safe_field"] == "visible"

    def test_module_logger_pattern(self):
        """Test module-level logger usage pattern."""
        # This tests the recommended pattern for module loggers
        from big_mood_detector.core.logging import get_module_logger
        
        # Get logger for a module
        logger = get_module_logger("big_mood_detector.domain.services")
        
        # Should have the module name
        assert hasattr(logger, "name")
        assert "big_mood_detector.domain.services" in str(logger)