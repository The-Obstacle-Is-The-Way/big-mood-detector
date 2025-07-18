"""
Test Error Handling Infrastructure

TDD for comprehensive error handling with proper patterns.
"""

from unittest.mock import Mock

import pytest


class TestErrorHandling:
    """Test error handling patterns and infrastructure."""

    def test_error_classes_can_be_imported(self):
        """Test that error classes can be imported."""
        from big_mood_detector.core.exceptions import (
            BigMoodError,
            DataParsingError,
            ValidationError,
        )

        assert BigMoodError is not None
        assert ValidationError is not None
        assert DataParsingError is not None

    def test_base_error_hierarchy(self):
        """Test error class hierarchy."""
        from big_mood_detector.core.exceptions import (
            BigMoodError,
            DataParsingError,
            ValidationError,
        )

        # All should inherit from BigMoodError
        assert issubclass(ValidationError, BigMoodError)
        assert issubclass(DataParsingError, BigMoodError)

    def test_error_with_context(self):
        """Test errors can carry context information."""
        from big_mood_detector.core.exceptions import ValidationError

        error = ValidationError(
            "Invalid date format",
            field="start_date",
            value="2024-13-01",
            expected_format="YYYY-MM-DD",
        )

        assert str(error) == "Invalid date format"
        assert error.context["field"] == "start_date"
        assert error.context["value"] == "2024-13-01"
        assert error.context["expected_format"] == "YYYY-MM-DD"

    def test_error_chaining(self):
        """Test error chaining for root cause tracking."""
        from big_mood_detector.core.exceptions import DataParsingError, ProcessingError

        # Original error
        original = ValueError("Invalid JSON")

        # Wrap in parsing error
        parsing_error = DataParsingError(
            "Failed to parse health data", file_path="/tmp/data.json", line_number=42
        ).with_cause(original)

        # Wrap in processing error
        processing_error = ProcessingError("Data processing failed").with_cause(
            parsing_error
        )

        # Should maintain chain
        assert processing_error.__cause__ is parsing_error
        assert parsing_error.__cause__ is original

    def test_error_handler_decorator(self):
        """Test error handler decorator for consistent handling."""
        from big_mood_detector.core.exceptions import ValidationError, handle_errors

        @handle_errors(default_return=None, reraise=False)
        def risky_function(value: int):
            if value < 0:
                raise ValidationError("Value must be positive", value=value)
            return value * 2

        # Normal operation
        assert risky_function(5) == 10

        # Error handling
        result = risky_function(-1)
        assert result is None

    def test_error_handler_with_logging(self):
        """Test error handler logs errors properly."""
        from big_mood_detector.core.exceptions import ProcessingError, handle_errors

        mock_logger = Mock()

        @handle_errors(logger=mock_logger)
        def failing_function():
            raise ProcessingError("Something went wrong")

        # Should log and re-raise
        with pytest.raises(ProcessingError):
            failing_function()

        # Check logging
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "error_occurred" in call_args[0]

    def test_global_error_handler(self):
        """Test global error handler for uncaught exceptions."""
        from big_mood_detector.core.exceptions import setup_global_error_handler

        mock_logger = Mock()

        # Setup handler
        setup_global_error_handler(mock_logger)

        # Test it catches uncaught exceptions
        # (This is tricky to test without actually crashing)

    def test_error_response_builder(self):
        """Test building consistent error responses."""
        from big_mood_detector.core.exceptions import (
            ValidationError,
            build_error_response,
        )

        error = ValidationError(
            "Invalid file format", file_type="PDF", allowed_types=["XML", "JSON"]
        )

        response = build_error_response(error, request_id="123")

        assert response.error_type == "ValidationError"
        assert response.message == "Invalid file format"
        assert response.request_id == "123"
        assert response.details["file_type"] == "PDF"
        assert response.details["allowed_types"] == ["XML", "JSON"]

    def test_retry_on_error_decorator(self):
        """Test retry decorator for transient errors."""
        from big_mood_detector.core.exceptions import (
            ExternalServiceError,
            retry_on_error,
        )

        call_count = 0

        @retry_on_error(
            max_attempts=3,
            exceptions=(ExternalServiceError,),
            delay=0.01,  # Short delay for testing
        )
        def flaky_function():
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise ExternalServiceError("Service unavailable")

            return "success"

        result = flaky_function()

        assert result == "success"
        assert call_count == 3

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker for external service calls."""
        from big_mood_detector.core.exceptions import (
            CircuitBreaker,
            CircuitOpenError,
            ExternalServiceError,
        )

        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            expected_exception=ExternalServiceError,
            name="test_breaker",
        )

        @breaker
        def external_call():
            raise ExternalServiceError("Service down")

        # Check initial state
        assert breaker.state == "closed"
        assert breaker.metrics["failure_count"] == 0

        # First two calls should fail normally
        with pytest.raises(ExternalServiceError):
            external_call()

        with pytest.raises(ExternalServiceError):
            external_call()

        # Circuit should now be open
        assert breaker.state == "open"
        assert breaker.metrics["failure_count"] == 2

        with pytest.raises(CircuitOpenError):
            external_call()

    def test_circuit_breaker_metrics_tracking(self):
        """Test that circuit breaker tracks metrics."""
        from big_mood_detector.core.exceptions import (
            CircuitBreaker,
            ExternalServiceError,
            error_metrics,
        )

        # Reset metrics
        error_metrics.reset()

        breaker = CircuitBreaker(
            failure_threshold=1,
            expected_exception=ExternalServiceError,
            name="metrics_test",
        )

        @breaker
        def flaky_service():
            raise ExternalServiceError("Down")

        # Trigger failure
        with pytest.raises(ExternalServiceError):
            flaky_service()

        # Check metrics
        stats = error_metrics.get_stats()
        assert stats["total_errors"] >= 2  # call_failed + state_change
        assert "CircuitBreaker" in stats["by_type"]

    def test_validation_error_collection(self):
        """Test collecting multiple validation errors."""
        from big_mood_detector.core.exceptions import (
            ValidationError,
            ValidationErrorCollector,
        )

        collector = ValidationErrorCollector()

        # Add errors
        collector.add_error("name", "Name is required")
        collector.add_error("age", "Age must be positive", value=-5)
        collector.add_error("email", "Invalid email format", value="not-an-email")

        # Check if has errors
        assert collector.has_errors()
        assert len(collector.errors) == 3

        # Raise if errors
        with pytest.raises(ValidationError) as exc_info:
            collector.raise_if_errors()

        error = exc_info.value
        assert "3 validation errors" in str(error)
        assert len(error.context["errors"]) == 3

    def test_error_metrics_tracking(self):
        """Test tracking error metrics."""
        from big_mood_detector.core.exceptions import ErrorMetrics

        metrics = ErrorMetrics()

        # Track some errors
        metrics.record_error("ValidationError", "invalid_date")
        metrics.record_error("ValidationError", "invalid_date")
        metrics.record_error("ProcessingError", "timeout")

        # Get metrics
        stats = metrics.get_stats()

        assert stats["total_errors"] == 3
        assert stats["by_type"]["ValidationError"] == 2
        assert stats["by_type"]["ProcessingError"] == 1

    def test_domain_specific_errors(self):
        """Test domain-specific error types."""
        from big_mood_detector.core.exceptions import (
            ClinicalThresholdError,
            InsufficientDataError,
            ModelNotFoundError,
        )

        # Insufficient data error
        error1 = InsufficientDataError(
            "Not enough sleep data",
            required_days=7,
            available_days=3,
            data_type="sleep",
        )
        assert error1.context["required_days"] == 7

        # Clinical threshold error
        error2 = ClinicalThresholdError(
            "Confidence below clinical threshold",
            threshold=0.7,
            actual=0.45,
            metric="depression_risk",
        )
        assert error2.context["threshold"] == 0.7

        # Model not found
        error3 = ModelNotFoundError(
            "XGBoost model not found",
            model_path="/models/xgboost.pkl",
            model_type="xgboost",
        )
        assert error3.context["model_path"] == "/models/xgboost.pkl"

    def test_keyword_only_decorators(self):
        """Test that decorators require keyword-only arguments."""
        from big_mood_detector.core.exceptions import handle_errors

        # Should work with keyword args
        @handle_errors(reraise=False, default_return=42)
        def func1():
            raise ValueError()

        assert func1() == 42

        # Should fail with positional args
        with pytest.raises(TypeError):

            @handle_errors(42, None, False)  # Positional not allowed
            def func2():
                pass
