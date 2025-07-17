"""
Error Handling Infrastructure

Comprehensive error handling with proper patterns and context.
Following Clean Code principles for exception handling.
"""

import functools
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from big_mood_detector.core.logging import get_module_logger

logger = get_module_logger(__name__)


class BigMoodError(Exception):
    """
    Base exception for all Big Mood Detector errors.

    Provides context storage and error chaining capabilities.
    """

    def __init__(self, message: str, **context):
        """
        Initialize with message and optional context.

        Args:
            message: Error message
            **context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.context = context
        self.timestamp = datetime.utcnow()

    def with_cause(self, cause: Exception) -> "BigMoodError":
        """
        Chain this error with a cause.

        Args:
            cause: The underlying exception

        Returns:
            Self for method chaining
        """
        self.__cause__ = cause
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.__cause__) if self.__cause__ else None,
        }


# Validation Errors
class ValidationError(BigMoodError):
    """Raised when input validation fails."""

    pass


class DataParsingError(BigMoodError):
    """Raised when data parsing fails."""

    pass


class ProcessingError(BigMoodError):
    """Raised when data processing fails."""

    pass


class ConfigurationError(BigMoodError):
    """Raised when configuration is invalid."""

    pass


class NotFoundError(BigMoodError):
    """Raised when a resource is not found."""

    pass


class PermissionError(BigMoodError):
    """Raised when permission is denied."""

    pass


class ExternalServiceError(BigMoodError):
    """Raised when external service call fails."""

    pass


class CircuitOpenError(BigMoodError):
    """Raised when circuit breaker is open."""

    pass


# Domain-specific errors
class InsufficientDataError(BigMoodError):
    """Raised when there's not enough data for analysis."""

    pass


class ClinicalThresholdError(BigMoodError):
    """Raised when clinical thresholds are not met."""

    pass


class ModelNotFoundError(BigMoodError):
    """Raised when ML model is not found."""

    pass


@dataclass
class ErrorResponse:
    """Standardized error response."""

    error_type: str
    message: str
    details: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": {
                "type": self.error_type,
                "message": self.message,
                "details": self.details,
                "timestamp": self.timestamp.isoformat(),
                "request_id": self.request_id,
            }
        }


def build_error_response(
    error: Exception, request_id: str | None = None
) -> ErrorResponse:
    """
    Build standardized error response from exception.

    Args:
        error: The exception
        request_id: Optional request ID for tracking

    Returns:
        ErrorResponse instance
    """
    if isinstance(error, BigMoodError):
        return ErrorResponse(
            error_type=error.__class__.__name__,
            message=str(error),
            details=error.context,
            request_id=request_id,
        )
    else:
        # Generic error
        return ErrorResponse(
            error_type=error.__class__.__name__,
            message=str(error),
            details={"traceback": traceback.format_exc()},
            request_id=request_id,
        )


def handle_errors(
    *,
    default_return: Any = None,
    logger: Any | None = None,
    reraise: bool = True,
) -> Callable:
    """
    Decorator for consistent error handling.

    Args:
        default_return: Value to return on error (if not reraising)
        logger: Logger instance to use
        reraise: Whether to reraise the exception

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_module_logger(func.__module__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                _logger.error(
                    "error_occurred",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )

                if reraise:
                    raise

                return default_return

        return wrapper

    return decorator


def retry_on_error(
    *,
    max_attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable:
    """
    Decorator for retrying on specific errors.

    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exceptions to retry on
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            error=str(e),
                            delay=current_delay,
                        )

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )

            raise last_exception

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls.

    Prevents cascading failures by opening circuit after
    threshold failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
        name: str | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to track
            name: Optional name for this breaker (for metrics)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or f"breaker_{id(self)}"

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half-open

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    @property
    def metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self._last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }

    def __call__(self, func: Callable) -> Callable:
        """Decorate function with circuit breaker."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self._state == "open":
                if self._should_attempt_recovery():
                    self._state = "half-open"
                else:
                    raise CircuitOpenError(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception:
                self._on_failure()
                raise

        return wrapper

    def _should_attempt_recovery(self) -> bool:
        """Check if we should attempt recovery."""
        return (
            self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful call."""
        old_state = self._state
        self._failure_count = 0
        self._state = "closed"

        # Track state change
        if old_state != "closed":
            error_metrics.record_error(
                "CircuitBreaker", f"state_change_{old_state}_to_closed"
            )
            logger.info("circuit_breaker_closed", previous_state=old_state)

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        # Track failure
        error_metrics.record_error("CircuitBreaker", "call_failed")

        if self._failure_count >= self.failure_threshold:
            old_state = self._state
            self._state = "open"

            # Track state change
            error_metrics.record_error(
                "CircuitBreaker", f"state_change_{old_state}_to_open"
            )

            logger.warning(
                "circuit_breaker_opened",
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
            )


class ValidationErrorCollector:
    """
    Collects multiple validation errors.

    Useful for form validation where you want to show
    all errors at once.
    """

    def __init__(self):
        """Initialize collector."""
        self.errors: list[dict[str, Any]] = []

    def add_error(self, field: str, message: str, **context) -> None:
        """
        Add a validation error.

        Args:
            field: Field name
            message: Error message
            **context: Additional context
        """
        self.errors.append(
            {
                "field": field,
                "message": message,
                **context,
            }
        )

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def raise_if_errors(self) -> None:
        """Raise ValidationError if there are errors."""
        if self.has_errors():
            raise ValidationError(
                f"{len(self.errors)} validation errors",
                errors=self.errors,
            )

    def clear(self) -> None:
        """Clear all errors."""
        self.errors.clear()


class ErrorMetrics:
    """
    Track error metrics for monitoring.

    Thread-safe error metrics collection.
    """

    def __init__(self):
        """Initialize metrics."""
        self._errors: dict[str, dict[str, int]] = {}
        self._total = 0

    def record_error(self, error_type: str, error_code: str) -> None:
        """
        Record an error occurrence.

        Args:
            error_type: Type of error (e.g., ValidationError)
            error_code: Specific error code
        """
        if error_type not in self._errors:
            self._errors[error_type] = {}

        if error_code not in self._errors[error_type]:
            self._errors[error_type][error_code] = 0

        self._errors[error_type][error_code] += 1
        self._total += 1

        logger.debug(
            "error_recorded",
            error_type=error_type,
            error_code=error_code,
            count=self._errors[error_type][error_code],
        )

    def get_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        by_type = {}

        for error_type, codes in self._errors.items():
            by_type[error_type] = sum(codes.values())

        return {
            "total_errors": self._total,
            "by_type": by_type,
            "details": self._errors,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._errors.clear()
        self._total = 0


def setup_global_error_handler(logger_instance=None) -> None:
    """
    Set up global error handler for uncaught exceptions.

    Args:
        logger_instance: Logger to use (defaults to module logger)
    """
    _logger = logger_instance or logger

    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        _logger.critical(
            "uncaught_exception",
            exc_type=exc_type.__name__,
            exc_value=str(exc_value),
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception


# Global error metrics instance
error_metrics = ErrorMetrics()
