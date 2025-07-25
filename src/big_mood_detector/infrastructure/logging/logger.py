"""
Logging Infrastructure

Structured logging with proper configuration and patterns.
Following Single Responsibility Principle - each component has one job.
"""

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, cast

import structlog
from structlog.types import FilteringBoundLogger

from big_mood_detector.infrastructure.settings import Settings, get_settings


@lru_cache
def get_logger() -> FilteringBoundLogger:
    """Get configured logger instance (cached singleton)."""
    return setup_logging(get_settings())


def _sanitize_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor that sanitizes sensitive data before rendering."""
    # Sanitize the entire event dict
    return sanitize_log_data(event_dict)


def setup_logging(config: Settings) -> FilteringBoundLogger:
    """Set up structured logging based on configuration.

    Args:
        config: Settings instance with logging configuration

    Returns:
        Configured structlog logger
    """
    # Configure structlog processors first
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        timestamper,
        _sanitize_processor,  # Sanitize BEFORE rendering
    ]

    if config.LOG_FORMAT == "json":
        # JSON output direct to console
        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict[str, Any],
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )
    else:
        # Human-readable console output
        structlog.configure(
            processors=shared_processors + [structlog.dev.ConsoleRenderer()],
            context_class=dict[str, Any],
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )

    return cast(FilteringBoundLogger, structlog.get_logger())


def get_module_logger(name: str) -> FilteringBoundLogger:
    """Get a logger for a specific module.

    Args:
        name: Module name (e.g., "big_mood_detector.domain.services")

    Returns:
        Logger bound to the module name
    """
    base_logger = get_logger()
    return base_logger.bind(logger=name)


class LoggerAdapter:
    """Adapter pattern for adding context to logger.

    This follows the Adapter pattern from GoF - adapts the logger
    interface to include additional context.
    """

    def __init__(self, logger: FilteringBoundLogger, context: dict[str, Any]):
        """Initialize adapter with logger and context.

        Args:
            logger: Base logger to adapt
            context: Context to add to all log messages
        """
        self.logger = logger
        self.context = context

    def _log(self, method: str, event: str, **kwargs: Any) -> None:
        """Internal log method that adds context."""
        combined_context = {**self.context, **kwargs}
        getattr(self.logger, method)(event, **combined_context)

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self._log("error", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log exception with context."""
        self._log("exception", event, **kwargs)


def log_performance(logger: FilteringBoundLogger | None = None) -> Callable[..., Any]:
    """Decorator for logging function performance.

    This follows the Decorator pattern - adds behavior without
    modifying the original function.

    Args:
        logger: Logger to use (defaults to module logger)

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            _logger = logger or get_logger()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                _logger.info(
                    "function_completed",
                    function_name=func.__name__,
                    duration=duration,
                    success=True,
                )

                return result
            except Exception as e:
                duration = time.time() - start_time

                _logger.exception(
                    "function_failed",
                    function_name=func.__name__,
                    duration=duration,
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator


@contextmanager
def log_context(
    logger: FilteringBoundLogger, **context: Any
) -> Iterator[FilteringBoundLogger]:
    """Context manager for temporary logging context.

    Args:
        logger: Logger to use
        **context: Context to add for duration of block

    Yields:
        Logger with added context
    """
    logger.info("context_started", **context)

    try:
        yield logger
    finally:
        logger.info("context_ended", **context)


def sanitize_log_data(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize sensitive data before logging.

    This follows the Strategy pattern - different sanitization
    strategies for different data types.

    Args:
        data: Data to sanitize

    Returns:
        Sanitized data safe for logging
    """
    sensitive_fields = {
        "password": lambda x: "***",
        "api_key": lambda x: x[:3] + "****" if len(x) > 3 else "****",
        "secret": lambda x: "***",
        "token": lambda x: "***",
        "email": lambda x: x[0] + "***" + x[x.find("@") :] if "@" in x else "***",
    }

    sanitized = {}

    for key, value in data.items():
        if key.lower() in sensitive_fields:
            sanitized[key] = sensitive_fields[key.lower()](str(value))  # type: ignore[no-untyped-call]
        elif isinstance(value, dict):
            # Recursive sanitization
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value

    return sanitized


# Module-level logger instance
logger = get_logger()
