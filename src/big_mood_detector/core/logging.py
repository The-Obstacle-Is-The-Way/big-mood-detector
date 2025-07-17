"""
Backward compatibility module for logging.

This module has been moved to infrastructure.logging.
This file provides compatibility imports during the transition.
"""

import warnings

warnings.warn(
    "Importing from big_mood_detector.core.logging is deprecated. "
    "Use big_mood_detector.infrastructure.logging instead.",
    DeprecationWarning,
    stacklevel=2,
)

from big_mood_detector.infrastructure.logging import (  # noqa: E402
    get_logger,
    get_module_logger,
)

__all__ = ["get_logger", "get_module_logger"]
