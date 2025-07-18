"""
Backward compatibility module for config.

This module has been moved to infrastructure.settings.
This file provides compatibility imports during the transition.
"""

import warnings

warnings.warn(
    "Importing from big_mood_detector.core.config is deprecated. "
    "Use big_mood_detector.infrastructure.settings instead.",
    DeprecationWarning,
    stacklevel=2,
)

from big_mood_detector.infrastructure.settings import (  # noqa: E402
    Settings,
    get_settings,
)
from big_mood_detector.infrastructure.settings.config import settings  # noqa: E402

__all__ = ["Settings", "get_settings", "settings"]
