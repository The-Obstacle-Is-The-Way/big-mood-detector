"""
Backward compatibility module for exceptions.

This module has been moved to infrastructure.exceptions.
This file provides compatibility imports during the transition.
"""

import warnings

warnings.warn(
    "Importing from big_mood_detector.core.exceptions is deprecated. "
    "Use big_mood_detector.infrastructure.exceptions instead.",
    DeprecationWarning,
    stacklevel=2,
)

from big_mood_detector.infrastructure.exceptions import *  # noqa: F403, E402
