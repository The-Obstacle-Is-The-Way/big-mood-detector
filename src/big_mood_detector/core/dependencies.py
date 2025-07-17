"""
Backward compatibility module for dependencies.

This module has been moved to infrastructure.di.
This file provides compatibility imports during the transition.
"""

import warnings

warnings.warn(
    "Importing from big_mood_detector.core.dependencies is deprecated. "
    "Use big_mood_detector.infrastructure.di instead.",
    DeprecationWarning,
    stacklevel=2,
)

from big_mood_detector.infrastructure.di import (  # noqa: E402
    Container,
    DependencyNotFoundError,
    Provide,
    get_container,
    inject,
    setup_dependencies,
)

__all__ = [
    "Container",
    "DependencyNotFoundError",
    "Provide",
    "get_container",
    "inject",
    "setup_dependencies",
]
