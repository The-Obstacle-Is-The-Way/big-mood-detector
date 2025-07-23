"""
Application Adapters

Adapters that bridge between different architectural layers and components.
Following the Adapter pattern from Clean Architecture.
"""

from big_mood_detector.application.adapters.orchestrator_adapter import (
    OrchestratorAdapter,
)

__all__ = ["OrchestratorAdapter"]
