"""Type stubs for Feast integration.

This module provides type hints for the Feast feature store when it's
installed as an optional dependency.
"""

from typing import Any, Protocol


class FeatureStoreProtocol(Protocol):
    """Protocol for Feast FeatureStore to avoid direct import."""
    
    def materialize(self, end_date: Any, feature_views: list[str] | None = None) -> None:
        """Materialize feature views up to end_date."""
        ...
    
    def push(self, feature_view_name: str, df: Any, to: Any | None = None) -> None:
        """Push features to online store."""
        ...
    
    def get_online_features(
        self, 
        features: list[str], 
        entity_rows: list[dict[str, Any]]
    ) -> Any:
        """Get features from online store."""
        ...