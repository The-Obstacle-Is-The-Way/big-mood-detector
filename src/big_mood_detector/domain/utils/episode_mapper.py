"""
Episode Mapper

Utility class for mapping between different episode type representations.
Centralizes the mapping logic to avoid duplication across services.
"""



class EpisodeMapper:
    """Maps between different episode type representations."""

    # Mapping from clinical diagnosis to DSM-5 episode types
    _TO_DSM5_MAP = {
        "depressive_episode": "depressive",
        "manic_episode": "manic",
        "hypomanic_episode": "hypomanic",
        "mixed_episode": "mixed",
        "euthymic": "euthymic",
    }

    # Reverse mapping
    _FROM_DSM5_MAP = {v: k for k, v in _TO_DSM5_MAP.items()}

    # Mapping from clinical diagnosis to treatment episode types
    _TO_TREATMENT_MAP = {
        "depressive_episode": "depressive",
        "manic_episode": "manic",
        "hypomanic_episode": "hypomanic",
        "mixed_episode": "mixed",
        "depressive_with_mixed_features": "depressive_mixed",
        "manic_with_mixed_features": "manic_mixed",
        "euthymic": "none",
    }

    @classmethod
    def to_dsm5_episode_type(cls, diagnosis: str) -> str:
        """
        Convert clinical diagnosis to DSM-5 episode type.

        Args:
            diagnosis: Clinical diagnosis (e.g., "depressive_episode")

        Returns:
            DSM-5 episode type (e.g., "depressive")
        """
        # Normalize to lowercase for case-insensitive mapping
        diagnosis_lower = diagnosis.lower() if diagnosis else ""
        return cls._TO_DSM5_MAP.get(diagnosis_lower, diagnosis_lower)

    @classmethod
    def from_dsm5_episode_type(cls, dsm5_type: str) -> str:
        """
        Convert DSM-5 episode type back to clinical diagnosis.

        Args:
            dsm5_type: DSM-5 episode type (e.g., "depressive")

        Returns:
            Clinical diagnosis (e.g., "depressive_episode")
        """
        dsm5_lower = dsm5_type.lower() if dsm5_type else ""
        return cls._FROM_DSM5_MAP.get(dsm5_lower, dsm5_lower)

    @classmethod
    def to_treatment_episode_type(cls, diagnosis: str) -> str:
        """
        Convert clinical diagnosis to treatment episode type.

        Args:
            diagnosis: Clinical diagnosis (e.g., "depressive_episode")

        Returns:
            Treatment episode type (e.g., "depressive")
        """
        diagnosis_lower = diagnosis.lower() if diagnosis else ""
        return cls._TO_TREATMENT_MAP.get(diagnosis_lower, diagnosis_lower)

    @classmethod
    def get_all_episode_types(cls) -> set[str]:
        """
        Get all known episode types.

        Returns:
            Set of all known episode type strings
        """
        all_types: set[str] = set()
        all_types.update(cls._TO_DSM5_MAP.keys())
        all_types.update(cls._TO_TREATMENT_MAP.keys())
        return all_types

    @classmethod
    def is_valid_episode_type(cls, episode_type: str | None) -> bool:
        """
        Check if an episode type is valid.

        Args:
            episode_type: Episode type to validate

        Returns:
            True if valid, False otherwise
        """
        if not episode_type:
            return False

        episode_lower = episode_type.lower()
        return episode_lower in cls.get_all_episode_types()
