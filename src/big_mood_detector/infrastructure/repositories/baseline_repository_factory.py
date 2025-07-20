"""
Baseline Repository Factory

Provides a factory for creating baseline repositories based on configuration.
Supports both file-based and TimescaleDB repositories.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from structlog import get_logger

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
)
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)
from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
    TimescaleBaselineRepository,
)

logger = get_logger()


@dataclass
class BaselineRepositorySettings:
    """Settings for baseline repository configuration."""

    repository_type: str = "file"  # "file" or "timescale"
    base_path: Path = Path("data/baselines")
    timescale_connection_string: str | None = None
    enable_feast_sync: bool = False
    feast_repo_path: Path | None = None


class BaselineRepositoryFactory:
    """
    Factory for creating baseline repositories.

    Supports:
    - File-based repository (default, no dependencies)
    - TimescaleDB repository (for production scalability)

    Configuration via:
    - Environment variables (BASELINE_REPOSITORY_TYPE)
    - Explicit parameters
    - Settings object
    """

    def __init__(
        self,
        base_path: Path | None = None,
        default_type: str = "file",
    ):
        """
        Initialize repository factory.

        Args:
            base_path: Base path for file repositories
            default_type: Default repository type if not specified
        """
        self.base_path = base_path or Path("data/baselines")
        self.default_type = default_type
        self._singleton_instance: BaselineRepositoryInterface | None = None

        logger.info(
            "baseline_repository_factory_initialized",
            base_path=str(self.base_path),
            default_type=default_type,
        )

    @classmethod
    def from_settings(cls, settings: BaselineRepositorySettings) -> "BaselineRepositoryFactory":
        """Create factory from settings object."""
        return cls(
            base_path=settings.base_path,
            default_type=settings.repository_type,
        )

    def create_repository(
        self,
        repository_type: str | None = None,
    ) -> BaselineRepositoryInterface:
        """
        Create a new baseline repository instance.

        Args:
            repository_type: Type of repository ("file" or "timescale")
                           If None, uses environment variable or default

        Returns:
            BaselineRepositoryInterface implementation

        Raises:
            ValueError: If repository type is unknown or required config missing
        """
        # Determine repository type
        repo_type = repository_type or os.getenv("BASELINE_REPOSITORY_TYPE", self.default_type)

        logger.info("creating_baseline_repository", repository_type=repo_type)

        if repo_type == "file":
            return self._create_file_repository()
        elif repo_type == "timescale":
            return self._create_timescale_repository()
        else:
            raise ValueError(f"Unknown repository type: {repo_type}")

    def get_repository(self) -> BaselineRepositoryInterface:
        """
        Get singleton instance of baseline repository.

        Returns the same instance on repeated calls.
        """
        if self._singleton_instance is None:
            self._singleton_instance = self.create_repository()
        return self._singleton_instance

    def _create_file_repository(self) -> FileBaselineRepository:
        """Create file-based repository."""
        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info("creating_file_baseline_repository", path=str(self.base_path))
        return FileBaselineRepository(self.base_path)

    def _create_timescale_repository(self) -> TimescaleBaselineRepository:
        """Create TimescaleDB repository."""
        # Get connection string from environment
        connection_string = os.getenv("TIMESCALE_CONNECTION_STRING")

        if not connection_string:
            raise ValueError(
                "TIMESCALE_CONNECTION_STRING environment variable required for TimescaleDB repository"
            )

        # Get optional Feast configuration
        enable_feast = os.getenv("ENABLE_FEAST_SYNC", "false").lower() == "true"
        feast_repo_path = os.getenv("FEAST_REPO_PATH")

        logger.info(
            "creating_timescale_baseline_repository",
            connection_string=connection_string.split("@")[-1],  # Log only host part
            enable_feast=enable_feast,
        )

        return TimescaleBaselineRepository(
            connection_string=connection_string,
            enable_feast_sync=enable_feast,
            feast_repo_path=Path(feast_repo_path) if feast_repo_path else None,
        )
