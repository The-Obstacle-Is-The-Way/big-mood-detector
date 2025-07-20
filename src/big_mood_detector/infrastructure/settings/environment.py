"""
Environment Configuration with Pydantic

Production-grade settings management with:
- Type validation
- Environment variable support
- Default values
- Documentation
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables with
    the BIGMOOD_ prefix. For example:
    - BIGMOOD_LOG_LEVEL=DEBUG
    - BIGMOOD_DATABASE_URL=postgresql://...
    """
    
    # Application
    app_name: str = Field(default="big-mood-detector", description="Application name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", 
        description="Deployment environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Database
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL connection string for TimescaleDB"
    )
    
    # Privacy
    user_id_salt: str = Field(
        default="big-mood-detector-default-salt",
        description="Salt for user ID hashing (CHANGE IN PRODUCTION!)"
    )
    
    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Base directory for data storage"
    )
    model_dir: Path = Field(
        default=Path("model_weights"),
        description="Directory containing ML model weights"
    )
    
    # Feature Store
    feast_repo_path: Path | None = Field(
        default=None,
        description="Path to Feast feature repository"
    )
    feast_retry_base: float = Field(
        default=0.5,
        description="Base retry delay for Feast sync (seconds)"
    )
    feast_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for Feast sync"
    )
    
    # ML Models
    ensemble_enabled: bool = Field(
        default=False,
        description="Enable PAT + XGBoost ensemble"
    )
    pat_model_path: Path | None = Field(
        default=None,
        description="Path to PAT model weights"
    )
    
    # Performance
    max_workers: int = Field(
        default=4,
        description="Maximum worker threads for concurrent operations"
    )
    batch_size: int = Field(
        default=1000,
        description="Batch size for data processing"
    )
    
    # Monitoring
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9090,
        description="Port for metrics endpoint"
    )
    
    @validator("data_dir", "model_dir", pre=True)
    def expand_path(cls, v: str | Path) -> Path:
        """Expand paths and resolve home directory."""
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser().resolve()
    
    @validator("database_url", pre=True)
    def validate_database_url(cls, v: str | None) -> str | None:
        """Validate database URL format."""
        if v and not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("Database URL must start with postgresql:// or postgres://")
        return v
    
    @validator("user_id_salt")
    def warn_default_salt(cls, v: str, values: dict) -> str:
        """Warn if using default salt in production."""
        if values.get("environment") == "production" and v == "big-mood-detector-default-salt":
            import warnings
            warnings.warn(
                "Using default salt in production! Set BIGMOOD_USER_ID_SALT to a secure value.",
                UserWarning
            )
        return v
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "output").mkdir(exist_ok=True)
        (self.data_dir / "baselines").mkdir(exist_ok=True)
        (self.data_dir / "labels").mkdir(exist_ok=True)
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "BIGMOOD_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None