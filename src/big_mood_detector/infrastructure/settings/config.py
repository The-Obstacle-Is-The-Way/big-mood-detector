"""
Configuration Settings

Centralized configuration management using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation and environment loading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=True,
        validate_assignment=True,
    )

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Big Mood Detector"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Paths
    # Support both DATA_DIR and BIGMOOD_DATA_DIR for flexibility
    DATA_DIR: Path = Path(
        os.environ.get("BIGMOOD_DATA_DIR", os.environ.get("DATA_DIR", "data"))
    )
    MODEL_WEIGHTS_PATH: Path = Field(
        default_factory=lambda: Path("model_weights/xgboost/converted")
    )
    OUTPUT_DIR: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("BIGMOOD_DATA_DIR", os.environ.get("DATA_DIR", "data"))
        )
        / "output"
    )
    UPLOAD_DIR: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("BIGMOOD_DATA_DIR", os.environ.get("DATA_DIR", "data"))
        )
        / "uploads"
    )
    TEMP_DIR: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("BIGMOOD_DATA_DIR", os.environ.get("DATA_DIR", "data"))
        )
        / "temp"
    )

    # File Processing
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list[str] = [".xml", ".json"]

    # ML Configuration
    CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    USE_PAT_MODEL: bool = False  # Until we implement PAT

    # Ensemble Model Weights
    ENSEMBLE_XGBOOST_WEIGHT: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for XGBoost model in ensemble (0.6 = 60%)",
    )
    ENSEMBLE_PAT_WEIGHT: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for PAT model in ensemble (0.4 = 40%)",
    )

    # Ensemble Timeouts
    ENSEMBLE_PAT_TIMEOUT: float = Field(
        default=10.0, gt=0, description="Timeout for PAT model in seconds"
    )
    ENSEMBLE_XGBOOST_TIMEOUT: float = Field(
        default=5.0, gt=0, description="Timeout for XGBoost model in seconds"
    )

    # Background Tasks
    TASK_TIMEOUT: int = Field(default=300, gt=0)  # 5 minutes
    MAX_RETRIES: int = Field(default=3, ge=0)

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # Clinical Thresholds
    DEPRESSION_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    HYPOMANIC_THRESHOLD: float = Field(default=0.3, ge=0.0, le=1.0)
    MANIC_THRESHOLD: float = Field(default=0.3, ge=0.0, le=1.0)

    # Data Quality Requirements
    MIN_OBSERVATION_DAYS: int = Field(
        default=7,
        ge=1,
        description="Minimum days of data required for feature extraction",
    )
    
    # Privacy
    USER_ID_SALT: str = Field(
        default="big-mood-detector-default-salt",
        description="Salt for user ID hashing (CHANGE IN PRODUCTION!)"
    )
    
    # Feature Store
    FEAST_REPO_PATH: Path | None = Field(
        default=None,
        description="Path to Feast feature repository"
    )
    FEAST_RETRY_BASE: float = Field(
        default=0.5,
        gt=0,
        description="Base retry delay for Feast sync (seconds)"
    )
    FEAST_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for Feast sync"
    )

    @model_validator(mode="after")
    def validate_ensemble_weights(self) -> "Settings":
        """Validate that ensemble weights sum to 1.0."""
        total = self.ENSEMBLE_XGBOOST_WEIGHT + self.ENSEMBLE_PAT_WEIGHT
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(
                f"Ensemble weights must sum to 1.0, got {total:.3f} "
                f"(XGBoost: {self.ENSEMBLE_XGBOOST_WEIGHT}, PAT: {self.ENSEMBLE_PAT_WEIGHT})"
            )
        return self
    
    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate critical settings for production environment."""
        if self.ENVIRONMENT == "production":
            if self.USER_ID_SALT == "big-mood-detector-default-salt":
                import warnings
                warnings.warn(
                    "Using default salt in production! Set USER_ID_SALT to a secure value.",
                    UserWarning,
                    stacklevel=2
                )
        return self

    def ensure_directories(self) -> None:
        """Create necessary directories. Call this after settings are loaded."""
        from .utils import initialize_directories

        initialize_directories(self)

    @computed_field
    def log_config(self) -> dict:
        """Generate logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
                },
                "text": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": self.LOG_FORMAT,
                    "level": self.LOG_LEVEL,
                }
            },
            "root": {"level": self.LOG_LEVEL, "handlers": ["console"]},
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
