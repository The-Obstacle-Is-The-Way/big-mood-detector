"""
Configuration Settings

Centralized configuration management using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
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

    # Paths
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data"))
    MODEL_WEIGHTS_PATH: Path = Field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data")) / "model_weights/xgboost/converted")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data")) / "output")
    UPLOAD_DIR: Path = Field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data")) / "uploads")
    TEMP_DIR: Path = Field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data")) / "temp")

    # File Processing
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list[str] = [".xml", ".json"]

    # ML Configuration
    CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    USE_PAT_MODEL: bool = False  # Until we implement PAT

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
