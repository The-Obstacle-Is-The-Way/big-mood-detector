"""
Test Configuration Settings

TDD for settings and configuration management.
"""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError


class TestSettings:
    """Test settings configuration."""

    def test_settings_can_be_imported(self):
        """Test that settings module can be imported."""
        from big_mood_detector.core.config import Settings, settings

        assert Settings is not None
        assert settings is not None

    def test_default_settings(self):
        """Test default settings values."""
        from big_mood_detector.core.config import Settings

        settings = Settings()

        # API settings
        assert settings.API_V1_STR == "/api/v1"
        assert settings.PROJECT_NAME == "Big Mood Detector"
        assert settings.VERSION == "0.4.0"
        assert settings.ENVIRONMENT == "local"

        # Paths
        assert settings.MODEL_WEIGHTS_PATH == Path("model_weights/xgboost/converted")
        assert settings.OUTPUT_DIR == Path("data/output")
        assert settings.UPLOAD_DIR == Path("data/uploads")

        # File processing
        assert settings.MAX_FILE_SIZE == 100 * 1024 * 1024  # 100MB
        assert settings.ALLOWED_EXTENSIONS == [".xml", ".json"]

        # ML settings
        assert settings.CONFIDENCE_THRESHOLD == 0.7
        assert settings.USE_PAT_MODEL is False

        # Clinical thresholds
        assert settings.DEPRESSION_THRESHOLD == 0.5
        assert settings.HYPOMANIC_THRESHOLD == 0.3
        assert settings.MANIC_THRESHOLD == 0.3

    def test_settings_from_env(self):
        """Test loading settings from environment variables."""
        from big_mood_detector.core.config import Settings

        # Set environment variables
        os.environ["ENVIRONMENT"] = "production"
        os.environ["LOG_LEVEL"] = "ERROR"
        os.environ["MAX_FILE_SIZE"] = "50000000"
        os.environ["CONFIDENCE_THRESHOLD"] = "0.9"

        try:
            settings = Settings()

            assert settings.ENVIRONMENT == "production"
            assert settings.LOG_LEVEL == "ERROR"
            assert settings.MAX_FILE_SIZE == 50000000
            assert settings.CONFIDENCE_THRESHOLD == 0.9
        finally:
            # Clean up
            for key in [
                "ENVIRONMENT",
                "LOG_LEVEL",
                "MAX_FILE_SIZE",
                "CONFIDENCE_THRESHOLD",
            ]:
                os.environ.pop(key, None)

    def test_path_validation(self):
        """Test that paths are validated and created."""
        from big_mood_detector.core.config import Settings

        with tempfile.TemporaryDirectory() as tmpdir:
            test_output = Path(tmpdir) / "test_output"
            test_upload = Path(tmpdir) / "test_upload"

            # Paths shouldn't exist yet
            assert not test_output.exists()
            assert not test_upload.exists()

            # Create settings with custom paths
            settings = Settings(
                OUTPUT_DIR=test_output,
                UPLOAD_DIR=test_upload,
            )

            # Paths shouldn't be created automatically
            assert not test_output.exists()
            assert not test_upload.exists()

            # Now ensure directories
            from big_mood_detector.infrastructure.settings.utils import (
                initialize_directories,
            )

            initialize_directories(settings)

            # Now paths should exist
            assert test_output.exists()
            assert test_upload.exists()

    def test_environment_validation(self):
        """Test environment value validation."""
        from big_mood_detector.core.config import Settings

        # Valid environments
        for env in ["local", "staging", "production"]:
            settings = Settings(ENVIRONMENT=env)
            assert settings.ENVIRONMENT == env

        # Invalid environment should raise error
        with pytest.raises(ValidationError):
            Settings(ENVIRONMENT="invalid")

    def test_log_level_validation(self):
        """Test log level validation."""
        from big_mood_detector.core.config import Settings

        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            settings = Settings(LOG_LEVEL=level)
            assert settings.LOG_LEVEL == level

        # Invalid log level should raise error
        with pytest.raises(ValidationError):
            Settings(LOG_LEVEL="INVALID")

    def test_log_config_computed_field(self):
        """Test log configuration generation."""
        from big_mood_detector.core.config import Settings

        settings = Settings(LOG_LEVEL="DEBUG", LOG_FORMAT="json")
        log_config = settings.log_config

        assert log_config["version"] == 1
        assert log_config["root"]["level"] == "DEBUG"
        assert log_config["handlers"]["console"]["formatter"] == "json"
        assert "json" in log_config["formatters"]
        assert "text" in log_config["formatters"]

    def test_settings_from_env_file(self):
        """Test loading settings from .env file."""
        from big_mood_detector.core.config import Settings

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("ENVIRONMENT=staging\n")
            f.write("PROJECT_NAME=Test Project\n")
            f.write("LOG_LEVEL=WARNING\n")
            f.write("DEPRESSION_THRESHOLD=0.6\n")
            env_file = f.name

        try:
            settings = Settings(_env_file=env_file)

            assert settings.ENVIRONMENT == "staging"
            assert settings.PROJECT_NAME == "Test Project"
            assert settings.LOG_LEVEL == "WARNING"
            assert settings.DEPRESSION_THRESHOLD == 0.6
        finally:
            os.unlink(env_file)

    def test_settings_singleton_pattern(self):
        """Test that settings can be used as a singleton."""
        from big_mood_detector.core.config import settings

        # Should be the same instance
        from big_mood_detector.core.config import settings as settings2

        assert settings is settings2

    def test_clinical_thresholds_validation(self):
        """Test clinical threshold validation."""
        from big_mood_detector.core.config import Settings

        # Valid thresholds (0-1)
        settings = Settings(
            DEPRESSION_THRESHOLD=0.0,
            HYPOMANIC_THRESHOLD=1.0,
            MANIC_THRESHOLD=0.5,
        )
        assert settings.DEPRESSION_THRESHOLD == 0.0
        assert settings.HYPOMANIC_THRESHOLD == 1.0
        assert settings.MANIC_THRESHOLD == 0.5

        # Invalid thresholds should raise error
        with pytest.raises(ValidationError):
            Settings(DEPRESSION_THRESHOLD=1.5)

        with pytest.raises(ValidationError):
            Settings(MANIC_THRESHOLD=-0.1)
