"""Test configuration settings."""

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

class TestSettings:
    """Test Settings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        from big_mood_detector.infrastructure.settings.config import Settings

        settings = Settings()

        assert settings.PROJECT_NAME == "Big Mood Detector"
        assert settings.ENVIRONMENT == "local"
        assert settings.LOG_LEVEL == "INFO"
        assert settings.USER_ID_SALT == "big-mood-detector-default-salt"
        assert settings.FEAST_RETRY_BASE == 0.5
        assert settings.FEAST_MAX_RETRIES == 3

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        from big_mood_detector.infrastructure.settings.config import Settings

        with patch.dict(os.environ, {
            "LOG_LEVEL": "DEBUG",
            "USER_ID_SALT": "custom-salt-123",
            "FEAST_RETRY_BASE": "1.0",
            "FEAST_MAX_RETRIES": "5"
        }):
            settings = Settings()

            assert settings.LOG_LEVEL == "DEBUG"
            assert settings.USER_ID_SALT == "custom-salt-123"
            assert settings.FEAST_RETRY_BASE == 1.0
            assert settings.FEAST_MAX_RETRIES == 5

    def test_production_salt_warning(self):
        """Test warning when using default salt in production."""
        from big_mood_detector.infrastructure.settings.config import Settings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            Settings(ENVIRONMENT="production")

            # Should trigger warning
            assert len(w) == 1
            assert "Using default salt in production" in str(w[0].message)

    def test_ensemble_weights_validation(self):
        """Test that ensemble weights must sum to 1.0."""
        from big_mood_detector.infrastructure.settings.config import Settings

        # Valid weights
        settings = Settings(ENSEMBLE_XGBOOST_WEIGHT=0.7, ENSEMBLE_PAT_WEIGHT=0.3)
        assert settings.ENSEMBLE_XGBOOST_WEIGHT == 0.7
        assert settings.ENSEMBLE_PAT_WEIGHT == 0.3

        # Invalid weights
        with pytest.raises(ValidationError, match="Ensemble weights must sum to 1.0"):
            Settings(ENSEMBLE_XGBOOST_WEIGHT=0.6, ENSEMBLE_PAT_WEIGHT=0.3)

    def test_path_expansion(self):
        """Test that paths are expanded correctly."""
        from big_mood_detector.infrastructure.settings.config import Settings

        settings = Settings(DATA_DIR="~/test_data")

        # Should expand home directory
        assert str(settings.DATA_DIR).startswith(str(Path.home()))
        assert settings.DATA_DIR.is_absolute()

    def test_validation_constraints(self):
        """Test field validation constraints."""
        from big_mood_detector.infrastructure.settings.config import Settings

        # Valid values
        settings = Settings(
            DEPRESSION_THRESHOLD=0.7,
            MIN_OBSERVATION_DAYS=14,
            FEAST_RETRY_BASE=2.0
        )
        assert settings.DEPRESSION_THRESHOLD == 0.7
        assert settings.MIN_OBSERVATION_DAYS == 14
        assert settings.FEAST_RETRY_BASE == 2.0

        # Invalid threshold (must be between 0 and 1)
        with pytest.raises(ValidationError):
            Settings(DEPRESSION_THRESHOLD=1.5)

        # Invalid observation days (must be >= 1)
        with pytest.raises(ValidationError):
            Settings(MIN_OBSERVATION_DAYS=0)

        # Invalid retry base (must be > 0)
        with pytest.raises(ValidationError):
            Settings(FEAST_RETRY_BASE=0)

    def test_log_level_validation(self):
        """Test that only valid log levels are accepted."""
        from big_mood_detector.infrastructure.settings.config import Settings

        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            settings = Settings(LOG_LEVEL=level)
            assert settings.LOG_LEVEL == level

        # Invalid level
        with pytest.raises(ValidationError):
            Settings(LOG_LEVEL="TRACE")

    def test_environment_validation(self):
        """Test that only valid environments are accepted."""
        from big_mood_detector.infrastructure.settings.config import Settings

        # Valid environments
        for env in ["local", "staging", "production"]:
            settings = Settings(ENVIRONMENT=env)
            assert settings.ENVIRONMENT == env

        # Invalid environment
        with pytest.raises(ValidationError):
            Settings(ENVIRONMENT="development")

    def test_get_settings_singleton(self):
        """Test that get_settings returns a singleton."""
        from big_mood_detector.infrastructure.settings.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance
        assert settings1 is settings2

        # Modifications should be visible
        settings1.LOG_LEVEL = "DEBUG"
        assert settings2.LOG_LEVEL == "DEBUG"
