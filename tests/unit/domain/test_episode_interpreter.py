"""
Test Episode Interpreter Service

Tests for the separated episode interpretation logic.
Following TDD - writing tests first.
"""

import pytest

from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestEpisodeInterpreter:
    """Test episode interpretation service."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path

        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def interpreter(self, config):
        """Create episode interpreter with test config."""
        from big_mood_detector.domain.services.episode_interpreter import (
            EpisodeInterpreter,
        )

        return EpisodeInterpreter(config)

    def test_interpret_depression_normal_range(self, interpreter):
        """Test depression interpretation for normal PHQ scores."""
        result = interpreter.interpret_depression(
            phq_score=3.0,
            sleep_hours=7.5,
            activity_steps=6000,
        )

        assert result.risk_level == "low"
        assert result.episode_type == "none"
        assert result.dsm5_criteria_met is False
        assert "normal range" in result.clinical_summary.lower()

    def test_interpret_depression_moderate(self, interpreter):
        """Test depression interpretation for moderate PHQ scores."""
        result = interpreter.interpret_depression(
            phq_score=12.0,  # Moderate range
            sleep_hours=7.5,
            activity_steps=6000,
        )

        assert result.risk_level == "moderate"
        assert result.episode_type == "depressive"
        assert result.dsm5_criteria_met is True
        assert "moderate depression" in result.clinical_summary.lower()

    def test_interpret_depression_with_suicidal_ideation(self, interpreter):
        """Test that suicidal ideation triggers critical risk."""
        result = interpreter.interpret_depression(
            phq_score=8.0,  # Below moderate
            sleep_hours=7.5,
            activity_steps=6000,
            suicidal_ideation=True,
        )

        assert result.risk_level == "critical"
        assert "urgent assessment needed" in result.clinical_summary.lower()

    def test_interpret_mania_normal_range(self, interpreter):
        """Test mania interpretation for normal ASRM scores."""
        result = interpreter.interpret_mania(
            asrm_score=4.0,
            sleep_hours=7.0,
            activity_steps=8000,
        )

        assert result.risk_level == "low"
        assert result.episode_type == "none"
        assert result.dsm5_criteria_met is False

    def test_interpret_mania_critical_sleep(self, interpreter):
        """Test that critical sleep reduction triggers mania."""
        result = interpreter.interpret_mania(
            asrm_score=4.0,  # Below threshold
            sleep_hours=2.5,  # Critical
            activity_steps=8000,
        )

        assert result.risk_level == "critical"
        assert result.episode_type == "manic"
        assert "critical sleep reduction" in result.clinical_summary.lower()

    def test_interpret_mixed_state_detection(self, interpreter):
        """Test mixed state detection with opposite pole symptoms."""
        result = interpreter.interpret_mixed_state(
            phq_score=15.0,  # Depression
            asrm_score=3.0,  # Below mania threshold
            sleep_hours=4.0,
            activity_steps=15000,
            # Manic symptoms during depression
            racing_thoughts=True,
            increased_energy=True,
            decreased_sleep=True,
        )

        assert result.episode_type == "depressive_with_mixed_features"
        assert result.risk_level == "high"
        assert "mixed features" in result.clinical_summary.lower()

    def test_uses_configuration_thresholds(self, interpreter, config):
        """Test that interpreter uses injected configuration."""
        # Test at exact threshold boundary
        moderate_threshold = config.depression.phq_cutoffs.moderate.min

        result = interpreter.interpret_depression(
            phq_score=moderate_threshold,
            sleep_hours=7.5,
            activity_steps=6000,
        )

        assert result.risk_level == "moderate"
        assert result.dsm5_criteria_met is True
