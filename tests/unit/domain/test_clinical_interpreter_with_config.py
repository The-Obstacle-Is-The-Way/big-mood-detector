"""
Test Clinical Interpreter with Configuration

Tests that the ClinicalInterpreter properly uses the external configuration
instead of hard-coded values.
"""

from pathlib import Path

import pytest

from big_mood_detector.domain.services.clinical_interpreter import (
    ClinicalInterpreter,
    RiskLevel,
    EpisodeType,
)
from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestClinicalInterpreterWithConfig:
    """Test clinical interpreter using configuration file."""

    @pytest.fixture
    def config(self):
        """Load clinical thresholds from configuration."""
        config_path = Path("config/clinical_thresholds.yaml")
        return load_clinical_thresholds(config_path)

    @pytest.fixture
    def interpreter(self, config):
        """Create interpreter with configuration."""
        return ClinicalInterpreter(config)

    def test_interpreter_uses_config_for_depression(self, interpreter, config):
        """Test that interpreter uses configuration values for depression."""
        # Test at the configured moderate threshold (PHQ=10)
        result = interpreter.interpret_depression_score(
            phq_score=config.depression.phq_cutoffs.moderate.min,
            sleep_hours=7.5,
            activity_steps=6000,
        )
        assert result.risk_level == RiskLevel.MODERATE
        assert result.episode_type == EpisodeType.DEPRESSIVE

        # Test just below moderate threshold (PHQ=9)
        result = interpreter.interpret_depression_score(
            phq_score=config.depression.phq_cutoffs.moderate.min - 1,
            sleep_hours=7.5,
            activity_steps=6000,
        )
        assert result.risk_level == RiskLevel.LOW

    def test_interpreter_uses_config_for_mania(self, interpreter, config):
        """Test that interpreter uses configuration values for mania."""
        # Test at the configured hypomanic threshold (ASRM=6)
        result = interpreter.interpret_mania_score(
            asrm_score=config.mania.asrm_cutoffs.hypomanic.min,
            sleep_hours=7,
            activity_steps=8000,
        )
        assert result.risk_level == RiskLevel.MODERATE
        assert result.episode_type == EpisodeType.HYPOMANIC

        # Test critical sleep threshold
        result = interpreter.interpret_mania_score(
            asrm_score=5,  # Below ASRM threshold
            sleep_hours=config.mania.sleep_hours.critical_threshold - 0.5,  # 2.5 hours
            activity_steps=8000,
        )
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.episode_type == EpisodeType.MANIC

    def test_interpreter_uses_biomarker_config(self, interpreter, config):
        """Test that interpreter uses biomarker configuration."""
        # Test sleep efficiency threshold
        result = interpreter.interpret_sleep_biomarkers(
            sleep_duration=7,
            sleep_efficiency=config.biomarkers.sleep.efficiency_threshold - 0.1,  # 0.75
            sleep_timing_variance=1.0,
        )
        assert result.mania_risk_factors > 0
        assert "Poor sleep efficiency" in result.clinical_notes[0]

        # Test circadian phase advance
        result = interpreter.interpret_circadian_biomarkers(
            circadian_phase_advance=config.biomarkers.circadian.phase_advance_threshold + 0.5,  # 2.5
            interdaily_stability=0.8,
            intradaily_variability=0.5,
        )
        assert result.mania_risk_factors > 0
        assert "circadian phase advance" in result.clinical_notes[0].lower()

    def test_interpreter_uses_dsm5_duration_config(self, interpreter, config):
        """Test that interpreter uses DSM-5 duration configuration."""
        # Test manic episode duration
        result = interpreter.evaluate_episode_duration(
            episode_type=EpisodeType.MANIC,
            symptom_days=config.dsm5_duration.manic_days,  # 7 days
            hospitalization=False,
        )
        assert result.meets_dsm5_criteria

        # Test just below threshold
        result = interpreter.evaluate_episode_duration(
            episode_type=EpisodeType.MANIC,
            symptom_days=config.dsm5_duration.manic_days - 1,  # 6 days
            hospitalization=False,
        )
        assert not result.meets_dsm5_criteria

    def test_interpreter_uses_mixed_features_config(self, interpreter, config):
        """Test that interpreter uses mixed features configuration."""
        # Count required symptoms from config
        manic_symptoms = len(config.mixed_features.depression_with_mixed.required_manic_symptoms)
        
        # Should detect mixed features with minimum symptoms
        result = interpreter.interpret_mixed_state(
            phq_score=15,  # Depression
            asrm_score=4,   # Below mania threshold
            sleep_hours=4,
            activity_steps=15000,
            racing_thoughts=True,
            increased_energy=True,
            # This gives us 3 manic symptoms total (racing thoughts, increased energy, reduced sleep)
        )
        assert result.episode_type == EpisodeType.DEPRESSIVE_MIXED
        assert "mixed features" in result.clinical_summary.lower()

    def test_configuration_validation(self):
        """Test that configuration is valid and complete."""
        config = load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))
        
        # Verify threshold ordering
        assert config.depression.phq_cutoffs.none.max < config.depression.phq_cutoffs.mild.min
        assert config.depression.phq_cutoffs.mild.max < config.depression.phq_cutoffs.moderate.min
        assert config.depression.phq_cutoffs.moderate.max < config.depression.phq_cutoffs.moderately_severe.min
        assert config.depression.phq_cutoffs.moderately_severe.max < config.depression.phq_cutoffs.severe.min
        
        # Verify mania thresholds
        assert config.mania.asrm_cutoffs.none.max < config.mania.asrm_cutoffs.hypomanic.min
        assert config.mania.asrm_cutoffs.hypomanic.max < config.mania.asrm_cutoffs.manic_moderate.min
        assert config.mania.asrm_cutoffs.manic_moderate.max < config.mania.asrm_cutoffs.manic_severe.min
        
        # Verify critical thresholds
        assert config.mania.sleep_hours.critical_threshold < config.mania.sleep_hours.reduced_threshold
        assert config.depression.activity_steps.severe_reduction < config.depression.activity_steps.moderate_reduction