"""
Test Biomarker Interpreter Service

Tests for interpreting digital biomarkers (sleep, activity, circadian).
Following TDD - Red, Green, Refactor.
"""

import pytest

from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestBiomarkerInterpreter:
    """Test biomarker interpretation service."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path
        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def interpreter(self, config):
        """Create biomarker interpreter with test config."""
        from big_mood_detector.domain.services.biomarker_interpreter import BiomarkerInterpreter
        return BiomarkerInterpreter(config)

    def test_interpret_sleep_critical_short(self, interpreter):
        """Test critical short sleep detection."""
        result = interpreter.interpret_sleep(
            sleep_duration=2.5,  # Critical < 3 hours
            sleep_efficiency=0.90,
            sleep_timing_variance=1.0,
        )
        
        assert result.mania_risk_factors > 0
        assert result.recommendation_priority == "urgent"
        assert "critical short sleep" in result.clinical_notes[0].lower()

    def test_interpret_sleep_poor_efficiency(self, interpreter, config):
        """Test poor sleep efficiency detection."""
        result = interpreter.interpret_sleep(
            sleep_duration=7.0,
            sleep_efficiency=0.75,  # Below 0.85 threshold
            sleep_timing_variance=1.0,
        )
        
        assert result.mania_risk_factors > 0
        assert "poor sleep efficiency" in result.clinical_notes[0].lower()

    def test_interpret_sleep_variable_timing(self, interpreter, config):
        """Test variable sleep timing detection."""
        result = interpreter.interpret_sleep(
            sleep_duration=7.0,
            sleep_efficiency=0.90,
            sleep_timing_variance=3.0,  # > 2 hour variance
        )
        
        assert result.mania_risk_factors > 0
        assert "variable sleep schedule" in result.clinical_notes[0].lower()

    def test_interpret_activity_elevated(self, interpreter, config):
        """Test elevated activity detection."""
        result = interpreter.interpret_activity(
            daily_steps=18000,  # > 15000 threshold
            sedentary_hours=8.0,
        )
        
        assert result.mania_risk_factors > 0
        assert "elevated activity" in result.clinical_notes[0].lower()

    def test_interpret_activity_extreme(self, interpreter, config):
        """Test extreme activity detection."""
        result = interpreter.interpret_activity(
            daily_steps=25000,  # > 20000 threshold
            sedentary_hours=3.0,  # < 4 hours
        )
        
        assert result.mania_risk_factors >= 2  # Both triggers
        assert any("extreme" in note.lower() for note in result.clinical_notes)
        assert any("minimal rest" in note.lower() for note in result.clinical_notes)

    def test_interpret_circadian_phase_advance(self, interpreter, config):
        """Test circadian phase advance detection."""
        result = interpreter.interpret_circadian(
            phase_advance=2.5,  # > 2 hours
            interdaily_stability=0.8,
            intradaily_variability=0.5,
        )
        
        assert result.mania_risk_factors > 0
        assert "phase advance" in result.clinical_notes[0].lower()

    def test_interpret_circadian_instability(self, interpreter, config):
        """Test circadian rhythm instability detection."""
        result = interpreter.interpret_circadian(
            phase_advance=1.0,
            interdaily_stability=0.4,  # < 0.5
            intradaily_variability=1.2,  # > 1.0
        )
        
        assert result.mood_instability_risk == "high"
        assert any("stability" in note.lower() for note in result.clinical_notes)
        assert any("fragmentation" in note.lower() for note in result.clinical_notes)

    def test_combined_risk_factors(self, interpreter):
        """Test that multiple risk factors accumulate."""
        # Test sleep with multiple issues
        result = interpreter.interpret_sleep(
            sleep_duration=2.5,  # Critical
            sleep_efficiency=0.70,  # Poor
            sleep_timing_variance=3.0,  # Variable
        )
        
        assert result.mania_risk_factors == 3
        assert len(result.clinical_notes) == 3
        assert result.recommendation_priority == "urgent"

    def test_uses_configuration_thresholds(self, interpreter, config):
        """Test that interpreter uses injected configuration."""
        # Test at exact threshold boundaries
        efficiency_threshold = config.biomarkers.sleep.efficiency_threshold
        
        # Just below threshold
        result = interpreter.interpret_sleep(
            sleep_duration=7.0,
            sleep_efficiency=efficiency_threshold - 0.01,
            sleep_timing_variance=1.0,
        )
        assert result.mania_risk_factors == 1
        
        # Just above threshold
        result = interpreter.interpret_sleep(
            sleep_duration=7.0,
            sleep_efficiency=efficiency_threshold + 0.01,
            sleep_timing_variance=1.0,
        )
        assert result.mania_risk_factors == 0