"""
Test Refactored Clinical Interpreter

Ensures the refactored facade works correctly with extracted services.
"""

import pytest

from big_mood_detector.domain.services.clinical_interpreter import (
    ClinicalInterpreter,
    EpisodeType,
    RiskLevel,
)
from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestClinicalInterpreterRefactored:
    """Test the refactored clinical interpreter facade."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path

        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def interpreter(self, config):
        """Create refactored interpreter."""
        return ClinicalInterpreter(config)

    def test_interpret_depression_delegates_correctly(self, interpreter):
        """Test that depression interpretation delegates to services."""
        result = interpreter.interpret_depression_score(
            phq_score=12.0,
            sleep_hours=7.5,
            activity_steps=6000,
        )

        assert result.risk_level == RiskLevel.MODERATE
        assert result.episode_type == EpisodeType.DEPRESSIVE
        assert result.dsm5_criteria_met is True
        assert len(result.recommendations) > 0
        assert "quetiapine" in [r.medication for r in result.recommendations]

    def test_interpret_mania_delegates_correctly(self, interpreter):
        """Test that mania interpretation delegates to services."""
        result = interpreter.interpret_mania_score(
            asrm_score=8.0,
            sleep_hours=5.0,
            activity_steps=12000,
        )

        assert result.risk_level == RiskLevel.MODERATE
        assert result.episode_type == EpisodeType.HYPOMANIC
        assert result.dsm5_criteria_met is True
        assert len(result.recommendations) > 0

    def test_biomarker_interpretation_delegates(self, interpreter):
        """Test biomarker interpretation delegation."""
        # Test sleep biomarkers
        sleep_result = interpreter.interpret_sleep_biomarkers(
            sleep_duration=2.5,
            sleep_efficiency=0.70,
            sleep_timing_variance=3.0,
        )

        assert sleep_result.mania_risk_factors == 3
        assert sleep_result.recommendation_priority == "urgent"

        # Test activity biomarkers
        activity_result = interpreter.interpret_activity_biomarkers(
            daily_steps=18000,
            sedentary_hours=6.0,
            activity_variance=0.5,
        )

        assert activity_result.mania_risk_factors > 0

    def test_treatment_recommendations_delegate(self, interpreter):
        """Test treatment recommendation delegation."""
        recs = interpreter.get_treatment_recommendations(
            episode_type=EpisodeType.MANIC,
            severity=RiskLevel.HIGH,
            current_medications=[],
        )

        assert len(recs) > 0
        med_names = [r.medication for r in recs]
        assert any(med in med_names for med in ["lithium", "quetiapine"])

    def test_clinical_rules_delegate(self, interpreter):
        """Test clinical rules delegation."""
        result = interpreter.apply_clinical_rules(
            diagnosis="bipolar_disorder",
            proposed_treatment="sertraline",
            current_medications=[],
            mood_state="depressed",
        )

        assert result.approved is False
        assert "contraindicated" in result.rationale

    def test_backward_compatibility(self, interpreter):
        """Test that the refactored version maintains backward compatibility."""
        # Test that all the expected attributes exist
        assert hasattr(interpreter, "interpret_depression_score")
        assert hasattr(interpreter, "interpret_mania_score")
        assert hasattr(interpreter, "interpret_sleep_biomarkers")
        assert hasattr(interpreter, "interpret_activity_biomarkers")
        assert hasattr(interpreter, "interpret_circadian_biomarkers")
        assert hasattr(interpreter, "get_treatment_recommendations")
        assert hasattr(interpreter, "apply_clinical_rules")
