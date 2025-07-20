"""
Test Clinical Interpreter Migration

Tests to ensure clinical_interpreter has all functionality from clinical_decision_engine.
These tests will be kept until formal release as regression guard.
"""

from datetime import datetime, timedelta

import pytest

from big_mood_detector.domain.services.clinical_interpreter import (
    ClinicalInterpreter,
)


class TestClinicalInterpreterMigration:
    """Test migration of clinical_decision_engine functionality to clinical_interpreter."""

    @pytest.fixture
    def interpreter(self):
        """Create clinical interpreter instance."""
        return ClinicalInterpreter()

    def test_make_clinical_assessment_exists(self, interpreter):
        """Test that make_clinical_assessment method exists."""
        assert hasattr(interpreter, "make_clinical_assessment")

    def test_make_clinical_assessment_depression(self, interpreter):
        """Test comprehensive depression assessment (from clinical_decision_engine)."""
        # Provide complete context to ensure DSM-5 criteria are met
        assessment = interpreter.make_clinical_assessment(
            mood_scores={"phq": 15.0, "asrm": 2.0},  # Clear depression, no mania
            biomarkers={
                "sleep_hours": 10.5,
                "activity_steps": 1500,
            },
            clinical_context={
                "symptom_days": 16,  # > 14 days for depression
                "hospitalization": False,
                "suicidal_ideation": False,
                "functional_impairment": True,  # Required for DSM-5
                "current_medications": ["sertraline"],
                "contraindications": [],
            },
        )

        assert assessment.primary_diagnosis == "depressive_episode"
        assert assessment.risk_level == "high"
        # Strict assertion - DSM-5 criteria MUST be met with proper context
        assert assessment.meets_dsm5_criteria is True, (
            f"DSM-5 criteria not met. Got: {assessment.meets_dsm5_criteria}. "
            f"Clinical summary: {assessment.clinical_summary}"
        )
        assert (
            len(assessment.treatment_options) > 0
        ), "Should have treatment recommendations"
        assert assessment.confidence > 0.7

    def test_make_longitudinal_assessment_exists(self, interpreter):
        """Test that make_longitudinal_assessment method exists."""
        assert hasattr(interpreter, "make_longitudinal_assessment")

    def test_make_longitudinal_assessment(self, interpreter):
        """Test assessment with historical data (from clinical_decision_engine)."""
        historical_data = [
            {
                "date": datetime.now() - timedelta(days=30),
                "phq_score": 8.0,
                "asrm_score": 3.0,
                "risk_level": "moderate",
            },
            {
                "date": datetime.now() - timedelta(days=15),
                "phq_score": 12.0,
                "asrm_score": 2.0,
                "risk_level": "moderate",
            },
        ]

        assessment = interpreter.make_longitudinal_assessment(
            current_scores={"phq": 16.0, "asrm": 1.0},
            current_biomarkers={
                "sleep_hours": 11.0,
                "activity_steps": 1000,
            },
            historical_assessments=historical_data,
        )

        assert assessment.trajectory == "worsening"
        assert assessment.pattern_detected == "escalating_depression"
        assert "increasing severity" in assessment.clinical_note.lower()

    def test_evaluate_intervention_need_exists(self, interpreter):
        """Test that evaluate_intervention_need method exists."""
        assert hasattr(interpreter, "evaluate_intervention_need")

    def test_evaluate_intervention_need(self, interpreter):
        """Test early intervention decision making (from clinical_decision_engine)."""
        decision = interpreter.evaluate_intervention_need(
            warning_indicators={
                "sleep_change": 2.5,  # hours
                "activity_change": -35,  # percent
                "circadian_shift": 1.5,  # hours
                "consecutive_days": 4,
            },
            current_risk="low",
            patient_history={
                "previous_episodes": 2,
                "time_since_last_episode_days": 180,
                "medication_adherent": True,
            },
        )

        assert decision.recommend_intervention is True
        assert decision.intervention_type == "preventive"
        assert len(decision.rationale) > 0
        assert len(decision.specific_actions) > 0
