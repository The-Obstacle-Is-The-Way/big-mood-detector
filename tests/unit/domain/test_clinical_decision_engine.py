"""
Test Clinical Decision Engine

TDD approach for creating the clinical decision orchestration engine.
This engine coordinates multiple specialized services to make clinical decisions.
"""

from datetime import datetime, timedelta

import pytest

from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestClinicalDecisionEngine:
    """Test clinical decision orchestration logic."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path

        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def engine(self, config):
        """Create ClinicalDecisionEngine instance."""
        from big_mood_detector.domain.services.clinical_decision_engine import (
            ClinicalDecisionEngine,
        )

        return ClinicalDecisionEngine(config)

    def test_make_depression_assessment(self, engine):
        """Test comprehensive depression assessment."""
        assessment = engine.make_clinical_assessment(
            mood_scores={"phq": 15.0},
            biomarkers={
                "sleep_hours": 10.5,
                "activity_steps": 1500,
            },
            clinical_context={
                "symptom_days": 16,
                "hospitalization": False,
                "suicidal_ideation": False,
            },
        )

        assert assessment.primary_diagnosis == "depressive_episode"
        assert assessment.risk_level == "high"
        assert assessment.meets_dsm5_criteria is True
        assert len(assessment.treatment_options) > 0
        assert assessment.confidence > 0.7

    def test_make_mania_assessment_with_psychosis(self, engine):
        """Test mania assessment with psychotic features."""
        assessment = engine.make_clinical_assessment(
            mood_scores={"asrm": 18.0},
            biomarkers={
                "sleep_hours": 2.0,
                "activity_steps": 25000,
            },
            clinical_context={
                "symptom_days": 8,
                "hospitalization": False,
                "psychotic_features": True,
            },
        )

        assert assessment.primary_diagnosis == "manic_episode"
        assert assessment.risk_level == "critical"
        assert assessment.requires_immediate_intervention is True
        assert "psychotic" in assessment.clinical_summary.lower()

    def test_make_mixed_state_assessment(self, engine):
        """Test mixed state assessment."""
        assessment = engine.make_clinical_assessment(
            mood_scores={
                "phq": 12.0,
                "asrm": 9.0,
            },
            biomarkers={
                "sleep_hours": 5.0,
                "activity_steps": 8000,
            },
            clinical_context={
                "symptom_days": 14,
                "racing_thoughts": True,
                "anhedonia": True,
                "increased_energy": True,
            },
        )

        assert "mixed" in assessment.primary_diagnosis
        assert assessment.risk_level in ["high", "critical"]
        assert assessment.complexity_score > 0.7  # Mixed states are complex

    def test_longitudinal_assessment(self, engine):
        """Test assessment with historical data."""
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

        assessment = engine.make_longitudinal_assessment(
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

    def test_early_intervention_decision(self, engine):
        """Test early intervention decision making."""
        # Subtle early warning signs
        decision = engine.evaluate_intervention_need(
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
        assert decision.urgency in ["moderate", "high"]  # Depends on warning severity
        assert "early warning signs" in decision.rationale.lower()

    def test_treatment_decision_with_contraindications(self, engine):
        """Test treatment decisions considering contraindications."""
        decision = engine.make_treatment_decision(
            diagnosis="manic_episode",
            severity="high",
            patient_factors={
                "current_medications": ["lithium"],
                "contraindications": ["valproate"],  # e.g., pregnancy
                "previous_response": {
                    "lithium": "partial",
                    "quetiapine": "good",
                },
            },
        )

        assert "valproate" not in [t.medication for t in decision.recommendations]
        assert any(t.medication == "quetiapine" for t in decision.recommendations)
        assert decision.considers_previous_response is True

    def test_confidence_adjustment_incomplete_data(self, engine):
        """Test confidence adjustment with incomplete data."""
        assessment = engine.make_clinical_assessment(
            mood_scores={"phq": 14.0},  # Only depression score
            biomarkers={},  # Missing biomarkers
            clinical_context={
                "symptom_days": 15,
            },
        )

        assert (
            assessment.confidence < 0.5
        )  # Lower confidence due to incomplete data (0.4 * 0.9 = 0.36)
        assert assessment.data_completeness < 0.5
        assert "limited data" in assessment.limitations[0].lower()

    def test_emergency_triage_decision(self, engine):
        """Test emergency triage for critical cases."""
        triage = engine.triage_urgency(
            indicators={
                "suicidal_ideation": True,
                "phq_score": 24.0,
                "sleep_hours": 2.0,
                "previous_attempts": 1,
            }
        )

        assert triage.urgency_level == "emergency"
        assert triage.recommended_action == "immediate_psychiatric_evaluation"
        assert triage.estimated_response_time == "within_24_hours"

    def test_personalized_decision_making(self, engine):
        """Test personalized clinical decisions."""
        personalized = engine.make_personalized_assessment(
            current_state={
                "phq": 10.0,
                "asrm": 4.0,
                "sleep_hours": 7.0,
            },
            individual_baseline={
                "typical_sleep": 8.5,
                "typical_activity": 7000,
                "sensitivity_profile": {
                    "sleep_sensitive": True,
                    "rapid_cycler": False,
                },
            },
        )

        assert personalized.uses_personalized_thresholds is True
        assert personalized.deviation_from_baseline > 0
        assert "individual baseline" in personalized.clinical_note.lower()

    def test_multidomain_integration(self, engine):
        """Test integration of multiple clinical domains."""
        integrated = engine.integrate_assessments(
            mood_assessment={"risk_level": "moderate", "episode_type": "depressive"},
            biomarker_assessment={"instability": "high", "circadian_disruption": True},
            early_warning_assessment={
                "warnings_detected": True,
                "trigger_intervention": False,
            },
            treatment_assessment={"current_effective": False, "needs_adjustment": True},
        )

        assert (
            integrated.overall_clinical_picture == "unstable_with_treatment_resistance"
        )
        assert integrated.priority_actions[0] == "medication_optimization"
        assert len(integrated.integrated_recommendations) >= 3

    def test_clinical_pathway_selection(self, engine):
        """Test selection of appropriate clinical pathway."""
        pathway = engine.select_clinical_pathway(
            presentation={
                "primary_symptoms": "depression",
                "comorbidities": ["anxiety", "insomnia"],
                "treatment_history": "multiple_failed_ssri",
                "psychosocial_factors": ["high_stress", "poor_support"],
            }
        )

        assert pathway.pathway_name == "treatment_resistant_depression"
        assert "psychotherapy" in pathway.recommended_interventions
        assert pathway.expected_timeline_weeks > 8
