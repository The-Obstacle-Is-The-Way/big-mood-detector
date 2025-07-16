"""
Test Risk Level Assessor

TDD approach for extracting risk assessment logic from ClinicalInterpreter.
Following Uncle Bob's principles: clean, testable, single responsibility.
"""

import pytest

from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestRiskLevelAssessor:
    """Test risk level assessment logic."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path

        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def assessor(self, config):
        """Create RiskLevelAssessor instance."""
        # Red phase - this will fail initially
        from big_mood_detector.domain.services.risk_level_assessor import (
            RiskLevelAssessor,
        )

        return RiskLevelAssessor(config)

    def test_assess_depression_risk_none(self, assessor):
        """Test no depression risk assessment."""
        result = assessor.assess_depression_risk(
            phq_score=2.0,
            sleep_hours=7.5,
            activity_steps=6000,
            suicidal_ideation=False,
        )

        assert result.risk_level == "low"
        assert result.severity_score == 2.0
        assert result.confidence >= 0.8
        assert "minimal depressive symptoms" in result.clinical_rationale.lower()

    def test_assess_depression_risk_moderate(self, assessor):
        """Test moderate depression risk."""
        result = assessor.assess_depression_risk(
            phq_score=12.0,
            sleep_hours=10.0,  # Hypersomnia
            activity_steps=2000,  # Low activity
            suicidal_ideation=False,
        )

        assert result.risk_level == "moderate"
        assert result.severity_score == 12.0
        assert len(result.risk_factors) >= 1  # PHQ score
        assert "moderate depression" in result.clinical_rationale.lower()

    def test_assess_depression_risk_high_with_suicidality(self, assessor):
        """Test high risk with suicidal ideation."""
        result = assessor.assess_depression_risk(
            phq_score=18.0,
            sleep_hours=12.0,
            activity_steps=1000,
            suicidal_ideation=True,
        )

        assert result.risk_level == "critical"  # Upgraded due to SI
        assert result.severity_score == 18.0
        assert "suicidal ideation" in result.clinical_rationale.lower()
        assert result.requires_immediate_action is True

    def test_assess_mania_risk_none(self, assessor):
        """Test no mania risk assessment."""
        result = assessor.assess_mania_risk(
            asrm_score=2.0,
            sleep_hours=7.0,
            activity_steps=6000,
            psychotic_features=False,
        )

        assert result.risk_level == "low"
        assert result.severity_score == 2.0
        assert "no manic symptoms" in result.clinical_rationale.lower()

    def test_assess_mania_risk_hypomanic(self, assessor):
        """Test hypomanic risk assessment."""
        result = assessor.assess_mania_risk(
            asrm_score=7.0,
            sleep_hours=4.5,  # Reduced sleep
            activity_steps=15000,  # Increased activity
            psychotic_features=False,
        )

        assert result.risk_level == "moderate"
        assert "hypomanic" in result.clinical_rationale.lower()
        assert len(result.risk_factors) >= 2  # ASRM + sleep (activity boundary not met)

    def test_assess_mania_risk_severe_with_psychosis(self, assessor):
        """Test severe mania with psychotic features."""
        result = assessor.assess_mania_risk(
            asrm_score=15.0,
            sleep_hours=2.0,
            activity_steps=25000,
            psychotic_features=True,
        )

        assert result.risk_level == "critical"
        assert "psychotic features" in result.clinical_rationale.lower()
        assert result.requires_immediate_action is True

    def test_assess_mixed_state_risk(self, assessor):
        """Test mixed state risk assessment."""
        result = assessor.assess_mixed_state_risk(
            phq_score=12.0,
            asrm_score=8.0,
            sleep_disturbance=True,
            racing_thoughts=True,
            anhedonia=True,
        )

        assert result.risk_level in ["high", "critical"]
        assert "mixed" in result.clinical_rationale.lower()
        assert result.mixed_features_count >= 3
        assert len(result.risk_factors) >= 4

    def test_calculate_composite_risk(self, assessor):
        """Test composite risk calculation from multiple factors."""
        factors = {
            "depression_score": 15.0,
            "mania_score": 3.0,
            "sleep_disruption": 2.5,  # Severe
            "activity_variance": 0.8,  # High
            "circadian_misalignment": True,
            "medication_adherence": 0.6,  # Poor
        }

        result = assessor.calculate_composite_risk(factors)

        assert result.overall_risk_level in [
            "moderate",
            "high",
            "critical",
        ]  # PHQ 15 is severe
        assert result.primary_concern == "depression"
        assert "multiple risk factors" in result.clinical_summary.lower()
        assert result.confidence_adjusted is True

    def test_risk_trajectory_analysis(self, assessor):
        """Test risk trajectory over time."""
        historical_risks = [
            {"date": "2024-01-01", "risk_level": "low", "score": 3.0},
            {"date": "2024-01-15", "risk_level": "moderate", "score": 8.0},
            {"date": "2024-02-01", "risk_level": "high", "score": 15.0},
        ]

        result = assessor.analyze_risk_trajectory(
            historical_risks,
            current_risk_level="high",
            current_score=16.0,
        )

        assert result.trend == "worsening"
        assert result.velocity > 0  # Increasing risk
        assert "increasing" in result.clinical_note.lower()
        assert result.predicted_trajectory == "continued_worsening"

    def test_biomarker_risk_modulation(self, assessor):
        """Test how biomarkers modulate risk assessment."""
        # Moderate PHQ but severe biomarkers
        result = assessor.assess_depression_risk(
            phq_score=10.0,  # Moderate
            sleep_hours=2.0,  # Severe insomnia
            activity_steps=500,  # Severe hypoactivity
            suicidal_ideation=False,
        )

        # Risk should be upgraded due to biomarkers
        assert result.risk_level == "high"
        assert "biomarker severity" in result.clinical_rationale.lower()
        assert result.biomarker_modifier > 1.0

    def test_confidence_adjustment_for_incomplete_data(self, assessor):
        """Test confidence adjustment when data is incomplete."""
        result = assessor.assess_depression_risk(
            phq_score=15.0,
            sleep_hours=None,  # Missing data
            activity_steps=None,  # Missing data
            suicidal_ideation=False,
        )

        assert result.risk_level == "high"
        assert result.confidence < 0.85  # Reduced confidence (0.9 * 0.9 = 0.81)
        assert "limited data" in result.clinical_rationale.lower()
        assert len(result.missing_factors) == 2
