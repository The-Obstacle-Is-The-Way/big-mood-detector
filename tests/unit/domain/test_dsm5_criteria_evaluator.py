"""
Test DSM-5 Criteria Evaluator

Following TDD approach - writing tests first for the DSM5CriteriaEvaluator
that will be extracted from ClinicalInterpreter.
"""

import pytest
from datetime import datetime, timedelta

from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestDSM5CriteriaEvaluator:
    """Test DSM-5 criteria evaluation logic."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path
        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def evaluator(self, config):
        """Create DSM5CriteriaEvaluator instance."""
        # This will fail initially (Red phase of TDD)
        from big_mood_detector.domain.services.dsm5_criteria_evaluator import (
            DSM5CriteriaEvaluator,
        )
        return DSM5CriteriaEvaluator(config)

    def test_evaluate_manic_episode_duration_met(self, evaluator):
        """Test manic episode duration criteria - 7 days required."""
        result = evaluator.evaluate_episode_duration(
            episode_type="manic",
            symptom_days=8,
            hospitalization=False,
        )
        
        assert result.duration_met is True
        assert result.meets_criteria is True
        assert "meets DSM-5 criteria" in result.clinical_note

    def test_evaluate_manic_episode_duration_not_met(self, evaluator):
        """Test manic episode with insufficient duration."""
        result = evaluator.evaluate_episode_duration(
            episode_type="manic",
            symptom_days=5,
            hospitalization=False,
        )
        
        assert result.duration_met is False
        assert result.meets_criteria is False
        assert "insufficient" in result.clinical_note.lower()
        assert "5 days < 7 days" in result.clinical_note

    def test_evaluate_manic_episode_hospitalization_override(self, evaluator):
        """Test that hospitalization overrides duration requirement for mania."""
        result = evaluator.evaluate_episode_duration(
            episode_type="manic",
            symptom_days=2,
            hospitalization=True,
        )
        
        assert result.duration_met is True
        assert result.meets_criteria is True
        assert "hospitalization overrides" in result.clinical_note

    def test_evaluate_hypomanic_episode_duration(self, evaluator):
        """Test hypomanic episode - 4 days required, no hospitalization."""
        # Sufficient duration
        result = evaluator.evaluate_episode_duration(
            episode_type="hypomanic",
            symptom_days=5,
            hospitalization=False,
        )
        assert result.duration_met is True
        
        # Insufficient duration
        result = evaluator.evaluate_episode_duration(
            episode_type="hypomanic",
            symptom_days=3,
            hospitalization=False,
        )
        assert result.duration_met is False
        
        # Hospitalization invalidates hypomania (becomes mania)
        result = evaluator.evaluate_episode_duration(
            episode_type="hypomanic",
            symptom_days=5,
            hospitalization=True,
        )
        assert result.duration_met is False
        assert "hospitalization" in result.clinical_note

    def test_evaluate_depressive_episode_duration(self, evaluator):
        """Test depressive episode - 14 days required."""
        # Sufficient duration
        result = evaluator.evaluate_episode_duration(
            episode_type="depressive",
            symptom_days=15,
            hospitalization=False,
        )
        assert result.duration_met is True
        
        # Insufficient duration
        result = evaluator.evaluate_episode_duration(
            episode_type="depressive",
            symptom_days=10,
            hospitalization=False,
        )
        assert result.duration_met is False
        assert "10 days < 14 days" in result.clinical_note

    def test_evaluate_mixed_episode_duration(self, evaluator):
        """Test mixed episodes follow primary pole duration."""
        # Depressive mixed - 14 days
        result = evaluator.evaluate_episode_duration(
            episode_type="depressive_with_mixed_features",
            symptom_days=15,
            hospitalization=False,
        )
        assert result.duration_met is True
        
        # Manic mixed - 7 days or hospitalization
        result = evaluator.evaluate_episode_duration(
            episode_type="manic_with_mixed_features",
            symptom_days=3,
            hospitalization=True,
        )
        assert result.duration_met is True

    def test_evaluate_symptom_count_depression(self, evaluator):
        """Test symptom count evaluation for depression."""
        # DSM-5 requires 5+ symptoms for major depression
        symptoms = [
            "depressed_mood",
            "anhedonia",
            "weight_change",
            "sleep_disturbance",
            "psychomotor_change",
        ]
        
        result = evaluator.evaluate_symptom_count(
            symptoms=symptoms,
            episode_type="depressive",
        )
        assert result.symptom_count_met is True
        assert result.symptom_count == 5
        assert "meets minimum" in result.clinical_note

    def test_evaluate_symptom_count_insufficient(self, evaluator):
        """Test insufficient symptoms for diagnosis."""
        symptoms = ["depressed_mood", "anhedonia", "sleep_disturbance"]
        
        result = evaluator.evaluate_symptom_count(
            symptoms=symptoms,
            episode_type="depressive",
        )
        assert result.symptom_count_met is False
        assert result.symptom_count == 3
        assert "insufficient" in result.clinical_note.lower()

    def test_evaluate_functional_impairment(self, evaluator):
        """Test functional impairment assessment."""
        # Significant impairment indicators
        result = evaluator.evaluate_functional_impairment(
            work_impairment=True,
            social_impairment=True,
            self_care_impairment=False,
            hospitalization=False,
        )
        assert result.functional_impairment_met is True
        assert "significant impairment" in result.clinical_note
        
        # No significant impairment
        result = evaluator.evaluate_functional_impairment(
            work_impairment=False,
            social_impairment=False,
            self_care_impairment=False,
            hospitalization=False,
        )
        assert result.functional_impairment_met is False

    def test_evaluate_complete_dsm5_criteria(self, evaluator):
        """Test complete DSM-5 criteria evaluation."""
        result = evaluator.evaluate_complete_criteria(
            episode_type="manic",
            symptom_days=8,
            symptoms=["elevated_mood", "decreased_sleep", "grandiosity", 
                     "flight_of_ideas", "increased_activity", "poor_judgment"],
            hospitalization=False,
            functional_impairment=True,
        )
        
        assert result.meets_all_criteria is True
        assert result.duration_criteria.duration_met is True
        assert result.symptom_criteria.symptom_count_met is True
        assert result.functional_criteria.functional_impairment_met is True
        assert "DSM-5 criteria for manic episode" in result.summary

    def test_generate_dsm5_summary(self, evaluator):
        """Test clinical summary generation."""
        evaluation = evaluator.evaluate_complete_criteria(
            episode_type="depressive",
            symptom_days=10,
            symptoms=["depressed_mood", "anhedonia", "sleep_disturbance"],
            hospitalization=False,
            functional_impairment=True,
        )
        
        summary = evaluator.generate_clinical_summary(evaluation)
        
        assert "depressive episode" in summary.lower()
        assert "duration: insufficient" in summary.lower()
        assert "symptoms: 3/5" in summary.lower()
        assert "functional impairment: present" in summary.lower()
        assert "does not meet" in summary.lower()