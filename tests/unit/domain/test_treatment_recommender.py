"""
Test Treatment Recommender Service

Tests for treatment recommendations and clinical rules.
Following TDD with clean code principles.
"""

import pytest

from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestTreatmentRecommender:
    """Test treatment recommendation service."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path
        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def recommender(self, config):
        """Create treatment recommender with test config."""
        from big_mood_detector.domain.services.treatment_recommender import TreatmentRecommender
        return TreatmentRecommender(config)

    def test_recommend_for_acute_mania(self, recommender):
        """Test recommendations for acute manic episode."""
        recommendations = recommender.get_recommendations(
            episode_type="manic",
            severity="high",
            current_medications=[],
        )
        
        assert len(recommendations) > 0
        # Should include first-line treatments
        med_names = [r.medication for r in recommendations]
        assert any(med in med_names for med in ["lithium", "quetiapine"])
        assert all(r.evidence_level == "first-line" for r in recommendations)

    def test_recommend_for_bipolar_depression(self, recommender):
        """Test recommendations for bipolar depression."""
        recommendations = recommender.get_recommendations(
            episode_type="depressive",
            severity="moderate",
            current_medications=[],
        )
        
        assert len(recommendations) > 0
        med_names = [r.medication for r in recommendations]
        assert "quetiapine" in med_names  # First-line for bipolar depression

    def test_avoid_duplicate_medications(self, recommender):
        """Test that current medications are not recommended again."""
        recommendations = recommender.get_recommendations(
            episode_type="manic",
            severity="high",
            current_medications=["lithium"],
        )
        
        med_names = [r.medication for r in recommendations]
        assert "lithium" not in med_names

    def test_recommend_for_mixed_state(self, recommender):
        """Test recommendations for mixed episodes."""
        recommendations = recommender.get_recommendations(
            episode_type="depressive_with_mixed_features",
            severity="high",
            current_medications=[],
        )
        
        assert len(recommendations) > 0
        med_names = [r.medication for r in recommendations]
        # Should include second-line treatments for mixed features
        assert any(med in med_names for med in ["cariprazine", "lurasidone"])

    def test_apply_clinical_rule_no_antidepressant_monotherapy(self, recommender):
        """Test rule: No antidepressant monotherapy in bipolar disorder."""
        decision = recommender.apply_clinical_rules(
            diagnosis="bipolar_disorder",
            proposed_treatment="sertraline",
            current_medications=[],
            mood_state="depressed",
        )
        
        assert decision.approved is False
        assert "monotherapy is contraindicated" in decision.rationale

    def test_apply_clinical_rule_antidepressant_with_mood_stabilizer_ok(self, recommender):
        """Test rule: Antidepressant OK with mood stabilizer."""
        decision = recommender.apply_clinical_rules(
            diagnosis="bipolar_disorder",
            proposed_treatment="sertraline",
            current_medications=["lithium"],
            mood_state="depressed",
        )
        
        assert decision.approved is True
        assert "mood stabilizer coverage" in decision.rationale

    def test_recommend_for_rapid_cycling(self, recommender):
        """Test recommendations exclude lamotrigine for rapid cycling."""
        recommendations = recommender.get_recommendations(
            episode_type="depressive",
            severity="moderate",
            current_medications=[],
            rapid_cycling=True,
        )
        
        med_names = [r.medication for r in recommendations]
        assert "lamotrigine" not in med_names  # Contraindicated in rapid cycling

    def test_prioritize_urgent_cases(self, recommender):
        """Test urgent recommendations for critical severity."""
        recommendations = recommender.get_recommendations(
            episode_type="manic",
            severity="critical",
            current_medications=[],
        )
        
        # Should include hospitalization evaluation
        assert any("hospitalization" in r.medication.lower() for r in recommendations)
        # Check for any urgent-related text
        assert any(
            "urgent" in r.description.lower() or 
            "immediate" in r.description.lower() 
            for r in recommendations
        )

    def test_recommendation_includes_description(self, recommender):
        """Test that recommendations include helpful descriptions."""
        recommendations = recommender.get_recommendations(
            episode_type="depressive",
            severity="moderate",
            current_medications=[],
        )
        
        for rec in recommendations:
            assert rec.medication
            assert rec.evidence_level
            assert rec.description
            assert len(rec.description) > 10  # Meaningful description