"""
Test Clinical Interpreter Service

Tests the clinical interpretation of mood prediction scores based on
DSM-5 criteria and evidence-based thresholds from the Clinical Dossier.
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass

from big_mood_detector.domain.services.clinical_interpreter import (
    ClinicalInterpreter,
    ClinicalInterpretation,
    RiskLevel,
    EpisodeType,
    ClinicalRecommendation,
    DSM5Criteria,
)


class TestClinicalInterpreter:
    """Test clinical interpretation of mood predictions."""

    @pytest.fixture
    def interpreter(self):
        """Create clinical interpreter instance."""
        return ClinicalInterpreter()

    def test_depression_risk_stratification(self, interpreter):
        """Test depression risk level categorization based on PHQ scores."""
        # Low risk
        result = interpreter.interpret_depression_score(
            phq_score=4,
            sleep_hours=7.5,
            activity_steps=6000,
        )
        assert result.risk_level == RiskLevel.LOW
        assert result.episode_type == EpisodeType.NONE
        assert "within normal range" in result.clinical_summary.lower()

        # Moderate risk - PHQ-8 ≥ 10
        result = interpreter.interpret_depression_score(
            phq_score=12,
            sleep_hours=9.5,
            activity_steps=4000,
        )
        assert result.risk_level == RiskLevel.MODERATE
        assert result.episode_type == EpisodeType.DEPRESSIVE
        assert "moderate depression" in result.clinical_summary.lower()

        # High risk - PHQ-8 15-19
        result = interpreter.interpret_depression_score(
            phq_score=17,
            sleep_hours=11,
            activity_steps=2000,
        )
        assert result.risk_level == RiskLevel.HIGH
        assert result.episode_type == EpisodeType.DEPRESSIVE
        assert "moderately severe" in result.clinical_summary.lower()

        # Critical risk - PHQ-8 ≥ 20
        result = interpreter.interpret_depression_score(
            phq_score=22,
            sleep_hours=13,
            activity_steps=500,
            suicidal_ideation=True,
        )
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.episode_type == EpisodeType.DEPRESSIVE
        assert "immediate intervention" in result.clinical_summary.lower()
        assert any(
            "urgent" in rec.description.lower()
            for rec in result.recommendations
        )

    def test_mania_risk_stratification(self, interpreter):
        """Test mania/hypomania risk level categorization based on ASRM scores."""
        # Low risk
        result = interpreter.interpret_mania_score(
            asrm_score=3,
            sleep_hours=7,
            activity_steps=7000,
        )
        assert result.risk_level == RiskLevel.LOW
        assert result.episode_type == EpisodeType.NONE

        # Moderate risk - ASRM 6-10
        result = interpreter.interpret_mania_score(
            asrm_score=8,
            sleep_hours=5,
            activity_steps=12000,
        )
        assert result.risk_level == RiskLevel.MODERATE
        assert result.episode_type == EpisodeType.HYPOMANIC

        # High risk - ASRM 11-15
        result = interpreter.interpret_mania_score(
            asrm_score=13,
            sleep_hours=3.5,
            activity_steps=18000,
        )
        assert result.risk_level == RiskLevel.HIGH
        assert result.episode_type == EpisodeType.HYPOMANIC

        # Critical risk - ASRM > 15 or sleep < 3 hours
        result = interpreter.interpret_mania_score(
            asrm_score=18,
            sleep_hours=2,
            activity_steps=25000,
            psychotic_features=True,
        )
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.episode_type == EpisodeType.MANIC
        assert "immediate" in result.clinical_summary.lower()

    def test_mixed_features_detection(self, interpreter):
        """Test detection of mixed features based on DSM-5 criteria."""
        # Depression with mixed features
        result = interpreter.interpret_mixed_state(
            phq_score=15,
            asrm_score=7,
            sleep_hours=4,
            activity_steps=15000,
            racing_thoughts=True,
            increased_energy=True,
        )
        assert result.episode_type == EpisodeType.DEPRESSIVE_MIXED
        assert "mixed features" in result.clinical_summary.lower()
        assert any(
            "cariprazine" in rec.description.lower() or
            "lurasidone" in rec.description.lower()
            for rec in result.recommendations
        )

        # Mania with mixed features
        result = interpreter.interpret_mixed_state(
            phq_score=8,
            asrm_score=12,
            sleep_hours=3,
            activity_steps=20000,
            depressed_mood=True,
            anhedonia=True,
            guilt=True,
        )
        assert result.episode_type == EpisodeType.MANIC_MIXED
        assert "mixed features" in result.clinical_summary.lower()

    def test_dsm5_duration_criteria(self, interpreter):
        """Test DSM-5 episode duration requirements."""
        # Too short for manic episode (< 7 days)
        result = interpreter.evaluate_episode_duration(
            episode_type=EpisodeType.MANIC,
            symptom_days=5,
            hospitalization=False,
        )
        assert not result.meets_dsm5_criteria
        assert result.clinical_note == "Duration insufficient for manic episode (5 days < 7 days required)"

        # Valid manic episode
        result = interpreter.evaluate_episode_duration(
            episode_type=EpisodeType.MANIC,
            symptom_days=8,
            hospitalization=False,
        )
        assert result.meets_dsm5_criteria

        # Manic episode with hospitalization (any duration)
        result = interpreter.evaluate_episode_duration(
            episode_type=EpisodeType.MANIC,
            symptom_days=3,
            hospitalization=True,
        )
        assert result.meets_dsm5_criteria
        assert "hospitalization" in result.clinical_note.lower()

        # Hypomanic episode (≥ 4 days)
        result = interpreter.evaluate_episode_duration(
            episode_type=EpisodeType.HYPOMANIC,
            symptom_days=4,
            hospitalization=False,
        )
        assert result.meets_dsm5_criteria

        # Major depressive episode (≥ 14 days)
        result = interpreter.evaluate_episode_duration(
            episode_type=EpisodeType.DEPRESSIVE,
            symptom_days=14,
            hospitalization=False,
        )
        assert result.meets_dsm5_criteria

    def test_digital_biomarker_interpretation(self, interpreter):
        """Test interpretation of digital biomarkers."""
        # Sleep duration biomarkers
        result = interpreter.interpret_sleep_biomarkers(
            sleep_duration=2.5,  # < 3 hours critical
            sleep_efficiency=0.65,  # < 85% poor
            sleep_timing_variance=3.5,  # > 2 hours variable
        )
        assert result.mania_risk_factors == 3
        assert "critical short sleep" in result.clinical_notes[0].lower()
        assert result.recommendation_priority == "urgent"

        # Activity biomarkers
        result = interpreter.interpret_activity_biomarkers(
            daily_steps=22000,  # > 15000 high
            sedentary_hours=2,  # very low
            activity_variance=8500,  # high variance
        )
        assert result.mania_risk_factors >= 2
        assert "elevated activity" in result.clinical_notes[0].lower()

        # Circadian rhythm biomarkers
        result = interpreter.interpret_circadian_biomarkers(
            circadian_phase_advance=3.5,  # > 2 hours
            interdaily_stability=0.3,  # < 0.5 low
            intradaily_variability=1.5,  # > 1 high
        )
        assert result.mood_instability_risk == "high"
        assert "circadian disruption" in result.clinical_summary.lower()

    def test_early_warning_triggers(self, interpreter):
        """Test early warning sign detection and triggers."""
        # Depression early warning
        warnings = interpreter.detect_early_warnings(
            sleep_increase_hours=2.5,  # > 2 hours
            activity_decrease_percent=35,  # > 30%
            circadian_delay_hours=1.5,  # > 1 hour
            consecutive_days=3,
        )
        assert warnings.depression_warning == True
        assert warnings.trigger_intervention == True
        assert "sleep increase" in warnings.warning_signs[0].lower()

        # Mania early warning
        warnings = interpreter.detect_early_warnings(
            sleep_decrease_hours=3,  # significant
            activity_increase_percent=60,  # > 50%
            speech_rate_increase=True,
            consecutive_days=2,
        )
        assert warnings.mania_warning == True
        assert len(warnings.warning_signs) >= 3

    def test_treatment_recommendations(self, interpreter):
        """Test evidence-based treatment recommendations."""
        # Acute mania recommendations
        recs = interpreter.get_treatment_recommendations(
            episode_type=EpisodeType.MANIC,
            severity=RiskLevel.HIGH,
            current_medications=["lithium"],
            contraindications=[],
        )
        assert len(recs) > 0
        assert any("quetiapine" in r.medication.lower() for r in recs)
        assert all(r.evidence_level in ["first-line", "second-line"] for r in recs)

        # Depression with antidepressant contraindication
        recs = interpreter.get_treatment_recommendations(
            episode_type=EpisodeType.DEPRESSIVE,
            severity=RiskLevel.MODERATE,
            current_medications=[],
            rapid_cycling=True,
        )
        assert not any("antidepressant monotherapy" in r.medication.lower() for r in recs)
        assert any("lamotrigine" not in r.medication.lower() for r in recs)  # contraindicated in rapid cycling

    def test_confidence_adjustment(self, interpreter):
        """Test confidence adjustment based on data quality."""
        # High quality data
        result = interpreter.adjust_confidence(
            base_confidence=0.85,
            data_completeness=0.95,  # > 75% threshold
            days_of_data=45,  # > 30 days minimum
            missing_features=[],
        )
        assert result.adjusted_confidence >= 0.85
        assert result.reliability == "high"

        # Poor quality data
        result = interpreter.adjust_confidence(
            base_confidence=0.85,
            data_completeness=0.60,  # < 75% threshold
            days_of_data=15,  # < 30 days minimum
            missing_features=["sleep", "heart_rate"],
        )
        assert result.adjusted_confidence < 0.70
        assert result.reliability == "low"
        assert "insufficient data" in result.limitations[0].lower()

    def test_clinical_summary_generation(self, interpreter):
        """Test generation of clinical summaries."""
        interpretation = ClinicalInterpretation(
            risk_level=RiskLevel.HIGH,
            episode_type=EpisodeType.DEPRESSIVE,
            confidence=0.82,
            clinical_summary="High risk for major depressive episode",
            dsm5_criteria_met=True,
            clinical_features={
                "phq_score": 18,
                "sleep_duration": 11.5,
                "activity_reduction": 45,
            },
        )
        
        summary = interpreter.generate_clinical_summary(interpretation)
        
        assert "high risk" in summary.lower()
        assert "major depressive episode" in summary.lower()
        assert "dsm-5 criteria" in summary.lower()
        assert "82%" in summary or "0.82" in summary
        assert summary.count(".") >= 3  # Multiple sentences

    def test_risk_trend_analysis(self, interpreter):
        """Test analysis of risk trends over time."""
        # Worsening trend
        trend = interpreter.analyze_risk_trend(
            risk_scores=[0.3, 0.4, 0.5, 0.65, 0.75],
            dates=[datetime.now() - timedelta(days=i) for i in range(4, -1, -1)],
            episode_type=EpisodeType.DEPRESSIVE,
        )
        assert trend.direction == "worsening"
        assert trend.velocity > 0
        assert "increasing" in trend.clinical_note.lower()

        # Improving trend
        trend = interpreter.analyze_risk_trend(
            risk_scores=[0.8, 0.7, 0.6, 0.45, 0.3],
            dates=[datetime.now() - timedelta(days=i) for i in range(4, -1, -1)],
            episode_type=EpisodeType.MANIC,
        )
        assert trend.direction == "improving"
        assert trend.velocity < 0

    def test_personalized_thresholds(self, interpreter):
        """Test personalized threshold adjustments."""
        # Individual with lower baseline
        thresholds = interpreter.calculate_personalized_thresholds(
            individual_baseline={
                "sleep_mean": 6.5,
                "sleep_std": 0.8,
                "activity_mean": 4500,
                "activity_std": 1200,
            },
            population_norms={
                "sleep_mean": 7.5,
                "activity_mean": 6631,
            },
        )
        assert thresholds.sleep_low < 6.5  # Adjusted for individual
        assert thresholds.activity_low < 4500

    def test_clinical_decision_rules(self, interpreter):
        """Test implementation of clinical decision rules."""
        # Rule: No antidepressant monotherapy in bipolar
        decision = interpreter.apply_clinical_rules(
            diagnosis="bipolar_disorder",
            proposed_treatment="sertraline",
            current_medications=[],
            mood_state="depressed",
        )
        assert decision.approved == False
        assert "contraindicated" in decision.rationale.lower()
        assert "antidepressant monotherapy" in decision.rationale.lower()

        # Rule: Lithium + antidepressant is acceptable
        decision = interpreter.apply_clinical_rules(
            diagnosis="bipolar_disorder",
            proposed_treatment="sertraline",
            current_medications=["lithium"],
            mood_state="depressed",
        )
        assert decision.approved == True
        assert "mood stabilizer coverage" in decision.rationale.lower()