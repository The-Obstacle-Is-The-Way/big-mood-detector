"""
Clinical Assessment Service

Handles comprehensive clinical assessments by orchestrating multiple evaluation services.
Extracted from clinical_interpreter to follow Single Responsibility Principle.
"""

from dataclasses import dataclass, field
from typing import Any

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)
from big_mood_detector.domain.services.dsm5_criteria_evaluator import (
    DSM5CriteriaEvaluator,
)
from big_mood_detector.domain.services.treatment_recommender import (
    TreatmentRecommender,
)


@dataclass(frozen=True)
class ClinicalAssessment:
    """Comprehensive clinical assessment result."""

    primary_diagnosis: str
    risk_level: str
    meets_dsm5_criteria: bool
    confidence: float
    clinical_summary: str
    treatment_options: list[dict[str, Any]] = field(default_factory=list)
    requires_immediate_intervention: bool = False
    complexity_score: float = 0.0
    data_completeness: float = 1.0
    limitations: list[str] = field(default_factory=list)


class ClinicalAssessmentService:
    """
    Service for making comprehensive clinical assessments.

    This service orchestrates DSM-5 evaluation and treatment recommendations
    to produce a complete clinical assessment.
    """

    def __init__(
        self,
        dsm5_evaluator: DSM5CriteriaEvaluator,
        treatment_recommender: TreatmentRecommender,
        config: ClinicalThresholdsConfig,
    ):
        """Initialize with required services."""
        self.dsm5_evaluator = dsm5_evaluator
        self.treatment_recommender = treatment_recommender
        self.config = config

    def make_clinical_assessment(
        self,
        mood_scores: dict[str, float],
        biomarkers: dict[str, float],
        clinical_context: dict[str, Any],
    ) -> ClinicalAssessment:
        """
        Make comprehensive clinical assessment.

        This method integrates mood scores, biomarkers, and clinical context
        to produce a complete clinical assessment.
        """
        # Interpret mood scores
        phq_score = mood_scores.get("phq", 0)
        asrm_score = mood_scores.get("asrm", 0)

        # Determine primary diagnosis
        primary_diagnosis = self._determine_primary_diagnosis(phq_score, asrm_score)

        # Assess risk level
        risk_level = self._calculate_risk_level(phq_score, asrm_score, clinical_context)

        # Check DSM-5 criteria
        meets_dsm5_criteria = self._evaluate_dsm5_criteria(
            primary_diagnosis, clinical_context, phq_score, asrm_score
        )

        # Get treatment recommendations
        treatment_options = self._get_treatment_recommendations(
            primary_diagnosis, risk_level, clinical_context
        )

        # Calculate confidence and complexity
        confidence = self._calculate_confidence(mood_scores, biomarkers)
        complexity_score = self._calculate_complexity(clinical_context)

        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            primary_diagnosis, risk_level, meets_dsm5_criteria
        )

        return ClinicalAssessment(
            primary_diagnosis=primary_diagnosis,
            risk_level=risk_level,
            meets_dsm5_criteria=meets_dsm5_criteria,
            confidence=confidence,
            clinical_summary=clinical_summary,
            treatment_options=treatment_options,
            requires_immediate_intervention=risk_level == "critical",
            complexity_score=complexity_score,
            data_completeness=self._calculate_data_completeness(mood_scores, biomarkers),
            limitations=self._identify_limitations(mood_scores, biomarkers),
        )

    def _determine_primary_diagnosis(self, phq_score: float, asrm_score: float) -> str:
        """Determine primary diagnosis based on scores."""
        if phq_score >= 10 and asrm_score < 6:
            return "depressive_episode"
        elif asrm_score >= 6 and phq_score < 10:
            return "manic_episode" if asrm_score >= 14 else "hypomanic_episode"
        elif phq_score >= 10 and asrm_score >= 6:
            return "mixed_episode"
        else:
            return "euthymic"

    def _calculate_risk_level(
        self, phq_score: float, asrm_score: float, clinical_context: dict[str, Any]
    ) -> str:
        """Calculate overall risk level."""
        if clinical_context.get("suicidal_ideation") or clinical_context.get("psychotic_features"):
            return "critical"
        elif phq_score >= 15 or asrm_score >= 14:
            return "high"
        elif phq_score >= 10 or asrm_score >= 6:
            return "moderate"
        else:
            return "low"

    def _evaluate_dsm5_criteria(
        self,
        primary_diagnosis: str,
        clinical_context: dict[str, Any],
        phq_score: float,
        asrm_score: float,
    ) -> bool:
        """Evaluate DSM-5 criteria for the diagnosis."""
        # For simplicity, assume symptoms based on scores
        symptoms = []
        if phq_score >= 10:
            symptoms.extend(["depressed_mood", "anhedonia", "sleep_disturbance", "fatigue", "concentration"])
        if asrm_score >= 6:
            symptoms.extend(["elevated_mood", "decreased_sleep", "increased_activity"])

        # Map diagnosis to DSM-5 episode type
        dsm5_episode_type = primary_diagnosis
        if primary_diagnosis == "depressive_episode":
            dsm5_episode_type = "depressive"
        elif primary_diagnosis == "manic_episode":
            dsm5_episode_type = "manic"
        elif primary_diagnosis == "hypomanic_episode":
            dsm5_episode_type = "hypomanic"
        elif primary_diagnosis == "mixed_episode":
            dsm5_episode_type = "mixed"

        result = self.dsm5_evaluator.evaluate_complete_criteria(
            episode_type=dsm5_episode_type,
            symptom_days=clinical_context.get("symptom_days", 0),
            symptoms=symptoms,
            functional_impairment=clinical_context.get("functional_impairment", False),
        )

        return result.meets_all_criteria

    def _get_treatment_recommendations(
        self,
        primary_diagnosis: str,
        risk_level: str,
        clinical_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Get treatment recommendations."""
        # Map diagnosis to episode type expected by treatment recommender
        episode_type = primary_diagnosis
        if primary_diagnosis == "depressive_episode":
            episode_type = "depressive"
        elif primary_diagnosis == "manic_episode":
            episode_type = "manic"
        elif primary_diagnosis == "hypomanic_episode":
            episode_type = "hypomanic"
        elif primary_diagnosis == "mixed_episode":
            episode_type = "mixed"

        recommendations = self.treatment_recommender.get_recommendations(
            episode_type=episode_type,
            severity=risk_level,
            current_medications=clinical_context.get("current_medications", []),
            contraindications=clinical_context.get("contraindications", []),
            rapid_cycling=clinical_context.get("rapid_cycling", False),
        )

        return [
            {
                "treatment": rec.medication,
                "priority": rec.evidence_level,
                "description": rec.description,
            }
            for rec in recommendations
        ]

    def _calculate_confidence(
        self, mood_scores: dict[str, float], biomarkers: dict[str, float]
    ) -> float:
        """Calculate confidence in assessment."""
        # Base confidence on data completeness
        data_points = len(mood_scores) + len(biomarkers)
        return min(0.5 + (data_points * 0.1), 0.95)

    def _calculate_complexity(self, clinical_context: dict[str, Any]) -> float:
        """Calculate clinical complexity score."""
        complexity_factors = [
            clinical_context.get("comorbidities", 0) > 0,
            clinical_context.get("substance_use", False),
            clinical_context.get("psychotic_features", False),
            clinical_context.get("mixed_features", False),
        ]
        return float(sum(complexity_factors)) / float(len(complexity_factors))

    def _generate_clinical_summary(
        self, diagnosis: str, risk_level: str, meets_criteria: bool
    ) -> str:
        """Generate clinical summary text."""
        criteria_text = "meets" if meets_criteria else "does not meet"
        return (
            f"Patient presents with {diagnosis.replace('_', ' ')} "
            f"with {risk_level} risk level. "
            f"Clinical presentation {criteria_text} DSM-5 criteria."
        )

    def _calculate_data_completeness(
        self, mood_scores: dict[str, float], biomarkers: dict[str, float]
    ) -> float:
        """Calculate data completeness score."""
        expected_scores = ["phq", "asrm"]
        expected_biomarkers = ["sleep_hours", "activity_steps"]

        score_completeness = sum(s in mood_scores for s in expected_scores) / len(expected_scores)
        bio_completeness = sum(b in biomarkers for b in expected_biomarkers) / len(expected_biomarkers)

        return (score_completeness + bio_completeness) / 2

    def _identify_limitations(
        self, mood_scores: dict[str, float], biomarkers: dict[str, float]
    ) -> list[str]:
        """Identify assessment limitations."""
        limitations = []
        if "phq" not in mood_scores:
            limitations.append("Missing depression screening data")
        if "sleep_hours" not in biomarkers:
            limitations.append("Missing sleep data")
        return limitations

