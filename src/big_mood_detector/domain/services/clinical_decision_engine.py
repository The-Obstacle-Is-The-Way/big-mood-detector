"""
Clinical Decision Engine

High-level orchestration of clinical decision-making using multiple specialized services.
This engine coordinates the various clinical interpretation services to provide
comprehensive clinical assessments and treatment decisions.

Design Patterns:
- Facade Pattern: Simplifies complex clinical decision-making
- Strategy Pattern: Different decision strategies based on presentation
- Chain of Responsibility: Sequential processing of clinical data
- Dependency Injection: All services injected via constructor
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from big_mood_detector.domain.services.biomarker_interpreter import (
    BiomarkerInterpreter,
)
from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)
from big_mood_detector.domain.services.dsm5_criteria_evaluator import (
    DSM5CriteriaEvaluator,
)
from big_mood_detector.domain.services.early_warning_detector import (
    EarlyWarningDetector,
)
from big_mood_detector.domain.services.episode_interpreter import (
    EpisodeInterpreter,
)
from big_mood_detector.domain.services.risk_level_assessor import (
    RiskLevelAssessor,
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


@dataclass(frozen=True)
class LongitudinalAssessment:
    """Assessment incorporating historical data."""

    trajectory: str  # improving, stable, worsening
    pattern_detected: str
    clinical_note: str
    risk_projection: str
    confidence_in_trend: float


@dataclass(frozen=True)
class InterventionDecision:
    """Decision about clinical intervention."""

    recommend_intervention: bool
    intervention_type: str  # preventive, acute, maintenance
    urgency: str  # low, moderate, high, emergency
    rationale: str
    specific_actions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TreatmentDecision:
    """Treatment recommendation decision."""

    recommendations: list[Any]  # List of treatment recommendations
    considers_previous_response: bool
    rationale: str
    alternative_options: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class TriageResult:
    """Emergency triage result."""

    urgency_level: str
    recommended_action: str
    estimated_response_time: str
    risk_factors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PersonalizedAssessment:
    """Assessment using personalized baselines."""

    clinical_state: str
    deviation_from_baseline: float
    uses_personalized_thresholds: bool
    clinical_note: str
    individualized_recommendations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntegratedAssessment:
    """Integration of multiple assessment domains."""

    overall_clinical_picture: str
    priority_actions: list[str]
    integrated_recommendations: list[dict[str, Any]]
    coordination_needed: bool = False


@dataclass(frozen=True)
class ClinicalPathway:
    """Recommended clinical pathway."""

    pathway_name: str
    recommended_interventions: list[str]
    expected_timeline_weeks: int
    key_milestones: list[str] = field(default_factory=list)


class ClinicalDecisionEngine:
    """
    Orchestrates clinical decision-making by coordinating multiple specialized services.

    This engine provides high-level clinical decision support by integrating
    assessments from various domains and applying clinical logic to generate
    comprehensive recommendations.
    """

    def __init__(self, config: ClinicalThresholdsConfig):
        """
        Initialize the decision engine with all necessary services.

        Args:
            config: Clinical thresholds configuration
        """
        self.config = config

        # Initialize all specialized services
        self.episode_interpreter = EpisodeInterpreter(config)
        self.biomarker_interpreter = BiomarkerInterpreter(config)
        self.treatment_recommender = TreatmentRecommender(config)
        self.dsm5_evaluator = DSM5CriteriaEvaluator(config)
        self.risk_assessor = RiskLevelAssessor(config)
        self.early_warning_detector = EarlyWarningDetector(config)

    def make_clinical_assessment(
        self,
        mood_scores: dict[str, float],
        biomarkers: dict[str, Any] | None = None,
        clinical_context: dict[str, Any] | None = None,
    ) -> ClinicalAssessment:
        """
        Make a comprehensive clinical assessment.

        Args:
            mood_scores: Dictionary of mood assessment scores (phq, asrm, etc.)
            biomarkers: Dictionary of biomarker values
            clinical_context: Additional clinical context

        Returns:
            Comprehensive clinical assessment
        """
        if biomarkers is None:
            biomarkers = {}
        if clinical_context is None:
            clinical_context = {}

        # Track data completeness
        data_fields = ["phq", "asrm", "sleep_hours", "activity_steps", "symptom_days"]
        available_fields = len(
            [
                f
                for f in data_fields
                if f in mood_scores or f in biomarkers or f in clinical_context
            ]
        )
        data_completeness = available_fields / len(data_fields)

        limitations = []
        if data_completeness < 0.8:
            limitations.append("Limited data available for assessment")

        # Determine primary presentation
        primary_diagnosis = "none"
        risk_level = "low"
        confidence = data_completeness * 0.9  # Base confidence on data completeness

        # Check for depression
        if "phq" in mood_scores:
            phq_score = mood_scores["phq"]
            depression_risk = self.risk_assessor.assess_depression_risk(
                phq_score=phq_score,
                sleep_hours=biomarkers.get("sleep_hours"),
                activity_steps=biomarkers.get("activity_steps"),
                suicidal_ideation=clinical_context.get("suicidal_ideation", False),
            )

            if depression_risk.severity_score >= 10:
                primary_diagnosis = "depressive_episode"
                risk_level = depression_risk.risk_level
                confidence = depression_risk.confidence * data_completeness

        # Check for mania
        if "asrm" in mood_scores:
            asrm_score = mood_scores["asrm"]
            mania_risk = self.risk_assessor.assess_mania_risk(
                asrm_score=asrm_score,
                sleep_hours=biomarkers.get("sleep_hours"),
                activity_steps=biomarkers.get("activity_steps"),
                psychotic_features=clinical_context.get("psychotic_features", False),
            )

            if mania_risk.severity_score >= 6:
                if primary_diagnosis == "depressive_episode":
                    primary_diagnosis = "mixed_episode"
                    # Mixed states are inherently higher risk
                    risk_level = "high" if risk_level == "moderate" else risk_level
                else:
                    primary_diagnosis = "manic_episode"
                # For mixed states, take the higher risk level
                risk_levels = ["low", "moderate", "high", "critical"]
                if risk_levels.index(mania_risk.risk_level) > risk_levels.index(
                    risk_level
                ):
                    risk_level = mania_risk.risk_level

        # Check DSM-5 criteria if we have symptom duration
        meets_dsm5 = False
        if "symptom_days" in clinical_context and primary_diagnosis != "none":
            dsm5_result = self.dsm5_evaluator.evaluate_episode_duration(
                episode_type=primary_diagnosis.replace("_episode", ""),
                symptom_days=clinical_context["symptom_days"],
                hospitalization=clinical_context.get("hospitalization", False),
            )
            meets_dsm5 = dsm5_result.meets_criteria

        # Get treatment recommendations for significant episodes
        treatment_options = []
        if (
            risk_level in ["moderate", "high", "critical"]
            and primary_diagnosis != "none"
        ):
            # Map diagnosis to episode type for treatment recommender
            episode_type_map = {
                "depressive_episode": "depressive",
                "manic_episode": "manic",
                "mixed_episode": "depressive_with_mixed_features",
            }
            episode_type = episode_type_map.get(primary_diagnosis, primary_diagnosis)

            recommendations = self.treatment_recommender.get_recommendations(
                episode_type=episode_type,
                severity=risk_level,
                current_medications=clinical_context.get("current_medications", []),
            )
            treatment_options = [
                {
                    "medication": rec.medication,
                    "evidence_level": rec.evidence_level,
                    "description": rec.description,
                }
                for rec in recommendations
            ]

        # Calculate complexity score for mixed states
        complexity_score = 0.0
        if "mixed" in primary_diagnosis:
            complexity_score = 0.8
        elif len([s for s in mood_scores.values() if s > 5]) > 1:
            complexity_score = 0.6

        # Generate clinical summary
        summary_parts = []
        summary_parts.append(
            f"Clinical assessment indicates {primary_diagnosis.replace('_', ' ')}"
        )
        summary_parts.append(f"with {risk_level} risk level")

        if meets_dsm5:
            summary_parts.append("meeting DSM-5 criteria")

        if clinical_context.get("psychotic_features"):
            summary_parts.append("with psychotic features")

        clinical_summary = ". ".join(summary_parts) + "."

        return ClinicalAssessment(
            primary_diagnosis=primary_diagnosis,
            risk_level=risk_level,
            meets_dsm5_criteria=meets_dsm5,
            confidence=confidence,
            clinical_summary=clinical_summary,
            treatment_options=treatment_options,
            requires_immediate_intervention=(risk_level == "critical"),
            complexity_score=complexity_score,
            data_completeness=data_completeness,
            limitations=limitations,
        )

    def make_longitudinal_assessment(
        self,
        current_scores: dict[str, float],
        current_biomarkers: dict[str, Any],
        historical_assessments: list[dict[str, Any]],
    ) -> LongitudinalAssessment:
        """
        Make assessment incorporating historical data.

        Args:
            current_scores: Current mood scores
            current_biomarkers: Current biomarker values
            historical_assessments: List of previous assessments

        Returns:
            Longitudinal assessment with trajectory analysis
        """
        # Extract historical scores
        historical_risks = []
        for assessment in historical_assessments:
            historical_risks.append(
                {
                    "date": (
                        assessment.get("date", "").isoformat()
                        if isinstance(assessment.get("date"), datetime)
                        else ""
                    ),
                    "risk_level": assessment.get("risk_level", "low"),
                    "score": assessment.get("phq_score", 0)
                    + assessment.get("asrm_score", 0),
                }
            )

        # Current combined score
        current_score = current_scores.get("phq", 0) + current_scores.get("asrm", 0)

        # Analyze trajectory
        trajectory_result = self.risk_assessor.analyze_risk_trajectory(
            historical_risks=historical_risks,
            current_risk_level="high" if current_score > 15 else "moderate",
            current_score=current_score,
        )

        # Detect patterns
        pattern = "stable"
        if trajectory_result.trend == "worsening":
            if current_scores.get("phq", 0) > current_scores.get("asrm", 0):
                pattern = "escalating_depression"
            else:
                pattern = "escalating_mania"
        elif trajectory_result.trend == "improving":
            pattern = "responding_to_treatment"

        # Risk projection
        if trajectory_result.trend == "worsening":
            risk_projection = "high_risk_of_full_episode"
        elif trajectory_result.trend == "stable":
            risk_projection = "continued_monitoring_needed"
        else:
            risk_projection = "favorable_prognosis"

        # Clinical note
        clinical_note = (
            f"Longitudinal analysis shows {trajectory_result.trend} trajectory. "
        )
        if trajectory_result.velocity > 0:
            clinical_note += "Symptoms showing increasing severity over time. "
        clinical_note += f"Pattern suggests {pattern.replace('_', ' ')}."

        return LongitudinalAssessment(
            trajectory=trajectory_result.trend,
            pattern_detected=pattern,
            clinical_note=clinical_note,
            risk_projection=risk_projection,
            confidence_in_trend=0.8 if len(historical_assessments) >= 3 else 0.6,
        )

    def evaluate_intervention_need(
        self,
        warning_indicators: dict[str, float],
        current_risk: str,
        patient_history: dict[str, Any] | None = None,
    ) -> InterventionDecision:
        """
        Evaluate need for clinical intervention.

        Args:
            warning_indicators: Early warning sign indicators
            current_risk: Current risk level
            patient_history: Patient's clinical history

        Returns:
            Intervention decision
        """
        if patient_history is None:
            patient_history = {}

        # Detect early warnings
        warning_result = self.early_warning_detector.detect_warnings(
            sleep_change_hours=warning_indicators.get("sleep_change", 0),
            activity_change_percent=warning_indicators.get("activity_change", 0),
            circadian_shift_hours=warning_indicators.get("circadian_shift", 0),
            consecutive_days=int(warning_indicators.get("consecutive_days", 1)),
        )

        # Determine intervention need
        recommend_intervention = False
        intervention_type = "monitoring"
        urgency = "low"
        specific_actions = []

        if warning_result.trigger_intervention:
            recommend_intervention = True
            intervention_type = "preventive"
            urgency = "high" if warning_result.urgency_level == "high" else "moderate"
            specific_actions.extend(
                [
                    "Schedule urgent clinical review",
                    "Consider medication adjustment",
                    "Increase monitoring frequency",
                ]
            )
        elif warning_result.depression_warning or warning_result.mania_warning:
            recommend_intervention = True
            intervention_type = "preventive"
            urgency = "moderate"
            specific_actions.extend(
                [
                    "Schedule clinical review within 1 week",
                    "Review current treatment plan",
                    "Patient education about early signs",
                ]
            )

        # Consider patient history
        if patient_history.get("previous_episodes", 0) >= 2:
            if urgency == "low" and len(warning_result.warning_signs) > 0:
                urgency = "moderate"
                recommend_intervention = True
                specific_actions.append("High-risk patient - maintain vigilance")

        # Generate rationale
        rationale = f"Based on early warning signs detected: {', '.join(warning_result.warning_signs)}. "
        if patient_history.get("previous_episodes"):
            rationale += f"Patient has history of {patient_history['previous_episodes']} previous episodes. "
        rationale += warning_result.clinical_summary

        return InterventionDecision(
            recommend_intervention=recommend_intervention,
            intervention_type=intervention_type,
            urgency=urgency,
            rationale=rationale,
            specific_actions=specific_actions,
        )

    def make_treatment_decision(
        self,
        diagnosis: str,
        severity: str,
        patient_factors: dict[str, Any] | None = None,
    ) -> TreatmentDecision:
        """
        Make treatment decisions considering patient factors.

        Args:
            diagnosis: Primary diagnosis
            severity: Severity level
            patient_factors: Patient-specific factors

        Returns:
            Treatment decision with recommendations
        """
        if patient_factors is None:
            patient_factors = {}

        # Map diagnosis to proper episode type
        episode_type_map = {
            "manic_episode": "manic",
            "depressive_episode": "depressive",
            "mixed_episode": "manic_with_mixed_features",
        }
        episode_type = episode_type_map.get(diagnosis, diagnosis)

        # Get base recommendations
        recommendations = self.treatment_recommender.get_recommendations(
            episode_type=episode_type,
            severity=severity,
            current_medications=patient_factors.get("current_medications", []),
            contraindications=patient_factors.get("contraindications", []),
        )

        # Filter out contraindicated medications
        contraindications = patient_factors.get("contraindications", [])
        filtered_recommendations = [
            rec for rec in recommendations if rec.medication not in contraindications
        ]

        # Consider previous response
        considers_previous = False
        previous_response = patient_factors.get("previous_response", {})
        if previous_response:
            considers_previous = True

            # Sort by previous response
            def response_score(rec: Any) -> int:
                response = previous_response.get(rec.medication, "unknown")
                scores = {"good": 3, "partial": 2, "poor": 1, "unknown": 0}
                return scores.get(response, 0)

            filtered_recommendations.sort(key=response_score, reverse=True)

        # Generate rationale
        rationale = f"Treatment recommendations for {severity} {diagnosis}. "
        if contraindications:
            rationale += (
                f"Avoided {', '.join(contraindications)} due to contraindications. "
            )
        if considers_previous:
            rationale += "Prioritized based on previous treatment response."

        return TreatmentDecision(
            recommendations=filtered_recommendations,
            considers_previous_response=considers_previous,
            rationale=rationale,
            alternative_options=[],  # Could add alternative treatments here
        )

    def triage_urgency(
        self,
        indicators: dict[str, Any],
    ) -> TriageResult:
        """
        Triage clinical urgency for emergency cases.

        Args:
            indicators: Clinical indicators

        Returns:
            Triage result with urgency assessment
        """
        urgency_level = "routine"
        recommended_action = "standard_clinical_review"
        estimated_response_time = "within_1_week"
        risk_factors = []

        # Check critical indicators
        if indicators.get("suicidal_ideation"):
            urgency_level = "emergency"
            recommended_action = "immediate_psychiatric_evaluation"
            estimated_response_time = "within_24_hours"
            risk_factors.append("suicidal ideation")

        if indicators.get("phq_score", 0) >= 20:
            if urgency_level != "emergency":
                urgency_level = "urgent"
            risk_factors.append("severe depression")

        if indicators.get("previous_attempts", 0) > 0:
            urgency_level = "emergency"
            risk_factors.append("previous suicide attempts")

        if indicators.get("sleep_hours", 8) <= 2:
            if urgency_level == "routine":
                urgency_level = "urgent"
            risk_factors.append("severe sleep deprivation")

        return TriageResult(
            urgency_level=urgency_level,
            recommended_action=recommended_action,
            estimated_response_time=estimated_response_time,
            risk_factors=risk_factors,
        )

    def make_personalized_assessment(
        self,
        current_state: dict[str, float],
        individual_baseline: dict[str, Any],
    ) -> PersonalizedAssessment:
        """
        Make assessment using personalized baselines.

        Args:
            current_state: Current clinical measurements
            individual_baseline: Individual's baseline values

        Returns:
            Personalized assessment
        """
        # Calculate deviations from baseline
        sleep_deviation = abs(
            current_state.get("sleep_hours", 7)
            - individual_baseline.get("typical_sleep", 7)
        )

        # Apply personalized thresholds
        uses_personalized = bool(individual_baseline.get("sensitivity_profile"))

        clinical_state = "stable"
        if current_state.get("phq", 0) > 10 or current_state.get("asrm", 0) > 6:
            clinical_state = "symptomatic"

        clinical_note = "Assessment using "
        if uses_personalized:
            clinical_note += "personalized thresholds based on individual baseline. "
        else:
            clinical_note += "standard population thresholds. "

        clinical_note += f"Sleep deviation from baseline: {sleep_deviation:.1f} hours."

        individualized_recommendations = []
        if sleep_deviation > 1.5:
            individualized_recommendations.append(
                "Focus on sleep hygiene and regulation"
            )

        if individual_baseline.get("sensitivity_profile", {}).get("sleep_sensitive"):
            individualized_recommendations.append(
                "Monitor sleep patterns closely - individual is sleep-sensitive"
            )

        return PersonalizedAssessment(
            clinical_state=clinical_state,
            deviation_from_baseline=sleep_deviation,
            uses_personalized_thresholds=uses_personalized,
            clinical_note=clinical_note,
            individualized_recommendations=individualized_recommendations,
        )

    def integrate_assessments(
        self,
        mood_assessment: dict[str, Any],
        biomarker_assessment: dict[str, Any],
        early_warning_assessment: dict[str, Any],
        treatment_assessment: dict[str, Any],
    ) -> IntegratedAssessment:
        """
        Integrate multiple assessment domains.

        Args:
            mood_assessment: Mood episode assessment
            biomarker_assessment: Biomarker assessment
            early_warning_assessment: Early warning assessment
            treatment_assessment: Treatment effectiveness assessment

        Returns:
            Integrated clinical assessment
        """
        # Determine overall clinical picture
        if (
            mood_assessment.get("risk_level") == "moderate"
            and biomarker_assessment.get("instability") == "high"
            and not treatment_assessment.get("current_effective")
        ):
            overall_picture = "unstable_with_treatment_resistance"
        elif early_warning_assessment.get("warnings_detected"):
            overall_picture = "prodromal_phase"
        else:
            overall_picture = "stable_on_current_regimen"

        # Prioritize actions
        priority_actions = []
        if treatment_assessment.get("needs_adjustment"):
            priority_actions.append("medication_optimization")
        if biomarker_assessment.get("circadian_disruption"):
            priority_actions.append("circadian_rhythm_intervention")
        if early_warning_assessment.get("warnings_detected"):
            priority_actions.append("intensify_monitoring")

        # Generate integrated recommendations
        integrated_recommendations = []
        integrated_recommendations.append(
            {
                "domain": "pharmacological",
                "action": "Review and optimize medication regimen",
                "priority": (
                    "high"
                    if not treatment_assessment.get("current_effective")
                    else "medium"
                ),
            }
        )
        integrated_recommendations.append(
            {
                "domain": "behavioral",
                "action": "Implement sleep hygiene protocol",
                "priority": (
                    "high"
                    if biomarker_assessment.get("circadian_disruption")
                    else "low"
                ),
            }
        )
        integrated_recommendations.append(
            {
                "domain": "monitoring",
                "action": "Increase assessment frequency",
                "priority": (
                    "high"
                    if early_warning_assessment.get("warnings_detected")
                    else "low"
                ),
            }
        )

        return IntegratedAssessment(
            overall_clinical_picture=overall_picture,
            priority_actions=priority_actions,
            integrated_recommendations=integrated_recommendations,
            coordination_needed=len(priority_actions) > 2,
        )

    def select_clinical_pathway(
        self,
        presentation: dict[str, Any],
    ) -> ClinicalPathway:
        """
        Select appropriate clinical pathway based on presentation.

        Args:
            presentation: Clinical presentation details

        Returns:
            Recommended clinical pathway
        """
        # Determine pathway based on presentation
        primary_symptoms = presentation.get("primary_symptoms", "")
        treatment_history = presentation.get("treatment_history", "")

        if "multiple_failed" in treatment_history and primary_symptoms == "depression":
            pathway_name = "treatment_resistant_depression"
            interventions = [
                "medication_augmentation",
                "psychotherapy",
                "consider_neuromodulation",
                "comprehensive_assessment",
            ]
            timeline_weeks = 12
        elif (
            presentation.get("comorbidities") and len(presentation["comorbidities"]) > 1
        ):
            pathway_name = "complex_comorbid_presentation"
            interventions = [
                "integrated_treatment_plan",
                "address_comorbidities",
                "care_coordination",
            ]
            timeline_weeks = 16
        else:
            pathway_name = "standard_mood_disorder_treatment"
            interventions = [
                "medication_initiation",
                "psychoeducation",
                "regular_monitoring",
            ]
            timeline_weeks = 8

        milestones = [
            "Initial assessment complete",
            "Treatment initiated",
            "First follow-up",
            "Symptom reassessment",
            "Treatment optimization",
        ]

        return ClinicalPathway(
            pathway_name=pathway_name,
            recommended_interventions=interventions,
            expected_timeline_weeks=timeline_weeks,
            key_milestones=milestones,
        )
