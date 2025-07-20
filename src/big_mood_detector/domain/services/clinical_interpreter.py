"""
Clinical Interpreter Service (Refactored)

Facade that orchestrates the extracted services for clinical interpretation.
This refactored version delegates to specialized services following SRP.

Design Patterns:
- Facade Pattern: Provides simplified interface to subsystems
- Dependency Injection: Services injected via constructor
- Delegation: Delegates to specialized services
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from big_mood_detector.domain.services.biomarker_interpreter import (
    BiomarkerInterpreter,
)
from big_mood_detector.domain.services.clinical_assessment_service import (
    ClinicalAssessment,
    ClinicalAssessmentService,
)
from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
    load_clinical_thresholds,
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
from big_mood_detector.domain.services.intervention_evaluation_service import (
    InterventionDecision,
    InterventionEvaluationService,
)
from big_mood_detector.domain.services.longitudinal_assessment_service import (
    LongitudinalAssessment,
    LongitudinalAssessmentService,
)
from big_mood_detector.domain.services.risk_level_assessor import (
    RiskLevelAssessor,
)
from big_mood_detector.domain.services.treatment_recommender import (
    TreatmentRecommender,
)


# Re-export enums for backward compatibility
class RiskLevel(Enum):
    """Clinical risk stratification levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EpisodeType(Enum):
    """DSM-5 mood episode types."""

    NONE = "none"
    DEPRESSIVE = "depressive"
    MANIC = "manic"
    HYPOMANIC = "hypomanic"
    DEPRESSIVE_MIXED = "depressive_with_mixed_features"
    MANIC_MIXED = "manic_with_mixed_features"


@dataclass
class ClinicalRecommendation:
    """Clinical treatment recommendation."""

    medication: str
    evidence_level: str
    description: str
    contraindications: list[str] = field(default_factory=list)


@dataclass
class DSM5Criteria:
    """DSM-5 episode criteria evaluation."""

    meets_dsm5_criteria: bool
    clinical_note: str
    duration_met: bool = True
    symptom_count_met: bool = True
    functional_impairment: bool = True


@dataclass
class EarlyWarningResult:
    """Early warning detection result."""

    depression_warning: bool = False
    mania_warning: bool = False
    trigger_intervention: bool = False
    warning_signs: list[str] = field(default_factory=list)


@dataclass
class ConfidenceAdjustment:
    """Confidence adjustment based on data quality."""

    adjusted_confidence: float
    reliability: str  # high, medium, low
    limitations: list[str] = field(default_factory=list)


@dataclass
class RiskTrend:
    """Risk trend analysis over time."""

    direction: str  # worsening, stable, improving
    velocity: float
    clinical_note: str


@dataclass
class PersonalizedThresholds:
    """Individualized clinical thresholds."""

    sleep_low: float
    sleep_high: float
    activity_low: float
    activity_high: float


@dataclass
class ClinicalDecision:
    """Clinical decision rule result."""

    approved: bool
    rationale: str


@dataclass
class ClinicalInterpretation:
    """Complete clinical interpretation of mood state."""

    risk_level: RiskLevel
    episode_type: EpisodeType
    confidence: float
    clinical_summary: str
    recommendations: list[ClinicalRecommendation] = field(default_factory=list)
    dsm5_criteria_met: bool = False
    clinical_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class BiomarkerInterpretation:
    """Interpretation of digital biomarkers."""

    mania_risk_factors: int = 0
    depression_risk_factors: int = 0
    clinical_notes: list[str] = field(default_factory=list)
    recommendation_priority: str = "routine"
    mood_instability_risk: str = "low"
    clinical_summary: str = ""


# Re-export dataclasses from new services for backward compatibility


class ClinicalInterpreter:
    """
    Refactored Clinical Interpreter - Facade for clinical interpretation services.

    This class now acts as a facade, delegating to specialized services:
    - EpisodeInterpreter: Interprets mood episodes
    - BiomarkerInterpreter: Interprets digital biomarkers
    - TreatmentRecommender: Provides treatment recommendations

    This follows the Single Responsibility Principle and makes the code
    more maintainable and testable.
    """

    def __init__(self, config: ClinicalThresholdsConfig | None = None):
        """
        Initialize the clinical interpreter with injected services.

        Args:
            config: Clinical thresholds configuration. If None, loads from default path.
        """
        if config is None:
            # Try to use settings if available, otherwise fall back to environment variable
            try:
                from big_mood_detector.infrastructure.settings.config import (
                    get_settings,
                )

                settings = get_settings()
                # Check in the root config directory first, then in data directory
                config_path = Path("config/clinical_thresholds.yaml")
                if not config_path.exists():
                    config_path = (
                        settings.DATA_DIR / "config" / "clinical_thresholds.yaml"
                    )
            except ImportError:
                # Fallback for tests or when settings module is not available
                import os

                config_path = Path(
                    os.environ.get(
                        "CLINICAL_CONFIG_PATH", "config/clinical_thresholds.yaml"
                    )
                )

            if config_path.exists():
                config = load_clinical_thresholds(config_path)
            else:
                raise ValueError(
                    f"No configuration provided and default not found at {config_path}"
                )

        self.config = config

        # Initialize specialized services
        self.episode_interpreter = EpisodeInterpreter(config)
        self.biomarker_interpreter = BiomarkerInterpreter(config)
        self.treatment_recommender = TreatmentRecommender(config)
        self.dsm5_evaluator = DSM5CriteriaEvaluator(config)
        self.risk_assessor = RiskLevelAssessor(config)
        self.early_warning_detector = EarlyWarningDetector(config)

        # Initialize new assessment services
        self.clinical_assessment_service = ClinicalAssessmentService(
            dsm5_evaluator=self.dsm5_evaluator,
            treatment_recommender=self.treatment_recommender,
            config=config,
        )
        self.longitudinal_assessment_service = LongitudinalAssessmentService()
        self.intervention_evaluation_service = InterventionEvaluationService(
            early_warning_detector=self.early_warning_detector
        )

    def interpret_depression_score(
        self,
        phq_score: float,
        sleep_hours: float,
        activity_steps: int,
        suicidal_ideation: bool = False,
    ) -> ClinicalInterpretation:
        """
        Interpret depression scores - delegates to EpisodeInterpreter.
        """
        result = self.episode_interpreter.interpret_depression(
            phq_score=phq_score,
            sleep_hours=sleep_hours,
            activity_steps=activity_steps,
            suicidal_ideation=suicidal_ideation,
        )

        # Convert to legacy format for backward compatibility
        risk_level = RiskLevel(result.risk_level)
        episode_type = EpisodeType(result.episode_type)

        # Get recommendations if needed
        recommendations = []
        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recs = self.treatment_recommender.get_recommendations(
                episode_type=result.episode_type,
                severity=result.risk_level,
                current_medications=[],
            )
            recommendations = [
                ClinicalRecommendation(
                    medication=r.medication,
                    evidence_level=r.evidence_level,
                    description=r.description,
                    contraindications=r.contraindications,
                )
                for r in recs
            ]

        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            recommendations=recommendations,
            dsm5_criteria_met=result.dsm5_criteria_met,
            clinical_features={
                "phq_score": phq_score,
                "sleep_hours": sleep_hours,
                "activity_steps": activity_steps,
            },
        )

    def interpret_mania_score(
        self,
        asrm_score: float,
        sleep_hours: float,
        activity_steps: int,
        psychotic_features: bool = False,
    ) -> ClinicalInterpretation:
        """
        Interpret mania scores - delegates to EpisodeInterpreter.
        """
        result = self.episode_interpreter.interpret_mania(
            asrm_score=asrm_score,
            sleep_hours=sleep_hours,
            activity_steps=activity_steps,
            psychotic_features=psychotic_features,
        )

        # Convert and get recommendations
        risk_level = RiskLevel(result.risk_level)
        episode_type = EpisodeType(result.episode_type)

        recommendations = []
        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recs = self.treatment_recommender.get_recommendations(
                episode_type=result.episode_type,
                severity=result.risk_level,
                current_medications=[],
            )
            recommendations = [
                ClinicalRecommendation(
                    medication=r.medication,
                    evidence_level=r.evidence_level,
                    description=r.description,
                    contraindications=r.contraindications,
                )
                for r in recs
            ]

        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            recommendations=recommendations,
            dsm5_criteria_met=result.dsm5_criteria_met,
            clinical_features={
                "asrm_score": asrm_score,
                "sleep_hours": sleep_hours,
                "activity_steps": activity_steps,
            },
        )

    def interpret_sleep_biomarkers(
        self,
        sleep_duration: float,
        sleep_efficiency: float,
        sleep_timing_variance: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret sleep biomarkers - delegates to BiomarkerInterpreter.
        """
        result = self.biomarker_interpreter.interpret_sleep(
            sleep_duration=sleep_duration,
            sleep_efficiency=sleep_efficiency,
            sleep_timing_variance=sleep_timing_variance,
        )

        # Convert to legacy format
        return BiomarkerInterpretation(
            mania_risk_factors=result.mania_risk_factors,
            depression_risk_factors=result.depression_risk_factors,
            clinical_notes=result.clinical_notes,
            recommendation_priority=result.recommendation_priority,
            mood_instability_risk=result.mood_instability_risk,
            clinical_summary=result.clinical_summary,
        )

    def interpret_activity_biomarkers(
        self,
        daily_steps: int,
        sedentary_hours: float,
        activity_variance: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret activity biomarkers - delegates to BiomarkerInterpreter.
        """
        result = self.biomarker_interpreter.interpret_activity(
            daily_steps=daily_steps,
            sedentary_hours=sedentary_hours,
        )

        return BiomarkerInterpretation(
            mania_risk_factors=result.mania_risk_factors,
            depression_risk_factors=result.depression_risk_factors,
            clinical_notes=result.clinical_notes,
            recommendation_priority=result.recommendation_priority,
            mood_instability_risk=result.mood_instability_risk,
            clinical_summary=result.clinical_summary,
        )

    def interpret_circadian_biomarkers(
        self,
        circadian_phase_advance: float,
        interdaily_stability: float,
        intradaily_variability: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret circadian biomarkers - delegates to BiomarkerInterpreter.
        """
        result = self.biomarker_interpreter.interpret_circadian(
            phase_advance=circadian_phase_advance,
            interdaily_stability=interdaily_stability,
            intradaily_variability=intradaily_variability,
        )

        return BiomarkerInterpretation(
            mania_risk_factors=result.mania_risk_factors,
            depression_risk_factors=result.depression_risk_factors,
            clinical_notes=result.clinical_notes,
            recommendation_priority=result.recommendation_priority,
            mood_instability_risk=result.mood_instability_risk,
            clinical_summary=result.clinical_summary,
        )

    def get_treatment_recommendations(
        self,
        episode_type: EpisodeType,
        severity: RiskLevel,
        current_medications: list[str],
        contraindications: list[str] | None = None,
        rapid_cycling: bool = False,
    ) -> list[ClinicalRecommendation]:
        """
        Get treatment recommendations - delegates to TreatmentRecommender.
        """
        recs = self.treatment_recommender.get_recommendations(
            episode_type=episode_type.value,
            severity=severity.value,
            current_medications=current_medications,
            contraindications=contraindications,
            rapid_cycling=rapid_cycling,
        )

        return [
            ClinicalRecommendation(
                medication=r.medication,
                evidence_level=r.evidence_level,
                description=r.description,
                contraindications=r.contraindications,
            )
            for r in recs
        ]

    def apply_clinical_rules(
        self,
        diagnosis: str,
        proposed_treatment: str,
        current_medications: list[str],
        mood_state: str,
    ) -> ClinicalDecision:
        """
        Apply clinical rules - delegates to TreatmentRecommender.
        """
        decision = self.treatment_recommender.apply_clinical_rules(
            diagnosis=diagnosis,
            proposed_treatment=proposed_treatment,
            current_medications=current_medications,
            mood_state=mood_state,
        )

        return ClinicalDecision(
            approved=decision.approved,
            rationale=decision.rationale,
        )

    def interpret_mixed_state(
        self,
        phq_score: float,
        asrm_score: float,
        sleep_hours: float,
        activity_steps: int,
        racing_thoughts: bool = False,
        increased_energy: bool = False,
        decreased_sleep: bool = False,
        depressed_mood: bool = False,
        anhedonia: bool = False,
        guilt: bool = False,
    ) -> ClinicalInterpretation:
        """
        Interpret mixed mood state - delegates to EpisodeInterpreter.

        Mixed episodes have symptoms of both depression and mania/hypomania.
        """
        result = self.episode_interpreter.interpret_mixed_state(
            phq_score=phq_score,
            asrm_score=asrm_score,
            sleep_hours=sleep_hours,
            activity_steps=activity_steps,
            racing_thoughts=racing_thoughts,
            increased_energy=increased_energy,
            decreased_sleep=decreased_sleep,
            depressed_mood=depressed_mood,
            anhedonia=anhedonia,
            guilt=guilt,
        )

        # Convert to legacy format for backward compatibility
        risk_level = RiskLevel(result.risk_level)
        episode_type = EpisodeType(result.episode_type)

        # Get recommendations
        recommendations = []
        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recs = self.treatment_recommender.get_recommendations(
                episode_type=result.episode_type,
                severity=result.risk_level,
                current_medications=[],
            )
            recommendations = [
                ClinicalRecommendation(
                    medication=r.medication,
                    evidence_level=r.evidence_level,
                    description=r.description,
                    contraindications=r.contraindications,
                )
                for r in recs
            ]

        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            recommendations=recommendations,
            dsm5_criteria_met=result.dsm5_criteria_met,
            clinical_features={
                "phq_score": phq_score,
                "asrm_score": asrm_score,
                "sleep_hours": sleep_hours,
                "activity_steps": activity_steps,
                "racing_thoughts": racing_thoughts,
                "increased_energy": increased_energy,
            },
        )

    def evaluate_episode_duration(
        self,
        episode_type: EpisodeType,
        symptom_days: int,
        hospitalization: bool = False,
    ) -> DSM5Criteria:
        """
        Evaluate if episode duration meets DSM-5 criteria.

        Delegates to DSM5CriteriaEvaluator for the actual evaluation.
        """
        # Delegate to the specialized evaluator
        result = self.dsm5_evaluator.evaluate_episode_duration(
            episode_type=episode_type.value,
            symptom_days=symptom_days,
            hospitalization=hospitalization,
        )

        # Convert to legacy DSM5Criteria format for backward compatibility
        return DSM5Criteria(
            meets_dsm5_criteria=result.meets_criteria,
            clinical_note=result.clinical_note,
            duration_met=result.duration_met,
            symptom_count_met=True,  # Assume this was evaluated separately
            functional_impairment=True,  # Assume this was evaluated separately
        )

    def detect_early_warnings(
        self,
        sleep_increase_hours: float = 0,
        sleep_decrease_hours: float = 0,
        activity_increase_percent: float = 0,
        activity_decrease_percent: float = 0,
        circadian_delay_hours: float = 0,
        speech_rate_increase: bool = False,
        consecutive_days: int = 1,
    ) -> EarlyWarningResult:
        """Detect early warning signs of mood episodes - delegates to EarlyWarningDetector."""
        # Convert to net changes for the detector
        sleep_change_hours = sleep_increase_hours - sleep_decrease_hours
        activity_change_percent = activity_increase_percent - activity_decrease_percent

        # Delegate to specialized detector
        detector_result = self.early_warning_detector.detect_warnings(
            sleep_change_hours=sleep_change_hours,
            activity_change_percent=activity_change_percent,
            circadian_shift_hours=circadian_delay_hours,
            consecutive_days=consecutive_days,
            speech_pattern_change=speech_rate_increase,
        )

        # Convert back to legacy EarlyWarningResult format
        return EarlyWarningResult(
            depression_warning=detector_result.depression_warning,
            mania_warning=detector_result.mania_warning,
            trigger_intervention=detector_result.trigger_intervention,
            warning_signs=detector_result.warning_signs,
        )

    def adjust_confidence(
        self,
        base_confidence: float,
        data_completeness: float,
        days_of_data: int,
        missing_features: list[str],
    ) -> ConfidenceAdjustment:
        """Adjust prediction confidence based on data quality."""
        adjusted = base_confidence
        limitations = []

        # Data completeness threshold (75%)
        if data_completeness < 0.75:
            adjusted *= 0.8
            limitations.append("Insufficient data completeness")

        # Minimum days requirement (30)
        if days_of_data < 30:
            adjusted *= 0.8
            limitations.append(f"Limited historical data ({days_of_data} days)")

        # Critical missing features
        critical_features = ["sleep_duration", "activity_level", "mood_scores"]
        missing_critical = [f for f in missing_features if f in critical_features]
        if missing_critical:
            adjusted *= 0.7
            limitations.append(
                f"Missing critical features: {', '.join(missing_critical)}"
            )

        # Determine reliability level
        if adjusted >= 0.8:
            reliability = "high"
        elif adjusted >= 0.6:
            reliability = "medium"
        else:
            reliability = "low"

        return ConfidenceAdjustment(
            adjusted_confidence=adjusted,
            reliability=reliability,
            limitations=limitations,
        )

    def generate_clinical_summary(self, interpretation: ClinicalInterpretation) -> str:
        """Generate human-readable clinical summary."""
        parts = []

        # Risk level
        parts.append(
            f"Clinical assessment indicates {interpretation.risk_level.value} risk"
        )

        # Episode type
        if interpretation.episode_type != EpisodeType.NONE:
            episode_text = interpretation.episode_type.value.replace("_", " ")
            if episode_text == "depressive":
                episode_text = "major depressive episode"
            parts.append(f"for {episode_text}")

        # DSM-5 criteria
        if interpretation.dsm5_criteria_met:
            parts.append("meeting DSM-5 criteria")

        # Confidence
        parts.append(f"(confidence: {interpretation.confidence:.0%})")

        # Clinical features
        if "phq_score" in interpretation.clinical_features:
            parts.append(f"PHQ score: {interpretation.clinical_features['phq_score']}")

        return ". ".join(parts) + "."

    def analyze_risk_trend(
        self,
        risk_scores: list[float],
        dates: list[datetime],
        episode_type: EpisodeType,
    ) -> RiskTrend:
        """Analyze trend in risk scores over time - delegates to RiskLevelAssessor."""
        # Convert to format expected by RiskLevelAssessor
        historical_risks = []
        for i in range(len(risk_scores) - 1):
            historical_risks.append(
                {
                    "date": (
                        dates[i].isoformat()
                        if i < len(dates)
                        else f"T-{len(risk_scores)-i}"
                    ),
                    "risk_level": self._score_to_risk_level(risk_scores[i]),
                    "score": risk_scores[i],
                }
            )

        # Get current risk level
        current_risk_level = self._score_to_risk_level(risk_scores[-1])
        current_score = risk_scores[-1]

        # Delegate to RiskLevelAssessor
        trajectory = self.risk_assessor.analyze_risk_trajectory(
            historical_risks=historical_risks,
            current_risk_level=current_risk_level,
            current_score=current_score,
        )

        # Convert back to legacy RiskTrend format
        return RiskTrend(
            direction=trajectory.trend,
            velocity=trajectory.velocity,
            clinical_note=trajectory.clinical_note,
        )

    def _score_to_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level string."""
        # Handle normalized scores (0-1 range) and absolute scores
        if score <= 1.0:
            # Normalized scores
            if score < 0.25:
                return "low"
            elif score < 0.5:
                return "moderate"
            elif score < 0.75:
                return "high"
            else:
                return "critical"
        else:
            # Absolute scores (e.g., PHQ-9, ASRM)
            if score < 5:
                return "low"
            elif score < 10:
                return "moderate"
            elif score < 15:
                return "high"
            else:
                return "critical"

    def calculate_personalized_thresholds(
        self,
        individual_baseline: dict[str, float],
        population_norms: dict[str, float],
    ) -> PersonalizedThresholds:
        """Calculate personalized clinical thresholds."""
        # Sleep thresholds (2 SD from individual mean)
        sleep_mean = individual_baseline["sleep_mean"]
        sleep_std = individual_baseline["sleep_std"]

        # Activity thresholds
        activity_mean = individual_baseline["activity_mean"]
        activity_std = individual_baseline["activity_std"]

        return PersonalizedThresholds(
            sleep_low=max(3, sleep_mean - 2 * sleep_std),  # Never below 3 hours
            sleep_high=min(12, sleep_mean + 2 * sleep_std),  # Never above 12 hours
            activity_low=max(0, activity_mean - 2 * activity_std),
            activity_high=activity_mean + 2 * activity_std,
        )

    # Delegate to new assessment services
    def make_clinical_assessment(
        self,
        mood_scores: dict[str, float],
        biomarkers: dict[str, float],
        clinical_context: dict[str, Any],
    ) -> ClinicalAssessment:
        """Make comprehensive clinical assessment."""
        return self.clinical_assessment_service.make_clinical_assessment(
            mood_scores=mood_scores,
            biomarkers=biomarkers,
            clinical_context=clinical_context,
        )

    def make_longitudinal_assessment(
        self,
        current_scores: dict[str, float],
        current_biomarkers: dict[str, float],
        historical_assessments: list[dict[str, Any]],
    ) -> LongitudinalAssessment:
        """Make assessment incorporating historical data."""
        return self.longitudinal_assessment_service.make_longitudinal_assessment(
            current_scores=current_scores,
            current_biomarkers=current_biomarkers,
            historical_assessments=historical_assessments,
        )

    def evaluate_intervention_need(
        self,
        warning_indicators: dict[str, float],
        current_risk: str,
        patient_history: dict[str, Any],
    ) -> InterventionDecision:
        """Evaluate need for clinical intervention based on early warning signs."""
        return self.intervention_evaluation_service.evaluate_intervention_need(
            warning_indicators=warning_indicators,
            current_risk=current_risk,
            patient_history=patient_history,
        )
