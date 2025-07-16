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
from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
    load_clinical_thresholds,
)
from big_mood_detector.domain.services.episode_interpreter import (
    EpisodeInterpreter,
)
from big_mood_detector.domain.services.treatment_recommender import (
    TreatmentRecommender,
)
from big_mood_detector.domain.services.dsm5_criteria_evaluator import (
    DSM5CriteriaEvaluator,
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
            default_path = Path("config/clinical_thresholds.yaml")
            if default_path.exists():
                config = load_clinical_thresholds(default_path)
            else:
                raise ValueError("No configuration provided and default not found")

        self.config = config

        # Initialize specialized services
        self.episode_interpreter = EpisodeInterpreter(config)
        self.biomarker_interpreter = BiomarkerInterpreter(config)
        self.treatment_recommender = TreatmentRecommender(config)
        self.dsm5_evaluator = DSM5CriteriaEvaluator(config)

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
            }
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
            }
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
            }
        )

    def evaluate_episode_duration(
        self,
        episode_type: EpisodeType,
        symptom_days: int,
        hospitalization: bool = False,
    ) -> DSM5Criteria:
        """
        Evaluate if episode duration meets DSM-5 criteria.

        DSM-5 Duration Requirements:
        - Manic: ≥7 days (or any duration if hospitalization)
        - Hypomanic: ≥4 days
        - Depressive: ≥14 days
        - Mixed: Follows primary episode requirements
        """
        # Get duration requirements from config
        duration_config = self.config.dsm5_duration

        # Check requirements based on episode type
        duration_met = False
        required_days = 0

        if episode_type == EpisodeType.MANIC:
            required_days = duration_config.manic_days
            duration_met = symptom_days >= required_days or hospitalization
        elif episode_type == EpisodeType.HYPOMANIC:
            required_days = duration_config.hypomanic_days
            duration_met = symptom_days >= required_days and not hospitalization
        elif episode_type == EpisodeType.DEPRESSIVE:
            required_days = duration_config.depressive_days
            duration_met = symptom_days >= required_days
        elif "mixed" in episode_type.value:
            # Mixed episodes follow primary pole duration
            if "depressive" in episode_type.value:
                required_days = duration_config.depressive_days
            else:
                required_days = duration_config.manic_days
            duration_met = symptom_days >= required_days or (
                "manic" in episode_type.value and hospitalization
            )

        # Create criteria result
        if duration_met:
            note = f"Episode duration of {symptom_days} days meets DSM-5 criteria"
        else:
            # Format message to match test expectations
            episode_name = episode_type.value.replace("_", " ")
            note = f"Duration insufficient for {episode_name} episode ({symptom_days} days < {required_days} days required)"

        if hospitalization and episode_type == EpisodeType.MANIC:
            note += " (hospitalization overrides duration requirement)"

        return DSM5Criteria(
            meets_dsm5_criteria=duration_met,
            clinical_note=note,
            duration_met=duration_met,
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
        """Detect early warning signs of mood episodes."""
        result = EarlyWarningResult()

        # Depression warnings
        if sleep_increase_hours > 2:
            result.warning_signs.append("Significant sleep increase")
            result.depression_warning = True

        if activity_decrease_percent > 30:
            result.warning_signs.append("Major activity reduction")
            result.depression_warning = True

        if circadian_delay_hours > 1:
            result.warning_signs.append("Circadian phase delay")
            result.depression_warning = True

        # Mania warnings
        if sleep_decrease_hours > 2:
            result.warning_signs.append("Significant sleep reduction")
            result.mania_warning = True

        if activity_increase_percent > 50:
            result.warning_signs.append("Major activity increase")
            result.mania_warning = True

        if speech_rate_increase:
            result.warning_signs.append("Increased speech rate")
            result.mania_warning = True

        # Trigger intervention if multiple signs for consecutive days
        if len(result.warning_signs) >= 3 and consecutive_days >= 3:
            result.trigger_intervention = True

        return result

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
            limitations.append(f"Missing critical features: {', '.join(missing_critical)}")

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
        parts.append(f"Clinical assessment indicates {interpretation.risk_level.value} risk")

        # Episode type
        if interpretation.episode_type != EpisodeType.NONE:
            episode_text = interpretation.episode_type.value.replace('_', ' ')
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
        """Analyze trend in risk scores over time."""
        if len(risk_scores) < 2:
            return RiskTrend(
                direction="stable",
                velocity=0.0,
                clinical_note="Insufficient data for trend analysis",
            )

        # Calculate trend using simple linear regression
        n = len(risk_scores)
        x = list(range(n))

        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(risk_scores) / n

        numerator = sum((x[i] - x_mean) * (risk_scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        # Determine direction
        if slope > 0.1:
            direction = "worsening"
        elif slope < -0.1:
            direction = "improving"
        else:
            direction = "stable"

        # Generate clinical note
        if direction == "worsening":
            note = f"Risk scores showing increasing trend for {episode_type.value} episode"
        elif direction == "improving":
            note = "Risk scores showing decreasing trend, clinical improvement noted"
        else:
            note = f"Risk scores remain stable for {episode_type.value} monitoring"

        return RiskTrend(
            direction=direction,
            velocity=slope,  # Keep sign to indicate direction
            clinical_note=note,
        )

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

