"""
Clinical Interpreter Service

Interprets ML predictions and digital biomarkers into clinical recommendations
based on DSM-5 criteria and evidence-based guidelines from the Clinical Dossier.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
    load_clinical_thresholds,
)


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
    evidence_level: str  # first-line, second-line
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


class ClinicalInterpreter:
    """
    Interprets ML predictions and biomarkers into clinical recommendations.

    Based on evidence from:
    - DSM-5 diagnostic criteria
    - CANMAT guidelines
    - Harvard/Brigham Fitbit study
    - Seoul National XGBoost study
    - VA clinical guidelines
    """

    def interpret_depression_score(
        self,
        phq_score: float,
        sleep_hours: float,
        activity_steps: int,
        suicidal_ideation: bool = False,
    ) -> ClinicalInterpretation:
        """
        Interpret depression scores based on PHQ-8/9 and biomarkers.

        Thresholds from Clinical Dossier:
        - PHQ-8/9 ≥ 10: Probable depression
        - PHQ-9 5-9: Mild
        - PHQ-9 10-14: Moderate
        - PHQ-9 15-19: Moderately severe
        - PHQ-9 20-27: Severe
        """
        # Determine risk level
        if phq_score < 5:
            risk_level = RiskLevel.LOW
            episode_type = EpisodeType.NONE
            summary = "Depression screening within normal range."
        elif phq_score < 10:
            risk_level = RiskLevel.LOW
            episode_type = EpisodeType.NONE
            summary = "Mild depressive symptoms below clinical threshold."
        elif phq_score < 15:
            risk_level = RiskLevel.MODERATE
            episode_type = EpisodeType.DEPRESSIVE
            summary = "Moderate depression detected requiring clinical attention."
        elif phq_score < 20:
            risk_level = RiskLevel.HIGH
            episode_type = EpisodeType.DEPRESSIVE
            summary = "Moderately severe depression requiring prompt intervention."
        else:
            risk_level = RiskLevel.CRITICAL
            episode_type = EpisodeType.DEPRESSIVE
            summary = "Severe depression requiring immediate intervention."

        # Check biomarker red flags
        if sleep_hours > 12:  # Critical long sleep
            summary += " Hypersomnia pattern detected."
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MODERATE

        if activity_steps < 2000 and risk_level != RiskLevel.CRITICAL:
            summary += " Severe activity reduction noted."

        if suicidal_ideation:
            risk_level = RiskLevel.CRITICAL
            summary += " Suicidal ideation present - urgent assessment needed."

        # Generate recommendations
        recommendations = self._get_depression_recommendations(risk_level, phq_score)

        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=0.85,  # Base confidence
            clinical_summary=summary,
            recommendations=recommendations,
            dsm5_criteria_met=(phq_score >= 10),
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
        Interpret mania/hypomania scores based on ASRM and biomarkers.

        Thresholds from Clinical Dossier:
        - ASRM ≥ 6: Probable manic/hypomanic episode
        - Sleep < 3 hours: Critical indicator
        - Activity > 15,000 steps: High activity marker
        """
        # Determine risk level
        if asrm_score < 6:
            risk_level = RiskLevel.LOW
            episode_type = EpisodeType.NONE
            summary = "Mania screening within normal range."
        elif asrm_score <= 10:
            risk_level = RiskLevel.MODERATE
            episode_type = EpisodeType.HYPOMANIC
            summary = "Hypomanic symptoms detected requiring monitoring."
        elif asrm_score <= 15:
            risk_level = RiskLevel.HIGH
            episode_type = EpisodeType.HYPOMANIC
            summary = "Significant hypomanic symptoms requiring intervention."
        else:
            risk_level = RiskLevel.CRITICAL
            episode_type = EpisodeType.MANIC
            summary = "Severe manic symptoms requiring immediate intervention."

        # Check critical biomarkers
        if sleep_hours < 3:
            risk_level = RiskLevel.CRITICAL
            episode_type = EpisodeType.MANIC
            summary += " Critical sleep reduction indicates mania."

        if psychotic_features:
            risk_level = RiskLevel.CRITICAL
            episode_type = EpisodeType.MANIC
            summary = "Manic episode with psychotic features - immediate hospitalization may be required."

        # Activity elevation
        if activity_steps > 20000:
            summary += " Significantly elevated activity level."
            if risk_level == RiskLevel.MODERATE:
                risk_level = RiskLevel.HIGH

        recommendations = self._get_mania_recommendations(risk_level, episode_type)

        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=0.85,
            clinical_summary=summary,
            recommendations=recommendations,
            dsm5_criteria_met=(asrm_score >= 6),
            clinical_features={
                "asrm_score": asrm_score,
                "sleep_hours": sleep_hours,
                "activity_steps": activity_steps,
            }
        )

    def interpret_mixed_state(
        self,
        phq_score: float,
        asrm_score: float,
        sleep_hours: float,
        activity_steps: int,
        # Mixed feature symptoms
        racing_thoughts: bool = False,
        increased_energy: bool = False,
        depressed_mood: bool = False,
        anhedonia: bool = False,
        guilt: bool = False,
    ) -> ClinicalInterpretation:
        """
        Detect and interpret mixed features based on DSM-5 criteria.

        Mixed features require:
        - Full criteria for one pole (depression or mania)
        - ≥3 symptoms from opposite pole
        """
        # Count opposite pole symptoms
        manic_symptoms = sum([racing_thoughts, increased_energy, sleep_hours < 6])
        depressive_symptoms = sum([depressed_mood, anhedonia, guilt])

        # Determine primary episode and check for mixed features
        if phq_score >= 10 and manic_symptoms >= 3:
            episode_type = EpisodeType.DEPRESSIVE_MIXED
            risk_level = RiskLevel.HIGH
            summary = "Major depressive episode with mixed features detected."
        elif asrm_score >= 6 and depressive_symptoms >= 3:
            episode_type = EpisodeType.MANIC_MIXED
            risk_level = RiskLevel.HIGH
            summary = "Manic episode with mixed features detected."
        else:
            # Fallback to pure episode interpretation
            if phq_score >= asrm_score:
                return self.interpret_depression_score(phq_score, sleep_hours, activity_steps)
            else:
                return self.interpret_mania_score(asrm_score, sleep_hours, activity_steps)

        # Get specialized mixed state recommendations
        recommendations = self._get_mixed_state_recommendations(episode_type)

        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=0.80,  # Slightly lower for mixed states
            clinical_summary=summary,
            recommendations=recommendations,
            dsm5_criteria_met=True,
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
        - Major Depressive: ≥14 days
        """
        duration_requirements = {
            EpisodeType.MANIC: 7,
            EpisodeType.HYPOMANIC: 4,
            EpisodeType.DEPRESSIVE: 14,
            EpisodeType.DEPRESSIVE_MIXED: 14,
            EpisodeType.MANIC_MIXED: 7,
        }

        required_days = duration_requirements.get(episode_type, 0)

        # Special case: hospitalization for mania
        if episode_type in [EpisodeType.MANIC, EpisodeType.MANIC_MIXED] and hospitalization:
            return DSM5Criteria(
                meets_dsm5_criteria=True,
                clinical_note="Manic episode criteria met due to hospitalization.",
                duration_met=True,
            )

        if symptom_days >= required_days:
            return DSM5Criteria(
                meets_dsm5_criteria=True,
                clinical_note=f"Episode duration meets DSM-5 criteria ({symptom_days} days).",
                duration_met=True,
            )
        else:
            return DSM5Criteria(
                meets_dsm5_criteria=False,
                clinical_note=f"Duration insufficient for {episode_type.value} episode ({symptom_days} days < {required_days} days required)",
                duration_met=False,
            )

    def interpret_sleep_biomarkers(
        self,
        sleep_duration: float,
        sleep_efficiency: float,
        sleep_timing_variance: float,
    ) -> BiomarkerInterpretation:
        """Interpret sleep-related digital biomarkers."""
        result = BiomarkerInterpretation()

        # Critical short sleep (< 3 hours)
        if sleep_duration < 3:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Critical short sleep duration indicates mania risk")
            result.recommendation_priority = "urgent"

        # Poor sleep efficiency (< 85%)
        if sleep_efficiency < 0.85:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Poor sleep efficiency")

        # Variable sleep timing (> 2 hours)
        if sleep_timing_variance > 2:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Highly variable sleep schedule")

        return result

    def interpret_activity_biomarkers(
        self,
        daily_steps: int,
        sedentary_hours: float,
        activity_variance: float,
    ) -> BiomarkerInterpretation:
        """Interpret activity-related digital biomarkers."""
        result = BiomarkerInterpretation()

        # High activity (> 15,000 steps)
        if daily_steps > 15000:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Significantly elevated activity level")

        # Very high activity (> 20,000 steps)
        if daily_steps > 20000:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Extreme activity elevation")

        # Low sedentary time
        if sedentary_hours < 4:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Minimal rest periods")

        return result

    def interpret_circadian_biomarkers(
        self,
        circadian_phase_advance: float,
        interdaily_stability: float,
        intradaily_variability: float,
    ) -> BiomarkerInterpretation:
        """Interpret circadian rhythm biomarkers."""
        result = BiomarkerInterpretation()

        # Phase advance > 2 hours (mania risk)
        if circadian_phase_advance > 2:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Significant circadian phase advance")

        # Low interdaily stability (< 0.5)
        if interdaily_stability < 0.5:
            result.mood_instability_risk = "high"
            result.clinical_notes.append("Low circadian rhythm stability")

        # High intradaily variability (> 1)
        if intradaily_variability > 1:
            result.mood_instability_risk = "high"
            result.clinical_notes.append("High circadian fragmentation")

        result.clinical_summary = "Significant circadian disruption detected"
        return result

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

    def get_treatment_recommendations(
        self,
        episode_type: EpisodeType,
        severity: RiskLevel,
        current_medications: list[str],
        contraindications: list[str] | None = None,
        rapid_cycling: bool = False,
    ) -> list[ClinicalRecommendation]:
        """Get evidence-based treatment recommendations."""
        if contraindications is None:
            contraindications = []

        recommendations = []

        if episode_type == EpisodeType.MANIC:
            # First-line for acute mania
            if "lithium" not in [m.lower() for m in current_medications]:
                recommendations.append(ClinicalRecommendation(
                    medication="lithium",
                    evidence_level="first-line",
                    description="First-line mood stabilizer for acute mania",
                ))

            recommendations.append(ClinicalRecommendation(
                medication="quetiapine",
                evidence_level="first-line",
                description="Atypical antipsychotic for acute mania",
            ))

        elif episode_type == EpisodeType.DEPRESSIVE:
            # For bipolar depression - quetiapine is first-line
            recommendations.append(ClinicalRecommendation(
                medication="quetiapine",
                evidence_level="first-line",
                description="First-line for bipolar depression",
            ))

            # Lamotrigine is contraindicated in rapid cycling
            if not rapid_cycling:
                recommendations.append(ClinicalRecommendation(
                    medication="lamotrigine",
                    evidence_level="first-line",
                    description="Mood stabilizer for bipolar depression (not for rapid cycling)",
                ))

        return recommendations

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

        # Missing critical features
        if missing_features:
            adjusted *= 0.9
            limitations.append(f"Missing features: {', '.join(missing_features)}")

        # Determine reliability
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
                velocity=0,
                clinical_note="Insufficient data for trend analysis",
            )

        # Calculate velocity (change per day)
        velocity = (risk_scores[-1] - risk_scores[0]) / len(risk_scores)

        # Determine direction
        if velocity > 0.05:
            direction = "worsening"
            note = f"Risk scores increasing for {episode_type.value}"
        elif velocity < -0.05:
            direction = "improving"
            note = f"Risk scores decreasing for {episode_type.value}"
        else:
            direction = "stable"
            note = "Risk scores remain stable"

        return RiskTrend(
            direction=direction,
            velocity=velocity,
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

    def apply_clinical_rules(
        self,
        diagnosis: str,
        proposed_treatment: str,
        current_medications: list[str],
        mood_state: str,
    ) -> ClinicalDecision:
        """Apply evidence-based clinical decision rules."""
        # Rule: No antidepressant monotherapy in bipolar
        if (diagnosis == "bipolar_disorder" and
            "depressed" in mood_state and
            proposed_treatment.lower() in ["sertraline", "fluoxetine", "escitalopram", "venlafaxine"] and
            not any(med.lower() in ["lithium", "valproate", "lamotrigine", "quetiapine"]
                   for med in current_medications)):
            return ClinicalDecision(
                approved=False,
                rationale="Antidepressant monotherapy is contraindicated in bipolar disorder",
            )

        # Rule: Antidepressant with mood stabilizer is acceptable
        if (diagnosis == "bipolar_disorder" and
            proposed_treatment.lower() in ["sertraline", "fluoxetine", "escitalopram"] and
            any(med.lower() in ["lithium", "valproate", "lamotrigine"]
                for med in current_medications)):
            return ClinicalDecision(
                approved=True,
                rationale="Antidepressant acceptable with mood stabilizer coverage",
            )

        # Default approval
        return ClinicalDecision(
            approved=True,
            rationale="Treatment within clinical guidelines",
        )

    # Private helper methods
    def _get_depression_recommendations(
        self,
        risk_level: RiskLevel,
        phq_score: float
    ) -> list[ClinicalRecommendation]:
        """Get depression-specific treatment recommendations."""
        recommendations = []

        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
            recommendations.append(ClinicalRecommendation(
                medication="quetiapine",
                evidence_level="first-line",
                description="First-line treatment for bipolar depression",
            ))

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append(ClinicalRecommendation(
                medication="urgent psychiatric evaluation",
                evidence_level="first-line",
                description="Urgent assessment required for severe depression",
            ))

        return recommendations

    def _get_mania_recommendations(
        self,
        risk_level: RiskLevel,
        episode_type: EpisodeType,
    ) -> list[ClinicalRecommendation]:
        """Get mania-specific treatment recommendations."""
        recommendations = []

        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH]:
            recommendations.extend([
                ClinicalRecommendation(
                    medication="lithium",
                    evidence_level="first-line",
                    description="First-line mood stabilizer",
                ),
                ClinicalRecommendation(
                    medication="quetiapine",
                    evidence_level="first-line",
                    description="Atypical antipsychotic for acute mania",
                ),
            ])

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append(ClinicalRecommendation(
                medication="urgent hospitalization evaluation",
                evidence_level="first-line",
                description="Immediate safety assessment required",
            ))

        return recommendations

    def _get_mixed_state_recommendations(
        self,
        episode_type: EpisodeType,
    ) -> list[ClinicalRecommendation]:
        """Get mixed state-specific treatment recommendations."""
        recommendations = []

        if episode_type == EpisodeType.DEPRESSIVE_MIXED:
            # Second-line for depression with mixed features
            recommendations.extend([
                ClinicalRecommendation(
                    medication="cariprazine",
                    evidence_level="second-line",
                    description="Cariprazine is effective for depression with mixed features",
                ),
                ClinicalRecommendation(
                    medication="lurasidone",
                    evidence_level="second-line",
                    description="Lurasidone is an alternative for mixed depression",
                ),
            ])
        elif episode_type == EpisodeType.MANIC_MIXED:
            # Second-line for mania with mixed features
            recommendations.extend([
                ClinicalRecommendation(
                    medication="asenapine",
                    evidence_level="second-line",
                    description="Effective for mania with mixed features",
                ),
                ClinicalRecommendation(
                    medication="cariprazine",
                    evidence_level="second-line",
                    description="Covers both manic and depressive symptoms",
                ),
            ])

        return recommendations
