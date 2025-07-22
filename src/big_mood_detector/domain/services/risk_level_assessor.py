"""
Risk Level Assessor Service

Determines clinical risk levels based on multiple factors including
mood scores, biomarkers, and clinical history.

Extracted from ClinicalInterpreter following Single Responsibility Principle.

Design Patterns:
- Strategy Pattern: Different assessment strategies for depression/mania/mixed
- Builder Pattern: Builds comprehensive risk assessment from multiple factors
- Value Objects: Immutable risk assessment results
"""

from dataclasses import dataclass, field

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)


@dataclass(frozen=True)
class RiskAssessment:
    """Immutable risk assessment result."""

    risk_level: str  # low, moderate, high, critical
    severity_score: float
    confidence: float
    clinical_rationale: str
    risk_factors: list[str] = field(default_factory=list)
    protective_factors: list[str] = field(default_factory=list)
    requires_immediate_action: bool = False
    biomarker_modifier: float = 1.0
    missing_factors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MixedStateAssessment(RiskAssessment):
    """Risk assessment specific to mixed states."""

    mixed_features_count: int = 0
    dominant_pole: str = "balanced"


@dataclass(frozen=True)
class CompositeRiskAssessment:
    """Composite risk from multiple sources."""

    overall_risk_level: str
    primary_concern: str
    clinical_summary: str
    confidence_adjusted: bool
    component_risks: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskTrajectory:
    """Risk trajectory analysis over time."""

    trend: str  # improving, stable, worsening
    velocity: float
    clinical_note: str
    predicted_trajectory: str


class RiskLevelAssessor:
    """
    Assesses clinical risk levels based on multiple factors.

    This service focuses solely on risk assessment logic,
    extracted from the monolithic ClinicalInterpreter.
    """

    def __init__(self, config: ClinicalThresholdsConfig):
        """
        Initialize with clinical configuration.

        Args:
            config: Clinical thresholds configuration
        """
        self.config = config

    def assess_depression_risk(
        self,
        phq_score: float,
        sleep_hours: float | None = None,
        activity_steps: int | None = None,
        suicidal_ideation: bool = False,
    ) -> RiskAssessment:
        """
        Assess depression risk level.

        Args:
            phq_score: PHQ-9 depression score
            sleep_hours: Average sleep hours
            activity_steps: Daily activity steps
            suicidal_ideation: Presence of suicidal ideation

        Returns:
            Risk assessment result
        """
        risk_factors = []
        missing_factors = []
        confidence = 1.0

        # Determine base risk from PHQ score
        phq_cutoffs = self.config.depression.phq_cutoffs
        if phq_score < phq_cutoffs.none.max:
            risk_level = "low"
            rationale = "Minimal depressive symptoms"
        elif phq_score < phq_cutoffs.mild.max:
            risk_level = "low"
            rationale = "Mild depressive symptoms"
            risk_factors.append("mild PHQ-9 score")
        elif phq_score < phq_cutoffs.moderate.max:
            risk_level = "moderate"
            rationale = "Moderate depression indicated"
            risk_factors.append("moderate PHQ-9 score")
        elif phq_score < phq_cutoffs.severe.max:
            risk_level = "high"
            rationale = "Severe depression indicated"
            risk_factors.append("severe PHQ-9 score")
        else:
            risk_level = "critical"
            rationale = "Very severe depression"
            risk_factors.append("very severe PHQ-9 score")

        # Biomarker modulation
        biomarker_modifier = 1.0

        if sleep_hours is not None:
            sleep_thresholds = self.config.depression.sleep_hours
            if sleep_hours < sleep_thresholds.normal_min:
                risk_factors.append("severe insomnia")
                biomarker_modifier *= 1.2
            elif sleep_hours > sleep_thresholds.hypersomnia_threshold:
                risk_factors.append("hypersomnia")
                biomarker_modifier *= 1.1
        else:
            missing_factors.append("sleep data")
            confidence *= 0.9

        if activity_steps is not None:
            activity_threshold = self.config.depression.activity_steps.severe_reduction
            if activity_steps <= activity_threshold:  # Include boundary
                risk_factors.append("severe hypoactivity")
                biomarker_modifier *= 1.2
        else:
            missing_factors.append("activity data")
            confidence *= 0.9

        # Apply biomarker modifier and potentially upgrade risk
        if biomarker_modifier > 1.3 and risk_level == "moderate":
            risk_level = "high"
            rationale += " with severe biomarker severity"

        # Critical upgrade for suicidal ideation
        if suicidal_ideation:
            risk_level = "critical"
            rationale = "Suicidal ideation present - immediate intervention required"
            risk_factors.append("suicidal ideation")

        # Adjust confidence for missing data
        if missing_factors:
            rationale += f" (limited data: {', '.join(missing_factors)} missing)"

        return RiskAssessment(
            risk_level=risk_level,
            severity_score=phq_score,
            confidence=confidence,
            clinical_rationale=rationale,
            risk_factors=risk_factors,
            requires_immediate_action=(risk_level == "critical"),
            biomarker_modifier=biomarker_modifier,
            missing_factors=missing_factors,
        )

    def assess_mania_risk(
        self,
        asrm_score: float,
        sleep_hours: float | None = None,
        activity_steps: int | None = None,
        psychotic_features: bool = False,
    ) -> RiskAssessment:
        """
        Assess mania/hypomania risk level.

        Args:
            asrm_score: ASRM mania score
            sleep_hours: Average sleep hours
            activity_steps: Daily activity steps
            psychotic_features: Presence of psychotic features

        Returns:
            Risk assessment result
        """
        risk_factors = []

        # Determine base risk from ASRM score
        asrm_cutoffs = self.config.mania.asrm_cutoffs
        if asrm_score < asrm_cutoffs.none.max:
            risk_level = "low"
            rationale = "No manic symptoms detected"
        elif asrm_score < asrm_cutoffs.hypomanic.max:
            risk_level = "moderate"
            rationale = "Hypomanic symptoms present"
            risk_factors.append("elevated ASRM score")
        elif asrm_score < asrm_cutoffs.manic_moderate.max:
            risk_level = "high"
            rationale = "Manic episode likely"
            risk_factors.append("high ASRM score")
        else:
            risk_level = "critical"
            rationale = "Severe manic episode"
            risk_factors.append("very high ASRM score")

        # Biomarker assessment
        if (
            sleep_hours is not None
            and sleep_hours < self.config.mania.sleep_hours.reduced_threshold
        ):
            risk_factors.append("reduced sleep")

        if (
            activity_steps is not None
            and activity_steps >= self.config.mania.activity_steps.elevated_threshold
        ):
            risk_factors.append("hyperactivity")

        # Critical upgrade for psychotic features
        if psychotic_features:
            risk_level = "critical"
            rationale = "Manic episode with psychotic features - immediate intervention required"
            risk_factors.append("psychotic features")

        return RiskAssessment(
            risk_level=risk_level,
            severity_score=asrm_score,
            confidence=0.85,
            clinical_rationale=rationale,
            risk_factors=risk_factors,
            requires_immediate_action=(risk_level == "critical"),
        )

    def assess_mixed_state_risk(
        self,
        phq_score: float,
        asrm_score: float,
        sleep_disturbance: bool = False,
        racing_thoughts: bool = False,
        anhedonia: bool = False,
        increased_energy: bool = False,
        **kwargs: bool,
    ) -> MixedStateAssessment:
        """
        Assess risk for mixed mood states.

        Args:
            phq_score: Depression score
            asrm_score: Mania score
            sleep_disturbance: Sleep issues present
            racing_thoughts: Racing thoughts present
            anhedonia: Loss of pleasure present
            increased_energy: Increased energy present

        Returns:
            Mixed state risk assessment
        """
        risk_factors = []
        mixed_features = 0

        # Count mixed features
        if sleep_disturbance:
            mixed_features += 1
            risk_factors.append("sleep disturbance")
        if racing_thoughts:
            mixed_features += 1
            risk_factors.append("racing thoughts")
        if anhedonia:
            mixed_features += 1
            risk_factors.append("anhedonia")
        if increased_energy:
            mixed_features += 1
            risk_factors.append("increased energy")

        # Both poles elevated
        if (
            phq_score >= self.config.depression.phq_cutoffs.moderate.min
            and asrm_score >= self.config.mania.asrm_cutoffs.hypomanic.min
        ):
            risk_factors.append("concurrent depression and mania symptoms")

        # Determine risk level based on mixed features
        if mixed_features >= 3:
            risk_level = "high"
            rationale = "Mixed mood episode with multiple features"
        elif mixed_features >= 2:
            risk_level = "high"
            rationale = "Mixed features present"
        else:
            risk_level = "moderate"
            rationale = "Possible mixed state"

        # Upgrade to critical if severe
        if (
            phq_score >= self.config.depression.phq_cutoffs.severe.min
            or asrm_score >= self.config.mania.asrm_cutoffs.manic_moderate.min
        ):
            risk_level = "critical"
            rationale = "Severe mixed episode"

        # Determine dominant pole
        if phq_score > asrm_score * 1.5:
            dominant_pole = "depressive"
        elif asrm_score > phq_score * 1.5:
            dominant_pole = "manic"
        else:
            dominant_pole = "balanced"

        return MixedStateAssessment(
            risk_level=risk_level,
            severity_score=max(phq_score, asrm_score),
            confidence=0.8,
            clinical_rationale=rationale,
            risk_factors=risk_factors,
            mixed_features_count=mixed_features,
            dominant_pole=dominant_pole,
        )

    def calculate_composite_risk(
        self,
        factors: dict[str, float],
    ) -> CompositeRiskAssessment:
        """
        Calculate composite risk from multiple factors.

        Args:
            factors: Dictionary of risk factors and their values

        Returns:
            Composite risk assessment
        """
        component_risks = {}

        # Assess each component
        if "depression_score" in factors:
            dep_risk = self._score_to_risk_level(
                factors["depression_score"],
                self.config.depression.phq_cutoffs,
            )
            component_risks["depression"] = dep_risk

        if "mania_score" in factors:
            mania_risk = self._score_to_risk_level(
                factors["mania_score"],
                self.config.mania.asrm_cutoffs,
            )
            component_risks["mania"] = mania_risk

        # Determine overall risk and primary concern
        risk_scores = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
        max_risk = "low"
        primary_concern = "none"

        for domain, risk in component_risks.items():
            if risk_scores.get(risk, 0) > risk_scores.get(max_risk, 0):
                max_risk = risk
                primary_concern = domain

        # Additional risk factors
        if factors.get("medication_adherence", 1.0) < 0.7:
            if max_risk == "moderate":
                max_risk = "high"

        summary = f"Composite assessment shows {max_risk} risk with primary concern: {primary_concern}. "
        summary += "Multiple risk factors present across domains."

        return CompositeRiskAssessment(
            overall_risk_level=max_risk,
            primary_concern=primary_concern,
            clinical_summary=summary,
            confidence_adjusted=True,
            component_risks=component_risks,
        )

    def analyze_risk_trajectory(
        self,
        historical_risks: list[dict],
        current_risk_level: str,
        current_score: float,
    ) -> RiskTrajectory:
        """
        Analyze risk trajectory over time.

        Args:
            historical_risks: List of historical risk assessments
            current_risk_level: Current risk level
            current_score: Current severity score

        Returns:
            Risk trajectory analysis
        """
        if len(historical_risks) < 2:
            return RiskTrajectory(
                trend="stable",
                velocity=0.0,
                clinical_note="Insufficient history for trajectory analysis",
                predicted_trajectory="unknown",
            )

        # Calculate trend
        scores = [r["score"] for r in historical_risks] + [current_score]

        # Simple linear regression for trend
        n = len(scores)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(scores) / n

        numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            velocity = 0.0
        else:
            velocity = numerator / denominator

        # Determine trend
        if velocity > 0.1:
            trend = "worsening"
            note = "Risk scores showing increasing trend"
            predicted = "continued_worsening"
        elif velocity < -0.1:
            trend = "improving"
            note = "Risk scores showing improvement"
            predicted = "continued_improvement"
        else:
            trend = "stable"
            note = "Risk scores relatively stable"
            predicted = "stable"

        return RiskTrajectory(
            trend=trend,
            velocity=velocity,
            clinical_note=note,
            predicted_trajectory=predicted,
        )

    def _score_to_risk_level(self, score: float, cutoffs: object) -> str:
        """Convert score to risk level based on cutoffs."""
        if hasattr(cutoffs, "none") and score < cutoffs.none.max:
            return "low"
        elif hasattr(cutoffs, "mild") and score < cutoffs.mild.max:
            return "low"
        elif hasattr(cutoffs, "moderate") and score < cutoffs.moderate.max:
            return "moderate"
        elif (
            hasattr(cutoffs, "moderately_severe")
            and score < cutoffs.moderately_severe.max
        ):
            return "high"
        elif hasattr(cutoffs, "severe") and score < cutoffs.severe.max:
            return "high"
        elif hasattr(cutoffs, "hypomanic") and score < cutoffs.hypomanic.max:
            return "moderate"
        elif hasattr(cutoffs, "manic_moderate") and score < cutoffs.manic_moderate.max:
            return "high"
        else:
            return "critical"
