"""
Episode Interpreter Service

Responsible for interpreting mood episodes (depression, mania, mixed states)
based on clinical scores and biomarkers.

Design Patterns:
- Strategy Pattern: Different interpretation strategies for each episode type
- Dependency Injection: Configuration injected via constructor
- Single Responsibility: Only handles episode interpretation
"""

from dataclasses import dataclass
from typing import Any, Protocol

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)


@dataclass
class EpisodeInterpretation:
    """Result of episode interpretation."""
    risk_level: str  # "low", "moderate", "high", "critical"
    episode_type: str  # "none", "depressive", "manic", "hypomanic", etc.
    dsm5_criteria_met: bool
    clinical_summary: str
    confidence: float = 0.85


class EpisodeInterpreterProtocol(Protocol):
    """Protocol for episode interpretation strategies."""

    def interpret(self, **kwargs: Any) -> EpisodeInterpretation:
        """Interpret episode based on inputs."""
        ...


class EpisodeInterpreter:
    """
    Interprets mood episodes based on clinical scores and biomarkers.

    This is extracted from the original ClinicalInterpreter to follow
    Single Responsibility Principle.
    """

    def __init__(self, config: ClinicalThresholdsConfig):
        """
        Initialize with clinical thresholds configuration.

        Args:
            config: Clinical thresholds configuration
        """
        self.config = config

    def interpret_depression(
        self,
        phq_score: float,
        sleep_hours: float,
        activity_steps: int,
        suicidal_ideation: bool = False,
    ) -> EpisodeInterpretation:
        """
        Interpret depression based on PHQ-8/9 scores and biomarkers.

        Uses configuration-driven thresholds instead of hard-coded values.
        """
        cutoffs = self.config.depression.phq_cutoffs

        # Determine base risk level and episode type
        if phq_score <= cutoffs.none.max:
            risk_level = "low"
            episode_type = "none"
            summary = "Depression screening within normal range."
        elif phq_score <= cutoffs.mild.max:
            risk_level = "low"
            episode_type = "none"
            summary = "Mild depressive symptoms below clinical threshold."
        elif phq_score <= cutoffs.moderate.max:
            risk_level = "moderate"
            episode_type = "depressive"
            summary = "Moderate depression detected requiring clinical attention."
        elif phq_score <= cutoffs.moderately_severe.max:
            risk_level = "high"
            episode_type = "depressive"
            summary = "Moderately severe depression requiring prompt intervention."
        else:
            risk_level = "critical"
            episode_type = "depressive"
            summary = "Severe depression requiring immediate intervention."

        # Apply biomarker modifiers
        if sleep_hours > self.config.depression.sleep_hours.hypersomnia_threshold:
            summary += " Hypersomnia pattern detected."
            if risk_level == "low":
                risk_level = "moderate"

        if activity_steps < self.config.depression.activity_steps.severe_reduction:
            if risk_level != "critical":
                summary += " Severe activity reduction noted."

        # Critical override for suicidal ideation
        if suicidal_ideation:
            risk_level = "critical"
            summary += " Suicidal ideation present - urgent assessment needed."

        return EpisodeInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            dsm5_criteria_met=(phq_score >= self.config.depression.phq_cutoffs.moderate.min),
            clinical_summary=summary,
        )

    def interpret_mania(
        self,
        asrm_score: float,
        sleep_hours: float,
        activity_steps: int,
        psychotic_features: bool = False,
    ) -> EpisodeInterpretation:
        """
        Interpret mania/hypomania based on ASRM scores and biomarkers.
        """
        cutoffs = self.config.mania.asrm_cutoffs

        # Determine base risk level and episode type
        if asrm_score <= cutoffs.none.max:
            risk_level = "low"
            episode_type = "none"
            summary = "Mania screening within normal range."
        elif asrm_score <= cutoffs.hypomanic.max:
            risk_level = "moderate"
            episode_type = "hypomanic"
            summary = "Hypomanic symptoms detected requiring monitoring."
        elif asrm_score <= cutoffs.manic_moderate.max:
            risk_level = "high"
            episode_type = "hypomanic"
            summary = "Significant hypomanic symptoms requiring intervention."
        else:
            risk_level = "critical"
            episode_type = "manic"
            summary = "Severe manic symptoms requiring immediate intervention."

        # Critical sleep indicator
        if sleep_hours < self.config.mania.sleep_hours.critical_threshold:
            risk_level = "critical"
            episode_type = "manic"
            summary += " Critical sleep reduction indicates mania."

        # Psychotic features override
        if psychotic_features:
            risk_level = "critical"
            episode_type = "manic"
            summary = "Manic episode with psychotic features - immediate hospitalization may be required."

        # Activity elevation modifier
        if activity_steps > self.config.mania.activity_steps.extreme_threshold:
            summary += " Significantly elevated activity level."
            if risk_level == "moderate":
                risk_level = "high"

        return EpisodeInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            dsm5_criteria_met=(asrm_score >= self.config.mania.asrm_cutoffs.hypomanic.min),
            clinical_summary=summary,
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
        decreased_sleep: bool = False,
        depressed_mood: bool = False,
        anhedonia: bool = False,
        guilt: bool = False,
    ) -> EpisodeInterpretation:
        """
        Detect and interpret mixed features based on DSM-5 criteria.

        Mixed features require:
        - Full criteria for one pole (depression or mania)
        - ≥3 symptoms from opposite pole
        """
        # Count opposite pole symptoms
        manic_symptoms = sum([
            racing_thoughts,
            increased_energy,
            decreased_sleep or sleep_hours < self.config.depression.sleep_hours.normal_min
        ])

        depressive_symptoms = sum([depressed_mood, anhedonia, guilt])

        # Check for mixed features
        depression_threshold = self.config.depression.phq_cutoffs.moderate.min
        mania_threshold = self.config.mania.asrm_cutoffs.hypomanic.min
        min_symptoms = self.config.mixed_features.minimum_opposite_symptoms

        if phq_score >= depression_threshold and manic_symptoms >= min_symptoms:
            return EpisodeInterpretation(
                risk_level="high",
                episode_type="depressive_with_mixed_features",
                dsm5_criteria_met=True,
                clinical_summary="Major depressive episode with mixed features detected.",
                confidence=0.80,  # Lower confidence for mixed states
            )
        elif asrm_score >= mania_threshold and depressive_symptoms >= min_symptoms:
            return EpisodeInterpretation(
                risk_level="high",
                episode_type="manic_with_mixed_features",
                dsm5_criteria_met=True,
                clinical_summary="Manic episode with mixed features detected.",
                confidence=0.80,
            )

        # No mixed features - interpret as pure episode
        if phq_score >= asrm_score:
            return self.interpret_depression(phq_score, sleep_hours, activity_steps)
        else:
            return self.interpret_mania(asrm_score, sleep_hours, activity_steps)
