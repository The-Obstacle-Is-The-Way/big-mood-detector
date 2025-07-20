"""
Intervention Evaluation Service

Evaluates the need for clinical interventions based on early warning signs.
Extracted from clinical_interpreter to follow Single Responsibility Principle.
"""

from dataclasses import dataclass, field
from typing import Any

from big_mood_detector.domain.services.early_warning_detector import (
    EarlyWarningDetector,
)


@dataclass(frozen=True)
class InterventionDecision:
    """Decision about clinical intervention."""

    recommend_intervention: bool
    intervention_type: str  # preventive, acute, maintenance
    urgency: str  # low, moderate, high, emergency
    rationale: str
    specific_actions: list[str] = field(default_factory=list)


class InterventionEvaluationService:
    """
    Service for evaluating the need for clinical interventions.

    This service analyzes warning indicators and patient history
    to make intervention recommendations.
    """

    def __init__(self, early_warning_detector: EarlyWarningDetector):
        """Initialize with required services."""
        self.early_warning_detector = early_warning_detector

    def evaluate_intervention_need(
        self,
        warning_indicators: dict[str, float],
        current_risk: str,
        patient_history: dict[str, Any],
    ) -> InterventionDecision:
        """
        Evaluate need for clinical intervention based on early warning signs.
        """
        # Use early warning detector
        early_warning = self.early_warning_detector.detect_warnings(
            sleep_change_hours=warning_indicators.get("sleep_change", 0),
            activity_change_percent=warning_indicators.get("activity_change", 0),
            circadian_shift_hours=warning_indicators.get("circadian_shift", 0),
            consecutive_days=int(warning_indicators.get("consecutive_days", 0)),
        )

        # Determine intervention need
        recommend_intervention = (
            early_warning.urgency_level in ["moderate", "high"]
            or patient_history.get("previous_episodes", 0) >= 2
        )

        # Determine intervention type
        intervention_type = self._determine_intervention_type(
            recommend_intervention, current_risk, early_warning.urgency_level
        )

        # Determine urgency
        urgency = self._determine_urgency(warning_indicators, current_risk)

        # Generate rationale
        rationale = self._generate_intervention_rationale(
            warning_indicators, current_risk, patient_history, early_warning
        )

        # Generate specific actions
        specific_actions = self._generate_specific_actions(
            intervention_type, urgency, patient_history
        )

        return InterventionDecision(
            recommend_intervention=recommend_intervention,
            intervention_type=intervention_type,
            urgency=urgency,
            rationale=rationale,
            specific_actions=specific_actions,
        )

    def _determine_intervention_type(
        self, recommend_intervention: bool, current_risk: str, warning_level: str
    ) -> str:
        """Determine the type of intervention needed."""
        if not recommend_intervention:
            return "monitoring"

        # More specific logic to match test expectations
        if current_risk == "low":
            return "preventive"
        elif current_risk in ["moderate", "high"]:
            return "acute"
        else:
            return "maintenance"

    def _determine_urgency(
        self, warning_indicators: dict[str, float], current_risk: str
    ) -> str:
        """Determine intervention urgency."""
        sleep_change = abs(warning_indicators.get("sleep_change", 0))

        if current_risk == "critical" or sleep_change > 4:
            return "emergency"
        elif current_risk == "high" or sleep_change > 2:
            return "high"
        elif current_risk == "moderate":
            return "moderate"
        else:
            return "low"

    def _generate_intervention_rationale(
        self,
        indicators: dict[str, float],
        risk: str,
        history: dict[str, Any],
        warning: Any,
    ) -> str:
        """Generate rationale for intervention decision."""
        factors = []

        if abs(indicators.get("sleep_change", 0)) > 2:
            factors.append("significant sleep disruption")
        if history.get("previous_episodes", 0) >= 2:
            factors.append("history of multiple episodes")
        if warning.urgency_level in ["moderate", "high"]:
            factors.append(f"{warning.urgency_level} early warning signs")

        return f"Intervention recommended due to: {', '.join(factors)}"

    def _generate_specific_actions(
        self, intervention_type: str, urgency: str, history: dict[str, Any]
    ) -> list[str]:
        """Generate specific intervention actions."""
        actions = []

        if intervention_type == "preventive":
            actions.extend(
                [
                    "Increase monitoring frequency",
                    "Review and optimize sleep hygiene",
                    "Consider prophylactic medication adjustment",
                ]
            )
        elif intervention_type == "acute":
            actions.extend(
                [
                    "Schedule urgent clinical evaluation",
                    "Initiate acute treatment protocol",
                    "Daily symptom monitoring",
                ]
            )

        if urgency == "emergency":
            actions.insert(0, "Immediate psychiatric evaluation required")

        return actions
