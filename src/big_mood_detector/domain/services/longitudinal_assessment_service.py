"""
Longitudinal Assessment Service

Analyzes mood patterns over time to identify trajectories and trends.
Extracted from clinical_interpreter to follow Single Responsibility Principle.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LongitudinalAssessment:
    """Assessment incorporating historical data."""

    trajectory: str  # improving, stable, worsening
    pattern_detected: str
    clinical_note: str
    risk_projection: str
    confidence_in_trend: float


class LongitudinalAssessmentService:
    """
    Service for analyzing mood patterns over time.

    This service examines historical data to identify trajectories,
    patterns, and project future risk.
    """

    def make_longitudinal_assessment(
        self,
        current_scores: dict[str, float],
        current_biomarkers: dict[str, float],
        historical_assessments: list[dict[str, Any]],
    ) -> LongitudinalAssessment:
        """
        Make assessment incorporating historical data.

        Analyzes trends and patterns over time to provide trajectory insights.
        """
        # Calculate trajectory
        trajectory = self._calculate_trajectory(current_scores, historical_assessments)

        # Detect patterns
        pattern = self._detect_pattern(current_scores, historical_assessments)

        # Generate clinical note
        clinical_note = self._generate_longitudinal_note(
            trajectory, pattern, current_scores, historical_assessments
        )

        # Project future risk
        risk_projection = self._project_risk(trajectory, pattern)

        # Calculate confidence
        confidence = self._calculate_trend_confidence(historical_assessments)

        return LongitudinalAssessment(
            trajectory=trajectory,
            pattern_detected=pattern,
            clinical_note=clinical_note,
            risk_projection=risk_projection,
            confidence_in_trend=confidence,
        )

    def _calculate_trajectory(
        self, current_scores: dict[str, float], historical: list[dict[str, Any]]
    ) -> str:
        """Calculate trajectory from historical data."""
        if not historical:
            return "unknown"

        # Get most recent historical score
        recent_phq = historical[-1].get("phq_score", 0)
        current_phq = current_scores.get("phq", 0)

        if current_phq > recent_phq + 3:
            return "worsening"
        elif current_phq < recent_phq - 3:
            return "improving"
        else:
            return "stable"

    def _detect_pattern(
        self, current_scores: dict[str, float], historical: list[dict[str, Any]]
    ) -> str:
        """Detect clinical patterns."""
        if not historical:
            return "insufficient_data"

        # Check for escalating depression
        phq_scores = [h.get("phq_score", 0) for h in historical]
        phq_scores.append(current_scores.get("phq", 0))

        if all(phq_scores[i] <= phq_scores[i + 1] for i in range(len(phq_scores) - 1)):
            return "escalating_depression"
        elif all(
            phq_scores[i] >= phq_scores[i + 1] for i in range(len(phq_scores) - 1)
        ):
            return "improving_depression"
        else:
            return "fluctuating"

    def _generate_longitudinal_note(
        self,
        trajectory: str,
        pattern: str,
        current: dict[str, float],
        historical: list[dict[str, Any]],
    ) -> str:
        """Generate longitudinal clinical note."""
        severity_note = ""
        if trajectory == "worsening" and "escalating" in pattern:
            severity_note = "Note: Increasing severity observed. "

        return (
            f"Patient shows {trajectory} trajectory with {pattern} pattern. "
            f"Current PHQ score: {current.get('phq', 'N/A')}. "
            f"Historical data points: {len(historical)}. "
            f"{severity_note}"
            f"Recommend continued monitoring with focus on {trajectory} trend."
        )

    def _project_risk(self, trajectory: str, pattern: str) -> str:
        """Project future risk based on trajectory."""
        if trajectory == "worsening" and "escalating" in pattern:
            return "high_risk_for_episode"
        elif trajectory == "improving":
            return "decreasing_risk"
        else:
            return "stable_risk"

    def _calculate_trend_confidence(self, historical: list[dict[str, Any]]) -> float:
        """Calculate confidence in trend analysis."""
        # More data points = higher confidence
        return min(0.5 + (len(historical) * 0.1), 0.9)
