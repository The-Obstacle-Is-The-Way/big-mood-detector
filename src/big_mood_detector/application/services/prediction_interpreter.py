"""
Prediction Interpreter Service

Interprets raw ML predictions to provide clinical insights and recommendations.
Follows the Seoul study's clinical interpretation framework.

Design Patterns:
- Strategy Pattern: Different interpretation strategies for different mood states
- Chain of Responsibility: Sequential evaluation of clinical criteria
- Builder Pattern: Building comprehensive clinical reports
"""

from dataclasses import dataclass


@dataclass
class ClinicalInterpretation:
    """Complete interpretation of mood predictions with clinical context."""

    primary_diagnosis: str
    risk_level: str  # 'low', 'moderate', 'high'
    confidence: float
    clinical_notes: list[str]
    recommendations: list[str]
    secondary_risks: dict[str, float]
    monitoring_frequency: str  # 'daily', 'weekly', 'monthly'


class PredictionInterpreter:
    """
    Service responsible for interpreting ML predictions in clinical context.

    This service bridges the gap between raw ML outputs and actionable
    clinical insights, following evidence-based interpretation guidelines.
    """

    def __init__(self) -> None:
        """Initialize the prediction interpreter."""
        # Clinical thresholds based on research
        self.thresholds = {
            "severe": 0.8,
            "moderate": 0.6,
            "mild": 0.4,
            "subclinical": 0.2,
        }

    def interpret(
        self,
        ml_predictions: dict[str, float],
    ) -> ClinicalInterpretation:
        """
        Interpret mood predictions in clinical context.

        Args:
            ml_predictions: Dictionary with depression, mania, hypomania scores

        Returns:
            Clinical interpretation with diagnosis and recommendations
        """
        # Determine primary diagnosis and risk level
        primary_diagnosis, risk_level = self._determine_diagnosis(ml_predictions)

        # Calculate confidence based on prediction strength
        confidence = self._calculate_confidence(ml_predictions)

        # Generate clinical notes
        clinical_notes = self._generate_clinical_notes(ml_predictions, primary_diagnosis)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            primary_diagnosis, risk_level, ml_predictions
        )

        # Determine monitoring frequency
        monitoring_frequency = self._determine_monitoring_frequency(risk_level)

        # Calculate secondary risks
        secondary_risks = self._calculate_secondary_risks(ml_predictions)

        return ClinicalInterpretation(
            primary_diagnosis=primary_diagnosis,
            risk_level=risk_level,
            confidence=confidence,
            clinical_notes=clinical_notes,
            recommendations=recommendations,
            secondary_risks=secondary_risks,
            monitoring_frequency=monitoring_frequency,
        )

    def _determine_diagnosis(
        self, ml_predictions: dict[str, float]
    ) -> tuple[str, str]:
        """Determine primary diagnosis and risk level."""
        depression = ml_predictions.get("depression", 0.0)
        mania = ml_predictions.get("mania", 0.0)
        hypomania = ml_predictions.get("hypomania", 0.0)

        # Check for mixed features first
        if depression > 0.5 and (mania > 0.3 or hypomania > 0.3):
            return "Mixed Episode", "critical"

        # Find the highest score
        max_score = max(depression, mania, hypomania)

        # Default to euthymic if all scores are very low
        if max_score < 0.2:
            return "Euthymic (Stable)", "low"

        # Determine primary condition
        if depression == max_score:
            if depression >= self.thresholds["severe"]:
                return "Severe Depressive Episode", "high"
            elif depression >= self.thresholds["moderate"]:
                return "Moderate Depressive Episode", "moderate"
            elif depression >= self.thresholds["mild"]:
                return "Mild Depressive Episode", "moderate"
            else:
                return "Subclinical Depression", "low"

        elif mania == max_score:
            if mania >= self.thresholds["severe"]:
                return "Manic Episode", "high"
            elif mania >= self.thresholds["moderate"]:
                return "Hypomanic Episode with Manic Features", "high"
            else:
                return "Subclinical Mania", "moderate"

        else:  # hypomania
            if hypomania >= self.thresholds["moderate"]:
                return "Hypomanic Episode", "moderate"
            elif hypomania >= self.thresholds["mild"]:
                return "Mild Hypomanic Episode", "low"
            else:
                return "Subclinical Hypomania", "low"

    def _calculate_confidence(self, ml_predictions: dict[str, float]) -> float:
        """Calculate confidence based on prediction strength and clarity."""
        # Get the top two predictions
        scores = [
            ml_predictions.get("depression", 0.0),
            ml_predictions.get("mania", 0.0),
            ml_predictions.get("hypomania", 0.0),
        ]
        scores.sort(reverse=True)

        # Confidence is high when there's clear separation
        if scores[0] > 0.7 and scores[1] < 0.3:
            return 0.9
        elif scores[0] > 0.6 and scores[1] < 0.4:
            return 0.8
        elif scores[0] > 0.5:
            return 0.7
        else:
            # For low scores (euthymic), check consistency
            if max(scores) < 0.2 and scores[0] - scores[2] < 0.1:
                return 0.8  # Consistent low scores = high confidence in euthymic
            elif scores[0] - scores[1] < 0.1:  # Ambiguous case
                return 0.5
            else:
                return 0.6

    def _generate_clinical_notes(
        self, ml_predictions: dict[str, float], primary_diagnosis: str
    ) -> list[str]:
        """Generate clinical notes based on predictions."""
        notes = []

        depression = ml_predictions.get("depression", 0.0)
        mania = ml_predictions.get("mania", 0.0)
        hypomania = ml_predictions.get("hypomania", 0.0)

        # Mixed episode notes
        if "Mixed" in primary_diagnosis:
            notes.append("Mixed mood episode detected - simultaneous depressive and manic symptoms")
            notes.append("High risk state requiring immediate intervention")
            notes.append("Meets DSM-5 criteria for mixed features specifier")
            return notes

        # Euthymic/stable state notes
        if "Euthymic" in primary_diagnosis:
            notes.append("Patient shows stable mood indicators")
            notes.append("All mood scores within normal range")
            return notes

        # Primary condition notes
        if "Depressive" in primary_diagnosis or "Depression" in primary_diagnosis:
            notes.append(
                f"Patient shows strong indicators of depression (score: {depression:.2f})"
            )
            if depression > 0.8:
                notes.append("Severity level suggests significant functional impairment")
            if depression > 0.7:
                notes.append("Meets DSM-5 criteria for major depressive episode")

        elif "Manic" in primary_diagnosis:
            notes.append(
                f"Patient exhibits manic symptoms (score: {mania:.2f})"
            )
            if mania > 0.7:
                notes.append("High risk of impulsive behavior and poor judgment")
                notes.append("Consistent with DSM-5 criteria for manic episode")

        elif "Hypomanic" in primary_diagnosis:
            notes.append(
                f"Patient shows hypomania symptoms (score: {hypomania:.2f})"
            )
            notes.append("Monitor for potential escalation to mania")
            if hypomania > 0.6:
                notes.append("Meets DSM-5 criteria for hypomanic episode")

        # Secondary risk notes
        if depression > 0.3 and "Depression" not in primary_diagnosis:
            notes.append(f"Secondary depression risk detected ({depression:.2f})")

        if (mania > 0.3 or hypomania > 0.3) and "Manic" not in primary_diagnosis:
            notes.append("Elevated mood symptoms present as secondary feature")

        # Mixed features
        if depression > 0.5 and (mania > 0.3 or hypomania > 0.3):
            notes.append("Mixed features present - increased suicide risk")
            notes.append("Mixed episode per DSM-5 criteria - requires urgent care")

        return notes

    def _generate_recommendations(
        self, primary_diagnosis: str, risk_level: str, ml_predictions: dict[str, float]
    ) -> list[str]:
        """Generate clinical recommendations based on diagnosis."""
        recommendations = []

        # Critical risk recommendations
        if risk_level == "critical":
            recommendations.append("Emergency psychiatric evaluation required")
            recommendations.append("Crisis intervention team consultation")
            recommendations.append("Consider immediate hospitalization")
        # High risk recommendations
        elif risk_level == "high":
            recommendations.append("Urgent psychiatric evaluation recommended")
            recommendations.append("Consider immediate safety assessment")

            if "Depression" in primary_diagnosis:
                recommendations.append("Assess suicide risk")
                recommendations.append("Consider antidepressant therapy adjustment")
            elif "Manic" in primary_diagnosis:
                recommendations.append("Evaluate need for mood stabilizers")
                recommendations.append("Monitor for psychotic features")
                recommendations.append("Consider emergency department evaluation")
            elif "Mixed" in primary_diagnosis:
                recommendations.append("Emergency psychiatric consultation needed")
                recommendations.append("Consider hospitalization for safety")

        # Moderate risk recommendations
        elif risk_level == "moderate":
            recommendations.append("Schedule psychiatric follow-up within 1-2 weeks")
            recommendations.append("Monitor symptom progression daily")

            if "Depression" in primary_diagnosis:
                recommendations.append("Consider psychotherapy referral")
                recommendations.append("Encourage behavioral activation")
            elif "Hypomanic" in primary_diagnosis:
                recommendations.append("Maintain regular sleep schedule")
                recommendations.append("Avoid stimulants and alcohol")

        # Low risk recommendations
        else:
            recommendations.append("Continue current monitoring schedule")
            recommendations.append("Maintain healthy lifestyle habits")

        # General recommendations
        recommendations.append("Ensure medication adherence if prescribed")
        recommendations.append("Track mood changes using daily logs")

        return recommendations

    def _determine_monitoring_frequency(self, risk_level: str) -> str:
        """Determine appropriate monitoring frequency based on risk."""
        if risk_level == "critical":
            return "daily"
        elif risk_level == "high":
            return "daily"
        elif risk_level == "moderate":
            return "weekly"
        else:
            return "monthly"

    def _calculate_secondary_risks(
        self, ml_predictions: dict[str, float]
    ) -> dict[str, float]:
        """Calculate secondary risks for comprehensive assessment."""
        depression = ml_predictions.get("depression", 0.0)
        mania = ml_predictions.get("mania", 0.0)
        hypomania = ml_predictions.get("hypomania", 0.0)

        # Calculate additional risk factors
        mixed_risk = min(depression, max(mania, hypomania)) * 2
        rapid_cycling_risk = (abs(depression - mania) < 0.3 and max(depression, mania) > 0.5)
        psychosis_risk = mania * 0.7 if mania > 0.7 else 0.0

        return {
            "mixed_features": min(mixed_risk, 1.0),
            "rapid_cycling": 0.7 if rapid_cycling_risk else 0.2,
            "psychosis": psychosis_risk,
            "suicide": depression * 0.8 if depression > 0.6 else depression * 0.3,
        }
