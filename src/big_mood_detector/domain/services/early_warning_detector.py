"""
Early Warning Detector Service

Detects early warning signs of mood episodes based on behavioral and
physiological changes over time.

Extracted from ClinicalInterpreter following Single Responsibility Principle.

Design Patterns:
- Observer Pattern: Could notify when warnings detected
- Strategy Pattern: Different detection strategies for depression/mania
- Value Objects: Immutable warning result objects
"""

from dataclasses import dataclass, field

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)


@dataclass(frozen=True)
class EarlyWarningResult:
    """Immutable early warning detection result."""
    depression_warning: bool = False
    mania_warning: bool = False
    trigger_intervention: bool = False
    warning_signs: list[str] = field(default_factory=list)
    urgency_level: str = "none"  # none, low, moderate, high
    severity_score: float = 0.0
    confidence: float = 1.0
    clinical_summary: str = ""


class EarlyWarningDetector:
    """
    Detects early warning signs of mood episodes.

    This service focuses solely on early warning detection,
    extracted from the monolithic ClinicalInterpreter.
    """

    def __init__(self, config: ClinicalThresholdsConfig):
        """
        Initialize with clinical configuration.

        Args:
            config: Clinical thresholds configuration
        """
        self.config = config

    def detect_warnings(
        self,
        sleep_change_hours: float,
        activity_change_percent: float,
        circadian_shift_hours: float,
        consecutive_days: int,
        speech_pattern_change: bool = False,
    ) -> EarlyWarningResult:
        """
        Detect early warning signs of mood episodes.

        Args:
            sleep_change_hours: Change in sleep duration (positive = increase)
            activity_change_percent: Percent change in activity (negative = decrease)
            circadian_shift_hours: Hours of circadian rhythm shift
            consecutive_days: Number of consecutive days with changes
            speech_pattern_change: Whether speech patterns have changed

        Returns:
            Early warning detection result
        """
        warning_signs = []
        depression_indicators = 0
        mania_indicators = 0

        # Check depression indicators
        sleep_threshold = 2.0
        activity_threshold = -30.0

        if sleep_change_hours > sleep_threshold:
            warning_signs.append("Significant sleep increase")
            depression_indicators += 1

        if activity_change_percent <= activity_threshold:
            warning_signs.append("Major activity reduction")
            depression_indicators += 1

        if circadian_shift_hours > 1:
            warning_signs.append("Circadian phase delay")
            depression_indicators += 1

        # Check mania indicators
        if sleep_change_hours < -2:
            warning_signs.append("Significant sleep reduction")
            mania_indicators += 1

        if activity_change_percent > 50:
            warning_signs.append("Major activity increase")
            mania_indicators += 1

        if speech_pattern_change:
            warning_signs.append("Increased speech rate")
            mania_indicators += 1

        # Determine warnings
        depression_warning = depression_indicators >= 1
        mania_warning = mania_indicators >= 1

        # Check for intervention trigger
        total_indicators = len(warning_signs)
        trigger_intervention = total_indicators >= 3 and consecutive_days >= 3

        # Calculate severity and urgency
        severity_score = min(1.0, total_indicators / 4.0)  # Max 4 indicators expected

        if trigger_intervention:
            urgency_level = "high"
        elif total_indicators >= 2 and consecutive_days >= 2:
            urgency_level = "moderate"
        elif total_indicators >= 1:
            urgency_level = "low"
        else:
            urgency_level = "none"

        # Generate clinical summary
        if not warning_signs:
            summary = "No significant warning signs detected"
        elif depression_warning and mania_warning:
            summary = "Mixed warning signs detected - both depression and mania indicators present"
        elif depression_warning:
            summary = f"Depression warning signs detected: {', '.join(sign for sign in warning_signs if 'increase' in sign or 'reduction' in sign or 'delay' in sign)}"
        elif mania_warning:
            # For mania, we want sleep reduction, activity increase, or speech changes
            summary = f"Mania warning signs detected: {', '.join(warning_signs)}"
        else:
            summary = f"Warning signs detected: {', '.join(warning_signs)}"

        if trigger_intervention:
            summary += ". Immediate clinical intervention recommended"
        elif consecutive_days < 3 and warning_signs:
            summary += ". Continued monitoring required"

        return EarlyWarningResult(
            depression_warning=depression_warning,
            mania_warning=mania_warning,
            trigger_intervention=trigger_intervention,
            warning_signs=warning_signs,
            urgency_level=urgency_level,
            severity_score=severity_score,
            confidence=1.0 if consecutive_days >= 3 else 0.8,
            clinical_summary=summary,
        )

    def detect_warnings_personalized(
        self,
        sleep_change_hours: float,
        activity_change_percent: float,
        circadian_shift_hours: float,
        consecutive_days: int,
        individual_thresholds: dict[str, float] | None = None,
        speech_pattern_change: bool = False,
    ) -> EarlyWarningResult:
        """
        Detect warnings with personalized sensitivity thresholds.

        Args:
            sleep_change_hours: Change in sleep duration
            activity_change_percent: Percent change in activity
            circadian_shift_hours: Hours of circadian rhythm shift
            consecutive_days: Number of consecutive days
            individual_thresholds: Personalized sensitivity multipliers
            speech_pattern_change: Whether speech patterns have changed

        Returns:
            Personalized early warning detection result
        """
        if individual_thresholds is None:
            individual_thresholds = {}

        # Apply personalized sensitivity
        sleep_sensitivity = individual_thresholds.get("sleep_sensitivity", 1.0)
        activity_sensitivity = individual_thresholds.get("activity_sensitivity", 1.0)

        # Adjust changes based on individual sensitivity
        adjusted_sleep_change = sleep_change_hours * sleep_sensitivity
        adjusted_activity_change = activity_change_percent * activity_sensitivity

        # Use adjusted values for detection
        result = self.detect_warnings(
            sleep_change_hours=adjusted_sleep_change,
            activity_change_percent=adjusted_activity_change,
            circadian_shift_hours=circadian_shift_hours,
            consecutive_days=consecutive_days,
            speech_pattern_change=speech_pattern_change,
        )

        # Add personalization note to summary
        if individual_thresholds and result.warning_signs:
            summary = result.clinical_summary.replace(
                "detected",
                "detected (using personalized thresholds)"
            )
            return EarlyWarningResult(
                depression_warning=result.depression_warning,
                mania_warning=result.mania_warning,
                trigger_intervention=result.trigger_intervention,
                warning_signs=result.warning_signs,
                urgency_level=result.urgency_level,
                severity_score=result.severity_score,
                confidence=result.confidence,
                clinical_summary=summary,
            )

        return result
