"""
Temporal Mood Assessment

Represents mood assessments across different time horizons.
Critical distinction: CURRENT state (PAT) vs FUTURE risk (XGBoost).
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CurrentMoodState:
    """
    Current mood state based on recent activity patterns.

    Derived from PAT analysis of past 7 days.
    Answers: "What is the person's state RIGHT NOW?"
    """
    depression_probability: float  # P(PHQ-9 >= 10 NOW)
    on_benzodiazepine_probability: float | None = None
    on_ssri_probability: float | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate probabilities are in valid range."""
        if not 0 <= self.depression_probability <= 1:
            raise ValueError(f"Invalid depression probability: {self.depression_probability}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Invalid confidence: {self.confidence}")


@dataclass(frozen=True)
class FutureMoodRisk:
    """
    Predicted mood episode risk for tomorrow.

    Derived from XGBoost circadian and activity features.
    Answers: "What episodes might occur TOMORROW?"
    """
    depression_risk: float  # P(depression episode tomorrow)
    hypomanic_risk: float   # P(hypomanic episode tomorrow)
    manic_risk: float       # P(manic episode tomorrow)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate risks are in valid range."""
        for risk in [self.depression_risk, self.hypomanic_risk, self.manic_risk]:
            if not 0 <= risk <= 1:
                raise ValueError(f"Invalid risk value: {risk}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Invalid confidence: {self.confidence}")


@dataclass(frozen=True)
class TemporalMoodAssessment:
    """
    Complete temporal mood assessment combining current state and future risk.

    This is the key innovation: We don't average current and future,
    we present them as complementary temporal perspectives.

    Clinical use cases:
    - Current state HIGH + Future risk LOW = Crisis now, but improving
    - Current state LOW + Future risk HIGH = Stable now, but warning signs
    - Both HIGH = Urgent intervention needed
    - Both LOW = Continue monitoring
    """
    # Current state (PAT-based, past 7 days → now)
    current_state: CurrentMoodState

    # Future risk (XGBoost-based, patterns → tomorrow)
    future_risk: FutureMoodRisk

    # Metadata
    assessment_timestamp: datetime
    user_id: str

    @property
    def requires_immediate_intervention(self) -> bool:
        """Check if current state warrants immediate clinical attention."""
        return self.current_state.depression_probability > 0.7

    @property
    def requires_preventive_action(self) -> bool:
        """Check if future risk warrants preventive measures today."""
        return (
            self.future_risk.depression_risk > 0.6 or
            self.future_risk.hypomanic_risk > 0.5 or
            self.future_risk.manic_risk > 0.4
        )

    @property
    def temporal_concordance(self) -> float:
        """
        Measure agreement between current state and future risk.

        High concordance = stable trajectory
        Low concordance = changing state (improvement or deterioration)
        """
        current_depression = self.current_state.depression_probability
        future_depression = self.future_risk.depression_risk

        # Simple concordance: how similar are current and future depression assessments?
        concordance = 1.0 - abs(current_depression - future_depression)
        return concordance

    def get_clinical_summary(self) -> dict[str, str]:
        """Generate human-readable clinical summary."""
        summary = {
            "current_status": self._get_current_status(),
            "tomorrow_outlook": self._get_tomorrow_outlook(),
            "recommended_action": self._get_recommended_action(),
            "temporal_pattern": self._get_temporal_pattern()
        }
        return summary

    def _get_current_status(self) -> str:
        """Describe current mood state."""
        dep_prob = self.current_state.depression_probability
        if dep_prob < 0.3:
            return "Currently stable"
        elif dep_prob < 0.6:
            return "Mild depression symptoms"
        else:
            return "Significant depression indicators"

    def _get_tomorrow_outlook(self) -> str:
        """Describe tomorrow's risk profile."""
        risks = [
            ("depression", self.future_risk.depression_risk),
            ("hypomania", self.future_risk.hypomanic_risk),
            ("mania", self.future_risk.manic_risk)
        ]

        highest_risk = max(risks, key=lambda x: x[1])
        risk_name, risk_value = highest_risk

        if risk_value < 0.3:
            return "Low risk for mood episodes tomorrow"
        elif risk_value < 0.6:
            return f"Moderate {risk_name} risk tomorrow"
        else:
            return f"High {risk_name} risk tomorrow"

    def _get_recommended_action(self) -> str:
        """Generate clinical recommendation based on temporal pattern."""
        if self.requires_immediate_intervention:
            return "Immediate clinical assessment recommended"
        elif self.requires_preventive_action:
            return "Implement preventive strategies today"
        elif self.temporal_concordance < 0.5:
            return "Monitor closely - state is changing"
        else:
            return "Continue current management plan"

    def _get_temporal_pattern(self) -> str:
        """Describe the temporal trajectory."""
        current_high = self.current_state.depression_probability > 0.5
        future_high = self.future_risk.depression_risk > 0.5

        if current_high and not future_high:
            return "Improving trajectory"
        elif not current_high and future_high:
            return "Deteriorating trajectory"
        elif current_high and future_high:
            return "Persistent elevation"
        else:
            return "Stable low risk"
