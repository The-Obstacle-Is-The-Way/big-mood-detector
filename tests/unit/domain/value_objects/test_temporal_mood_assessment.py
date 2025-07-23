"""
Test Temporal Mood Assessment

Tests for the temporal mood assessment value objects.
Ensures we properly separate current state from future predictions.
"""

from datetime import datetime

import pytest

from big_mood_detector.domain.value_objects.temporal_mood_assessment import (
    CurrentMoodState,
    FutureMoodRisk,
    TemporalMoodAssessment,
)


class TestCurrentMoodState:
    """Test current mood state value object."""

    def test_valid_current_state(self):
        """Should create valid current mood state."""
        state = CurrentMoodState(
            depression_probability=0.7,
            confidence=0.8
        )

        assert state.depression_probability == 0.7
        assert state.confidence == 0.8
        assert state.on_benzodiazepine_probability is None
        assert state.on_ssri_probability is None

    def test_invalid_probability_raises_error(self):
        """Should validate probability range."""
        with pytest.raises(ValueError, match="Invalid depression probability"):
            CurrentMoodState(depression_probability=1.5, confidence=0.8)

        with pytest.raises(ValueError, match="Invalid depression probability"):
            CurrentMoodState(depression_probability=-0.1, confidence=0.8)

    def test_invalid_confidence_raises_error(self):
        """Should validate confidence range."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            CurrentMoodState(depression_probability=0.5, confidence=1.5)


class TestFutureMoodRisk:
    """Test future mood risk value object."""

    def test_valid_future_risk(self):
        """Should create valid future mood risk."""
        risk = FutureMoodRisk(
            depression_risk=0.3,
            hypomanic_risk=0.2,
            manic_risk=0.1,
            confidence=0.9
        )

        assert risk.depression_risk == 0.3
        assert risk.hypomanic_risk == 0.2
        assert risk.manic_risk == 0.1
        assert risk.confidence == 0.9

    def test_invalid_risk_raises_error(self):
        """Should validate risk ranges."""
        with pytest.raises(ValueError, match="Invalid risk value"):
            FutureMoodRisk(
                depression_risk=1.1,
                hypomanic_risk=0.2,
                manic_risk=0.1
            )


class TestTemporalMoodAssessment:
    """Test complete temporal mood assessment."""

    @pytest.fixture
    def current_state_high(self):
        """Current state with high depression."""
        return CurrentMoodState(
            depression_probability=0.8,
            confidence=0.85
        )

    @pytest.fixture
    def current_state_low(self):
        """Current state with low depression."""
        return CurrentMoodState(
            depression_probability=0.2,
            confidence=0.9
        )

    @pytest.fixture
    def future_risk_high(self):
        """Future risk with high depression."""
        return FutureMoodRisk(
            depression_risk=0.7,
            hypomanic_risk=0.1,
            manic_risk=0.05,
            confidence=0.8
        )

    @pytest.fixture
    def future_risk_low(self):
        """Future risk with low depression."""
        return FutureMoodRisk(
            depression_risk=0.2,
            hypomanic_risk=0.1,
            manic_risk=0.05,
            confidence=0.85
        )

    def test_temporal_assessment_creation(self, current_state_high, future_risk_low):
        """Should create temporal assessment."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_high,
            future_risk=future_risk_low,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        assert assessment.current_state == current_state_high
        assert assessment.future_risk == future_risk_low
        assert assessment.user_id == "test_user"

    def test_requires_immediate_intervention(self, current_state_high, future_risk_low):
        """Should identify need for immediate intervention."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_high,
            future_risk=future_risk_low,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        assert assessment.requires_immediate_intervention is True

    def test_requires_preventive_action(self, current_state_low, future_risk_high):
        """Should identify need for preventive action."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_low,
            future_risk=future_risk_high,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        assert assessment.requires_immediate_intervention is False
        assert assessment.requires_preventive_action is True

    def test_temporal_concordance_high(self, current_state_high, future_risk_high):
        """Should calculate high concordance when states agree."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_high,
            future_risk=future_risk_high,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        # Both high (0.8 and 0.7) = high concordance
        assert assessment.temporal_concordance > 0.8

    def test_temporal_concordance_low(self, current_state_high, future_risk_low):
        """Should calculate low concordance when states differ."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_high,
            future_risk=future_risk_low,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        # Current high (0.8) vs future low (0.2) = low concordance
        assert assessment.temporal_concordance < 0.5

    def test_improving_trajectory(self, current_state_high, future_risk_low):
        """Should identify improving trajectory."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_high,
            future_risk=future_risk_low,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        summary = assessment.get_clinical_summary()
        assert summary["temporal_pattern"] == "Improving trajectory"
        assert "Significant depression" in summary["current_status"]
        assert "Low risk" in summary["tomorrow_outlook"]

    def test_deteriorating_trajectory(self, current_state_low, future_risk_high):
        """Should identify deteriorating trajectory."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_low,
            future_risk=future_risk_high,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        summary = assessment.get_clinical_summary()
        assert summary["temporal_pattern"] == "Deteriorating trajectory"
        assert "stable" in summary["current_status"]
        assert "depression risk tomorrow" in summary["tomorrow_outlook"]

    def test_persistent_elevation(self, current_state_high, future_risk_high):
        """Should identify persistent elevation."""
        assessment = TemporalMoodAssessment(
            current_state=current_state_high,
            future_risk=future_risk_high,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )

        summary = assessment.get_clinical_summary()
        assert summary["temporal_pattern"] == "Persistent elevation"

    def test_clinical_recommendations(self, current_state_high, future_risk_low):
        """Should generate appropriate clinical recommendations."""
        # Test immediate intervention
        assessment = TemporalMoodAssessment(
            current_state=current_state_high,
            future_risk=future_risk_low,
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )
        summary = assessment.get_clinical_summary()
        assert "Immediate clinical assessment" in summary["recommended_action"]

        # Test preventive action
        assessment2 = TemporalMoodAssessment(
            current_state=CurrentMoodState(depression_probability=0.4, confidence=0.8),
            future_risk=FutureMoodRisk(
                depression_risk=0.7,
                hypomanic_risk=0.1,
                manic_risk=0.05,
                confidence=0.8
            ),
            assessment_timestamp=datetime.now(),
            user_id="test_user"
        )
        summary2 = assessment2.get_clinical_summary()
        assert "preventive strategies" in summary2["recommended_action"]
