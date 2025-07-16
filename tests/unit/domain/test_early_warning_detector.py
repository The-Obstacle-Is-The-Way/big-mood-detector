"""
Test Early Warning Detector

TDD approach for extracting early warning detection logic from ClinicalInterpreter.
Following Uncle Bob's principles: clean, testable, single responsibility.
"""

import pytest

from big_mood_detector.domain.services.clinical_thresholds import (
    load_clinical_thresholds,
)


class TestEarlyWarningDetector:
    """Test early warning detection logic."""

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        from pathlib import Path
        return load_clinical_thresholds(Path("config/clinical_thresholds.yaml"))

    @pytest.fixture
    def detector(self, config):
        """Create EarlyWarningDetector instance."""
        # Red phase - this will fail initially
        from big_mood_detector.domain.services.early_warning_detector import (
            EarlyWarningDetector,
        )
        return EarlyWarningDetector(config)

    def test_detect_depression_warning_sleep_increase(self, detector):
        """Test depression warning from sleep increase."""
        result = detector.detect_warnings(
            sleep_change_hours=2.5,  # >2 hours increase
            activity_change_percent=-10,  # Mild decrease
            circadian_shift_hours=0.5,
            consecutive_days=3,
        )
        
        assert result.depression_warning is True
        assert "sleep increase" in result.clinical_summary.lower()
        assert len(result.warning_signs) >= 1

    def test_detect_depression_warning_activity_decrease(self, detector):
        """Test depression warning from activity decrease."""
        result = detector.detect_warnings(
            sleep_change_hours=0.5,
            activity_change_percent=-35,  # >30% decrease
            circadian_shift_hours=0,
            consecutive_days=3,
        )
        
        assert result.depression_warning is True
        assert "activity reduction" in result.clinical_summary.lower()
        assert not result.mania_warning

    def test_detect_mania_warning_sleep_decrease(self, detector):
        """Test mania warning from sleep decrease."""
        result = detector.detect_warnings(
            sleep_change_hours=-3,  # 3 hours decrease
            activity_change_percent=20,
            circadian_shift_hours=0,
            consecutive_days=2,
        )
        
        assert result.mania_warning is True
        assert "sleep reduction" in result.clinical_summary.lower()
        assert not result.depression_warning

    def test_detect_mania_warning_activity_increase(self, detector):
        """Test mania warning from activity increase."""
        result = detector.detect_warnings(
            sleep_change_hours=-1,
            activity_change_percent=60,  # >50% increase
            circadian_shift_hours=0,
            consecutive_days=2,
        )
        
        assert result.mania_warning is True
        assert "activity increase" in result.clinical_summary.lower()

    def test_detect_mixed_warnings(self, detector):
        """Test detection of mixed warning signs."""
        result = detector.detect_warnings(
            sleep_change_hours=-2.5,  # Mania sign
            activity_change_percent=-35,  # Depression sign
            circadian_shift_hours=1.5,  # Depression sign
            consecutive_days=3,
        )
        
        assert result.depression_warning is True
        assert result.mania_warning is True
        assert "mixed" in result.clinical_summary.lower()
        assert len(result.warning_signs) >= 3

    def test_trigger_intervention_threshold(self, detector):
        """Test intervention triggering based on multiple signs."""
        # Multiple warning signs for consecutive days
        result = detector.detect_warnings(
            sleep_change_hours=3,  # Significant
            activity_change_percent=-40,  # Significant
            circadian_shift_hours=2,  # Significant
            consecutive_days=3,
            speech_pattern_change=True,
        )
        
        assert result.trigger_intervention is True
        assert result.urgency_level == "high"
        assert "immediate" in result.clinical_summary.lower()

    def test_no_warnings_normal_variation(self, detector):
        """Test no warnings for normal variations."""
        result = detector.detect_warnings(
            sleep_change_hours=0.5,  # Normal variation
            activity_change_percent=15,  # Normal variation
            circadian_shift_hours=0.25,  # Normal variation
            consecutive_days=1,
        )
        
        assert result.depression_warning is False
        assert result.mania_warning is False
        assert result.trigger_intervention is False
        assert result.urgency_level == "none"

    def test_insufficient_consecutive_days(self, detector):
        """Test that warnings require minimum consecutive days."""
        result = detector.detect_warnings(
            sleep_change_hours=3,  # Significant but...
            activity_change_percent=60,  # Significant but...
            circadian_shift_hours=2,  # Significant but...
            consecutive_days=1,  # Not enough days
        )
        
        # Should detect signs but not trigger intervention
        assert len(result.warning_signs) > 0
        assert result.trigger_intervention is False
        assert "monitoring required" in result.clinical_summary.lower()

    def test_speech_pattern_mania_indicator(self, detector):
        """Test speech pattern changes as mania indicator."""
        result = detector.detect_warnings(
            sleep_change_hours=-1,
            activity_change_percent=30,
            circadian_shift_hours=0,
            consecutive_days=2,
            speech_pattern_change=True,  # Key mania indicator
        )
        
        assert result.mania_warning is True
        assert "speech" in result.clinical_summary.lower()
        assert any("speech" in sign.lower() for sign in result.warning_signs)

    def test_calculate_warning_severity(self, detector):
        """Test calculation of warning severity scores."""
        result = detector.detect_warnings(
            sleep_change_hours=4,  # Severe
            activity_change_percent=-50,  # Severe
            circadian_shift_hours=3,  # Severe
            consecutive_days=5,  # Extended period
        )
        
        assert result.severity_score > 0.7  # High severity
        assert result.confidence > 0.8  # High confidence with multiple indicators

    def test_personalized_thresholds(self, detector):
        """Test early warning detection with personalized thresholds."""
        # Individual with different baseline
        personalized_result = detector.detect_warnings_personalized(
            sleep_change_hours=1.5,  # Below general threshold but...
            activity_change_percent=-20,  # Below general threshold but...
            circadian_shift_hours=0.5,
            consecutive_days=3,
            individual_thresholds={
                "sleep_sensitivity": 1.2,  # More sensitive to sleep changes
                "activity_sensitivity": 1.5,  # More sensitive to activity changes
            }
        )
        
        assert personalized_result.depression_warning is True
        assert "personalized threshold" in personalized_result.clinical_summary.lower()