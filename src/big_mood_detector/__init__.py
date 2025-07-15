"""
Big Mood Detector - Clinical-grade bipolar mood detection backend.

This package provides a comprehensive backend system for detecting bipolar mood episodes
using Apple HealthKit data, following Clean Architecture principles.
"""

__version__ = "0.1.0"
__author__ = "The-Obstacle-Is-The-Way"
__email__ = "The-Obstacle-Is-The-Way@users.noreply.github.com"

# Public API exports
from big_mood_detector.application.services.mood_detection_service import (
    MoodDetectionService,
)
from big_mood_detector.domain.entities.mood_prediction import MoodPrediction
from big_mood_detector.domain.entities.health_data import HealthData

__all__ = [
    "MoodDetectionService",
    "MoodPrediction", 
    "HealthData",
] 