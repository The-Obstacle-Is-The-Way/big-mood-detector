"""
API Dependencies for FastAPI.

Provides singleton instances and dependency injection for performance.
"""

from functools import lru_cache

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.services.mood_predictor import MoodPredictor


@lru_cache(maxsize=1)
def get_mood_predictor() -> MoodPredictor:
    """
    Get singleton MoodPredictor instance.
    
    This ensures models are loaded only once at startup,
    not on every request.
    
    Returns:
        Cached MoodPredictor instance
    """
    return MoodPredictor()


@lru_cache(maxsize=1)
def get_mood_pipeline() -> MoodPredictionPipeline:
    """
    Get singleton MoodPredictionPipeline instance.
    
    This ensures the pipeline is created only once at startup,
    not on every request.
    
    Returns:
        Cached MoodPredictionPipeline instance
    """
    return MoodPredictionPipeline()
