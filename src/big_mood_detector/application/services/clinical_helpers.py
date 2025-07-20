"""
Clinical Helper Functions

Shared utilities for clinical interpretation of ML predictions.
"""

from big_mood_detector.application.services.prediction_interpreter import (
    ClinicalInterpretation,
    PredictionInterpreter,
)


def get_clinical_interpretation(
    ml_predictions: dict[str, float],
) -> ClinicalInterpretation:
    """
    Get clinical interpretation from ML predictions.

    Args:
        ml_predictions: Dictionary with depression, mania, hypomania scores

    Returns:
        Clinical interpretation with diagnosis and recommendations
    """
    interpreter = PredictionInterpreter()
    return interpreter.interpret(ml_predictions)
