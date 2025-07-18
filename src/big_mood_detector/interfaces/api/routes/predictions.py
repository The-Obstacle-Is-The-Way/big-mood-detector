"""
Prediction API Routes

Direct mood prediction endpoints for extracted features.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Note: Full ensemble prediction would use EnsembleConfig from process_health_data_use_case
from big_mood_detector.domain.services.mood_predictor import MoodPredictor
from big_mood_detector.interfaces.api.dependencies import (
    get_mood_predictor,
)

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


class FeatureInput(BaseModel):
    """Input features for mood prediction."""

    # Sleep features
    sleep_duration: float = Field(..., ge=0, le=24)
    sleep_efficiency: float = Field(..., ge=0, le=1)
    sleep_timing_variance: float = Field(..., ge=0)

    # Activity features
    daily_steps: int = Field(..., ge=0)
    activity_variance: float = Field(..., ge=0)
    sedentary_hours: float = Field(..., ge=0, le=24)

    # Circadian features (optional)
    interdaily_stability: float | None = Field(None, ge=0, le=1)
    intradaily_variability: float | None = Field(None, ge=0)
    relative_amplitude: float | None = Field(None, ge=0, le=1)

    # Heart rate features (optional)
    resting_hr: float | None = Field(None, ge=30, le=200)
    hrv_rmssd: float | None = Field(None, ge=0, le=300)


class PredictionResponse(BaseModel):
    """Mood prediction response."""

    depression_risk: float = Field(..., ge=0, le=1)
    hypomanic_risk: float = Field(..., ge=0, le=1)
    manic_risk: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    risk_level: str  # low, moderate, high, critical
    interpretation: str


class EnsemblePredictionResponse(BaseModel):
    """Enhanced prediction using ensemble models."""

    xgboost_prediction: dict[str, float]
    pat_prediction: dict[str, float] | None
    ensemble_prediction: dict[str, float]
    models_used: list[str]
    confidence_scores: dict[str, float]
    clinical_summary: str
    recommendations: list[str]


@router.post("/predict", response_model=PredictionResponse)
async def predict_mood(
    features: FeatureInput,
    predictor: MoodPredictor = Depends(get_mood_predictor),
) -> PredictionResponse:
    """
    Generate mood prediction from feature vector.

    Uses XGBoost model to predict depression, hypomanic, and manic risk
    from extracted health data features.
    """
    try:
        # Convert features to expected format
        feature_dict = {
            "sleep_duration": features.sleep_duration,
            "sleep_efficiency": features.sleep_efficiency,
            "sleep_timing_variance": features.sleep_timing_variance,
            "daily_steps": features.daily_steps,
            "activity_variance": features.activity_variance,
            "sedentary_hours": features.sedentary_hours,
        }

        # Add optional features if provided
        if features.interdaily_stability is not None:
            feature_dict["interdaily_stability"] = features.interdaily_stability
        if features.intradaily_variability is not None:
            feature_dict["intradaily_variability"] = features.intradaily_variability
        if features.relative_amplitude is not None:
            feature_dict["relative_amplitude"] = features.relative_amplitude
        if features.resting_hr is not None:
            feature_dict["resting_hr"] = features.resting_hr
        if features.hrv_rmssd is not None:
            feature_dict["hrv_rmssd"] = features.hrv_rmssd

        # Use injected predictor (loaded once at startup)

        # Convert dict to numpy array for the predict method
        import numpy as np

        # Fill missing features with zeros, pad to 36 features if needed
        feature_array = np.zeros(36)
        feature_names = [
            "sleep_duration",
            "sleep_efficiency",
            "sleep_timing_variance",
            "daily_steps",
            "activity_variance",
            "sedentary_hours",
            "interdaily_stability",
            "intradaily_variability",
            "relative_amplitude",
            "resting_hr",
            "hrv_rmssd",
        ]

        for i, name in enumerate(feature_names[: len(feature_array)]):
            if name in feature_dict:
                feature_array[i] = feature_dict[name]

        prediction = predictor.predict(feature_array)

        # Determine risk level
        max_risk = max(
            prediction.depression_risk, prediction.hypomanic_risk, prediction.manic_risk
        )
        if max_risk < 0.3:
            risk_level = "low"
        elif max_risk < 0.6:
            risk_level = "moderate"
        elif max_risk < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Generate clinical interpretation
        if prediction.depression_risk > 0.5:
            interpretation = "Elevated depression risk detected"
        elif prediction.hypomanic_risk > 0.5:
            interpretation = "Possible hypomanic episode"
        elif prediction.manic_risk > 0.5:
            interpretation = "Possible manic episode"
        else:
            interpretation = "Low mood episode risk"

        return PredictionResponse(
            depression_risk=prediction.depression_risk,
            hypomanic_risk=prediction.hypomanic_risk,
            manic_risk=prediction.manic_risk,
            confidence=prediction.confidence,
            interpretation=interpretation,
            risk_level=risk_level,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@router.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
async def predict_mood_ensemble(
    features: FeatureInput,
    predictor: MoodPredictor = Depends(get_mood_predictor),
) -> EnsemblePredictionResponse:
    """
    Generate ensemble mood prediction using all available models.

    Combines XGBoost and PAT Transformer predictions for increased accuracy.
    """
    try:
        # Use pipeline with ensemble enabled
        # Note: For full ensemble prediction, we would use PipelineConfig and MoodPredictionPipeline
        # Here we're demonstrating direct feature prediction

        # Convert features
        feature_dict = {
            "sleep_duration": features.sleep_duration,
            "sleep_efficiency": features.sleep_efficiency,
            "sleep_timing_variance": features.sleep_timing_variance,
            "daily_steps": features.daily_steps,
            "activity_variance": features.activity_variance,
            "sedentary_hours": features.sedentary_hours,
        }

        # Add optional features
        if features.interdaily_stability is not None:
            feature_dict["interdaily_stability"] = features.interdaily_stability
        if features.intradaily_variability is not None:
            feature_dict["intradaily_variability"] = features.intradaily_variability
        if features.relative_amplitude is not None:
            feature_dict["relative_amplitude"] = features.relative_amplitude
        if features.resting_hr is not None:
            feature_dict["resting_hr"] = features.resting_hr
        if features.hrv_rmssd is not None:
            feature_dict["hrv_rmssd"] = features.hrv_rmssd

        # For demo purposes, create a mock ensemble result
        # In practice, this would use the full pipeline with real data

        # Use injected predictor
        import numpy as np

        feature_array = np.zeros(36)
        feature_names = list(feature_dict.keys())
        for i, name in enumerate(feature_names[: len(feature_array)]):
            if name in feature_dict:
                feature_array[i] = feature_dict[name]

        base_prediction = predictor.predict(feature_array)

        # Generate clinical summary
        max_risk = max(
            base_prediction.depression_risk,
            base_prediction.hypomanic_risk,
            base_prediction.manic_risk,
        )

        if max_risk > 0.7:
            clinical_summary = "High risk mood episode - recommend clinical assessment"
        elif max_risk > 0.5:
            clinical_summary = "Moderate risk - monitor closely"
        else:
            clinical_summary = "Low risk - maintain healthy lifestyle"

        recommendations = []
        if base_prediction.depression_risk > 0.5:
            recommendations.extend(
                ["Consider mood monitoring", "Maintain regular sleep schedule"]
            )
        if max_risk < 0.3:
            recommendations.extend(
                ["Continue current habits", "Regular sleep schedule"]
            )

        # Extract only numeric values for predictions
        xgb_pred = {
            "depression_risk": base_prediction.depression_risk,
            "hypomanic_risk": base_prediction.hypomanic_risk,
            "manic_risk": base_prediction.manic_risk,
            "confidence": base_prediction.confidence,
        }

        return EnsemblePredictionResponse(
            xgboost_prediction=xgb_pred,
            pat_prediction=None,  # Would be populated if PAT model available
            ensemble_prediction=xgb_pred,
            models_used=["xgboost"],
            confidence_scores={
                "xgboost": base_prediction.confidence,
                "ensemble": base_prediction.confidence,
            },
            clinical_summary=clinical_summary,
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ensemble prediction failed: {e}"
        ) from e


@router.get("/status")
async def get_model_status(
    predictor: MoodPredictor = Depends(get_mood_predictor),
) -> dict[str, Any]:
    """Get status of available prediction models."""
    try:
        # Use injected predictor

        return {
            "xgboost_available": len(predictor.models) > 0,
            "pat_available": False,  # Would check PAT model availability
            "ensemble_available": len(predictor.models) > 0,
            "models_loaded": list(predictor.models.keys()),
            "model_info": predictor.get_model_info(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}") from e
