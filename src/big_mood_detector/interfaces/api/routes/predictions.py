"""
Prediction API Routes

Direct mood prediction endpoints for extracted features.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    PredictMoodEnsembleUseCase,
)
from big_mood_detector.domain.services.mood_predictor import MoodPredictor

router = APIRouter(prefix="/api/v1/predict", tags=["predictions"])


class FeatureInput(BaseModel):
    """Feature vector for mood prediction."""

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
    hrv_rmssd: float | None = Field(None, ge=0)


class PredictionResponse(BaseModel):
    """Mood prediction response."""

    depression_risk: float = Field(..., ge=0, le=1)
    hypomanic_risk: float = Field(..., ge=0, le=1) 
    manic_risk: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_used: str
    clinical_interpretation: str
    risk_level: str


class EnsemblePredictionResponse(BaseModel):
    """Ensemble prediction response with model details."""

    xgboost_prediction: dict[str, float]
    pat_prediction: dict[str, float] | None
    ensemble_prediction: dict[str, float]
    confidence: float
    models_used: list[str]
    clinical_summary: str
    recommendations: list[str]


@router.post("/single", response_model=PredictionResponse)
async def predict_mood(features: FeatureInput) -> PredictionResponse:
    """
    Generate mood prediction from feature vector.
    
    Uses XGBoost model to predict depression, hypomanic, and manic risk
    from extracted health data features.
    """
    try:
        # Convert to feature dict
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
            
        # Get prediction
        predictor = MoodPredictor()
        prediction = predictor.predict_single(feature_dict)
        
        # Determine risk level
        max_risk = max(prediction.depression, prediction.hypomanic, prediction.manic)
        if max_risk < 0.3:
            risk_level = "low"
        elif max_risk < 0.6:
            risk_level = "moderate" 
        elif max_risk < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"
            
        # Generate clinical interpretation
        if prediction.depression > 0.5:
            interpretation = f"Elevated depression risk ({prediction.depression:.1%})"
        elif prediction.manic > 0.5:
            interpretation = f"Elevated manic risk ({prediction.manic:.1%})"
        elif prediction.hypomanic > 0.5:
            interpretation = f"Elevated hypomanic risk ({prediction.hypomanic:.1%})"
        else:
            interpretation = "Low mood episode risk"
            
        return PredictionResponse(
            depression_risk=prediction.depression,
            hypomanic_risk=prediction.hypomanic,
            manic_risk=prediction.manic,
            confidence=prediction.confidence,
            model_used="XGBoost",
            clinical_interpretation=interpretation,
            risk_level=risk_level,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@router.post("/ensemble", response_model=EnsemblePredictionResponse)
async def predict_mood_ensemble(features: FeatureInput) -> EnsemblePredictionResponse:
    """
    Generate ensemble mood prediction using all available models.
    
    Combines XGBoost and PAT Transformer predictions for increased accuracy.
    """
    try:
        # Use ensemble use case
        use_case = PredictMoodEnsembleUseCase()
        
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
            
        # Get ensemble prediction
        result = use_case.execute(feature_dict)
        
        # Generate clinical summary
        max_risk = max(
            result.ensemble_prediction["depression"],
            result.ensemble_prediction["hypomanic"], 
            result.ensemble_prediction["manic"]
        )
        
        if max_risk > 0.7:
            clinical_summary = "High risk mood episode - recommend clinical assessment"
            recommendations = [
                "Schedule clinical evaluation within 1-2 weeks",
                "Monitor sleep patterns closely",
                "Consider mood tracking app"
            ]
        elif max_risk > 0.4:
            clinical_summary = "Moderate risk - monitor symptoms"
            recommendations = [
                "Continue regular monitoring",
                "Maintain sleep hygiene",
                "Regular exercise routine"
            ]
        else:
            clinical_summary = "Low risk - routine monitoring"
            recommendations = [
                "Continue healthy lifestyle",
                "Regular sleep schedule"
            ]
            
        return EnsemblePredictionResponse(
            xgboost_prediction=result.xgboost_prediction,
            pat_prediction=result.pat_prediction,
            ensemble_prediction=result.ensemble_prediction,
            confidence=result.confidence,
            models_used=result.models_used,
            clinical_summary=clinical_summary,
            recommendations=recommendations,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {e}") from e


@router.get("/models/status")
async def get_model_status() -> dict[str, Any]:
    """Get status of available prediction models."""
    try:
        predictor = MoodPredictor()
        
        return {
            "xgboost_available": len(predictor.models) > 0,
            "pat_available": False,  # Will be updated when PAT is loaded
            "models_loaded": list(predictor.models.keys()) if predictor.models else [],
            "feature_count": len(predictor.expected_features),
            "expected_features": predictor.expected_features,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}") from e 