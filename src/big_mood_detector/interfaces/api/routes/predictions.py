"""
Prediction API Routes

Direct mood prediction endpoints for extracted features.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from big_mood_detector.application.services.clinical_helpers import (
    get_clinical_interpretation as interpret_predictions,
)
from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    EnsembleOrchestrator,
)
from big_mood_detector.core.feature_constants import (
    API_TO_XGBOOST_MAPPING,
    XGBOOST_FEATURE_NAMES,
)
from big_mood_detector.domain.services.mood_predictor import MoodPredictor
from big_mood_detector.interfaces.api.dependencies import (
    get_ensemble_orchestrator,
    get_mood_predictor,
)
from big_mood_detector.interfaces.api.middleware.rate_limit import rate_limit

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
@rate_limit("predict")
async def predict_mood(
    request: Request,
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
@rate_limit("ensemble_predict")
async def predict_mood_ensemble(
    request: Request,
    features: FeatureInput,
    orchestrator: EnsembleOrchestrator | None = Depends(get_ensemble_orchestrator),
) -> EnsemblePredictionResponse:
    """
    Generate ensemble mood prediction using all available models.

    Combines XGBoost and PAT Transformer predictions for increased accuracy.

    NOTE: For PAT predictions, this endpoint requires activity sequence data,
    which is not provided through this simple feature interface. For full
    ensemble predictions with PAT, use the file upload endpoints which process
    raw health data.
    """
    try:
        if orchestrator is None:
            # Check if it's a TensorFlow issue
            from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE

            if not PAT_AVAILABLE:
                raise HTTPException(
                    status_code=501,
                    detail="PAT model requires TensorFlow. Install with: pip install tensorflow>=2.14.0",
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Ensemble models not available. Check server logs.",
                )

        # Convert features to numpy array
        import numpy as np

        # Build feature array (36 features expected by XGBoost)
        feature_array = np.zeros(36, dtype=np.float32)

        # Fill in provided features using proper mapping
        feature_dict = features.model_dump(exclude_none=True)
        for name, idx in API_TO_XGBOOST_MAPPING.items():
            if name in feature_dict:
                feature_array[idx] = feature_dict[name]

        # Since we don't have activity records for PAT, we'll use XGBoost only
        # but through the ensemble orchestrator
        ensemble_result = orchestrator.predict(
            statistical_features=feature_array,
            activity_records=None,  # No activity data available from features
            prediction_date=None,
        )

        # Use PredictionInterpreter for clinical insights
        ml_predictions = {
            "depression": ensemble_result.ensemble_prediction.depression_risk,
            "mania": ensemble_result.ensemble_prediction.manic_risk,
            "hypomania": ensemble_result.ensemble_prediction.hypomanic_risk,
        }
        interpretation = interpret_predictions(ml_predictions)

        # Generate clinical summary from interpretation
        clinical_summary = (
            f"{interpretation.primary_diagnosis} - {interpretation.risk_level} risk"
        )

        # Use interpreter recommendations (limit to top 5 for API response)
        recommendations = interpretation.recommendations[:5]

        # Format response
        xgb_pred = None
        if ensemble_result.xgboost_prediction:
            xgb_pred = {
                "depression_risk": ensemble_result.xgboost_prediction.depression_risk,
                "hypomanic_risk": ensemble_result.xgboost_prediction.hypomanic_risk,
                "manic_risk": ensemble_result.xgboost_prediction.manic_risk,
                "confidence": ensemble_result.xgboost_prediction.confidence,
            }

        pat_pred = None
        if ensemble_result.pat_enhanced_prediction:
            pat_pred = {
                "depression_risk": ensemble_result.pat_enhanced_prediction.depression_risk,
                "hypomanic_risk": ensemble_result.pat_enhanced_prediction.hypomanic_risk,
                "manic_risk": ensemble_result.pat_enhanced_prediction.manic_risk,
                "confidence": ensemble_result.pat_enhanced_prediction.confidence,
            }

        ensemble_pred = {
            "depression_risk": ensemble_result.ensemble_prediction.depression_risk,
            "hypomanic_risk": ensemble_result.ensemble_prediction.hypomanic_risk,
            "manic_risk": ensemble_result.ensemble_prediction.manic_risk,
            "confidence": ensemble_result.ensemble_prediction.confidence,
        }

        return EnsemblePredictionResponse(
            xgboost_prediction=xgb_pred or ensemble_pred,
            pat_prediction=pat_pred,
            ensemble_prediction=ensemble_pred,
            models_used=ensemble_result.models_used,
            confidence_scores=ensemble_result.confidence_scores,
            clinical_summary=clinical_summary,
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ensemble prediction failed: {e}"
        ) from e


@router.get("/status")
@rate_limit("status")
async def get_model_status(
    request: Request,
    predictor: MoodPredictor = Depends(get_mood_predictor),
    orchestrator: EnsembleOrchestrator | None = Depends(get_ensemble_orchestrator),
) -> dict[str, Any]:
    """Get status of available prediction models."""
    try:
        # Check PAT availability
        pat_available = False
        pat_info = None
        if orchestrator and orchestrator.pat_model:
            pat_available = orchestrator.pat_model.is_loaded
            if pat_available:
                pat_info = orchestrator.pat_model.get_model_info()

        # Check ensemble availability
        ensemble_available = (
            orchestrator is not None and orchestrator.xgboost_predictor.is_loaded
        )

        return {
            "xgboost_available": len(predictor.models) > 0,
            "pat_available": pat_available,
            "ensemble_available": ensemble_available,
            "models_loaded": list(predictor.models.keys()),
            "model_info": predictor.get_model_info(),
            "pat_info": pat_info,
            "ensemble_config": (
                {
                    "xgboost_weight": (
                        orchestrator.config.xgboost_weight if orchestrator else None
                    ),
                    "pat_weight": (
                        orchestrator.config.pat_weight if orchestrator else None
                    ),
                }
                if orchestrator
                else None
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}") from e


class ClinicalInterpretationResponse(BaseModel):
    """Clinical interpretation of mood predictions."""

    # Core clinical assessment
    primary_diagnosis: str
    risk_level: str
    confidence: float

    # Detailed clinical insights
    clinical_notes: list[str]
    recommendations: list[str]

    # Secondary risk assessment
    secondary_risks: dict[str, float]
    monitoring_frequency: str

    # Original predictions
    ml_predictions: dict[str, float]

    # Metadata
    interpretation_version: str = "1.0"
    dsm5_compliant: bool = True


@router.post("/clinical", response_model=ClinicalInterpretationResponse)
@rate_limit("predict")
async def get_clinical_interpretation(
    request: Request,
    features: FeatureInput,
    predictor: MoodPredictor = Depends(get_mood_predictor),
    orchestrator: EnsembleOrchestrator | None = Depends(get_ensemble_orchestrator),
) -> ClinicalInterpretationResponse:
    """
    Get clinical interpretation of mood predictions.

    This endpoint provides:
    - DSM-5 compliant diagnoses
    - Risk stratification
    - Clinical recommendations
    - Monitoring frequency guidance

    The interpretation follows evidence-based guidelines from the Seoul study.
    """
    try:
        # Convert input features to array format
        feature_dict = features.model_dump()

        # Map to XGBoost feature order
        xgboost_features = [0.0] * len(
            XGBOOST_FEATURE_NAMES
        )  # Initialize all features dynamically

        # Validate feature mapping in debug mode
        import os

        if os.getenv("DEBUG_FEATURES", "0") == "1":
            for api_name in API_TO_XGBOOST_MAPPING:
                if api_name not in feature_dict:
                    raise ValueError(f"Missing expected API feature: {api_name}")

        for api_name, xgb_index in API_TO_XGBOOST_MAPPING.items():
            value = feature_dict.get(api_name, 0.0)
            if value is not None:
                xgboost_features[xgb_index] = float(value)

        # Get prediction (ensemble if available, XGBoost otherwise)
        import numpy as np

        feature_array = np.array(xgboost_features, dtype=np.float32)

        if orchestrator and orchestrator.xgboost_predictor.is_loaded:
            # Use ensemble prediction
            ensemble_result = orchestrator.predict(
                statistical_features=feature_array,
                activity_records=None,  # No activity data from direct API
                prediction_date=None,
            )

            ml_predictions = {
                "depression": ensemble_result.ensemble_prediction.depression_risk,
                "mania": ensemble_result.ensemble_prediction.manic_risk,
                "hypomania": ensemble_result.ensemble_prediction.hypomanic_risk,
            }
        else:
            # Fallback to XGBoost only
            prediction = predictor.predict(feature_array.astype(np.float64))
            ml_predictions = {
                "depression": prediction.depression_risk,
                "mania": prediction.manic_risk,
                "hypomania": prediction.hypomanic_risk,
            }

        # Get clinical interpretation
        interpretation = interpret_predictions(ml_predictions)

        return ClinicalInterpretationResponse(
            primary_diagnosis=interpretation.primary_diagnosis,
            risk_level=interpretation.risk_level,
            confidence=interpretation.confidence,
            clinical_notes=interpretation.clinical_notes,
            recommendations=interpretation.recommendations,
            secondary_risks=interpretation.secondary_risks,
            monitoring_frequency=interpretation.monitoring_frequency,
            ml_predictions=ml_predictions,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Clinical interpretation failed: {e}"
        ) from e
