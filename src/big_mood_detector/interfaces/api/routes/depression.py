"""
Depression Prediction API Routes

Provides endpoints for PAT-based depression risk assessment.
Following Clean Architecture - this is an interface adapter.
"""

from datetime import datetime
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface
from big_mood_detector.infrastructure.di import get_container
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


class ActivitySequenceRequest(BaseModel):
    """Request model for activity-based depression prediction."""
    activity_sequence: list[float] = Field(
        ...,
        description="7-day minute-level activity data (10,080 values)"
    )
    
    @field_validator("activity_sequence")
    @classmethod
    def validate_sequence_length(cls, v: list[float]) -> list[float]:
        """Ensure activity sequence is exactly 7 days."""
        if len(v) != 10080:
            raise ValueError(f"Activity sequence must be exactly 10,080 timesteps (7 days), got {len(v)}")
        return v


class EmbeddingsRequest(BaseModel):
    """Request model for embeddings-based depression prediction."""
    embeddings: list[float] = Field(
        ...,
        description="Pre-computed PAT embeddings (96 dimensions)",
        min_length=96,
        max_length=96
    )


class DepressionPredictionResponse(BaseModel):
    """Response model for depression predictions."""
    depression_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Probability of depression (PHQ-9 >= 10)"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Model confidence in prediction"
    )
    model_version: str = Field(
        default="pat_conv_l_v0.5929",
        description="Model version used for prediction"
    )
    prediction_timestamp: str = Field(
        ...,
        description="ISO timestamp of prediction"
    )


class EmbeddingsPredictionResponse(DepressionPredictionResponse):
    """Extended response when using embeddings."""
    benzodiazepine_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Probability of benzodiazepine use (not yet implemented)"
    )


def get_pat_predictor() -> PATPredictorInterface:
    """Get PAT predictor from DI container."""
    container = get_container()
    return container.resolve(PATPredictorInterface)  # type: ignore[type-abstract]


@router.post("/depression", response_model=DepressionPredictionResponse)
async def predict_depression(
    request: ActivitySequenceRequest,
    predictor: Annotated[PATPredictorInterface, Depends(get_pat_predictor)]
) -> DepressionPredictionResponse:
    """
    Predict depression risk from 7-day activity sequence.
    
    This endpoint uses the PAT-Conv-L model (0.5929 AUC) to assess
    current depression risk based on wearable activity data.
    
    Args:
        request: Activity sequence data
        predictor: PAT predictor (injected)
        
    Returns:
        Depression probability and confidence
        
    Raises:
        503: Model not loaded
        500: Prediction failed
    """
    # Check if model is loaded
    if not predictor.is_loaded:
        logger.error("PAT model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Depression prediction model not loaded"
        )
    
    try:
        # Convert to numpy array
        activity_array = np.array(request.activity_sequence, dtype=np.float32)
        
        # Get prediction
        depression_prob = predictor.predict_depression(activity_array)
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(depression_prob - 0.5) * 2.0
        confidence = min(confidence, 0.95)  # Cap at 95%
        
        logger.info(
            f"Depression prediction: {depression_prob:.3f} "
            f"(confidence: {confidence:.3f})"
        )
        
        return DepressionPredictionResponse(
            depression_probability=depression_prob,
            confidence=confidence,
            model_version="pat_conv_l_v0.5929",
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Depression prediction failed"
        )


@router.post("/depression/from-embeddings", response_model=EmbeddingsPredictionResponse)
async def predict_depression_from_embeddings(
    request: EmbeddingsRequest,
    predictor: Annotated[PATPredictorInterface, Depends(get_pat_predictor)]
) -> EmbeddingsPredictionResponse:
    """
    Predict depression risk from pre-computed PAT embeddings.
    
    This is useful when embeddings have been cached or computed elsewhere.
    
    Args:
        request: PAT embeddings
        predictor: PAT predictor (injected)
        
    Returns:
        Depression and medication predictions with confidence
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Depression prediction model not loaded"
        )
    
    try:
        # Convert to numpy array
        embeddings_array = np.array(request.embeddings, dtype=np.float32)
        
        # Get predictions
        predictions = predictor.predict_from_embeddings(embeddings_array)
        
        logger.info(
            f"Embeddings prediction - Depression: {predictions.depression_probability:.3f}, "
            f"Confidence: {predictions.confidence:.3f}"
        )
        
        return EmbeddingsPredictionResponse(
            depression_probability=predictions.depression_probability,
            benzodiazepine_probability=predictions.benzodiazepine_probability,
            confidence=predictions.confidence,
            model_version="pat_conv_l_v0.5929",
            prediction_timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Embeddings prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Depression prediction failed"
        )