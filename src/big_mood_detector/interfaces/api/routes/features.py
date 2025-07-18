"""
Feature Extraction API Routes

Direct feature extraction from health data files.
"""

import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.application.services.data_parsing_service import DataParsingService
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
)
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)

router = APIRouter(prefix="/api/v1/features", tags=["features"])


class FeatureExtractionResponse(BaseModel):
    """Response model for feature extraction."""

    features: dict[str, float]
    metadata: dict[str, Any]
    processing_time_seconds: float
    feature_count: int


@router.post("/extract", response_model=FeatureExtractionResponse)
async def extract_features(file: UploadFile = File(...)) -> FeatureExtractionResponse:
    """
    Extract clinical features from uploaded health data.

    Accepts JSON or XML health data files and returns extracted features
    ready for mood prediction.

    Args:
        file: Uploaded health data file (JSON or XML format)

    Returns:
        Extracted features and metadata

    Raises:
        HTTPException: If file processing fails
    """
    start_time = time.time()

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".json", ".xml"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload JSON or XML file.",
        )

    # Save uploaded file temporarily
    try:
        with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        # Process the file - currently only XML is supported for feature extraction
        if file_ext != ".xml":
            raise HTTPException(
                status_code=400,
                detail="Currently only XML Apple Health exports are supported. "
                       "Please upload an export.xml file from Apple Health.",
            )
        
        # TODO: Implement full feature extraction pipeline
        # This MVP implementation returns sample features for testing
        # Full implementation should:
        # 1. Parse the XML file using DataParsingService
        # 2. Aggregate daily features using AggregationPipeline  
        # 3. Extract clinical features using ClinicalFeatureExtractor
        # 4. Return actual extracted features from the uploaded data
        processing_time = time.time() - start_time
        
        # Create a basic response to test the endpoint
        features = {
            "sleep_duration_hours": 7.5,
            "sleep_efficiency": 0.85,
            "sleep_onset_hour": 23.0,
            "wake_time_hour": 6.5,
            "sleep_midpoint_hour": 2.75,
            "sleep_regularity_index": 85.0,
            "social_jet_lag_hours": 0.5,
            "weekday_weekend_difference": 0.75,
            "total_episodes": 1,
            "fragmentation_index": 0.1,
            "wake_after_sleep_onset": 30.0,
            "longest_sleep_episode": 7.0,
            "short_sleep_percent": 0.0,
            "long_sleep_percent": 0.0,
            "heart_rate_mean": 65.0,
            "heart_rate_std": 8.0,
            "hrv_mean": 45.0,
            "hrv_std": 12.0,
            "resting_heart_rate": 58.0,
            "activity_calories": 350.0,
            "basal_calories": 1500.0,
            "total_distance_km": 5.5,
            "step_count": 8000,
            "flights_climbed": 10,
            "stand_hours": 12,
            "exercise_minutes": 30,
            "high_intensity_minutes": 10,
            "activity_level_sedentary": 600,
            "activity_level_light": 180,
            "activity_level_moderate": 30,
            "activity_level_vigorous": 10,
            "correlation_sleep_activity": 0.3,
            "phase_alignment_score": 0.8,
            "disruption_index": 0.2,
            "stability_score": 0.85,
            "quality_score": 0.8,
        }
        
        
        # Build response
        response = FeatureExtractionResponse(
            features=features,
            metadata={
                "filename": file.filename,
                "file_size_bytes": len(content),
                "file_type": file_ext[1:],  # Remove the dot
                "note": "MVP implementation - returns sample features",
            },
            processing_time_seconds=processing_time,
            feature_count=len(features),
        )

        logger.info(
            "Extracted features from file",
            filename=file.filename,
            feature_count=len(features),
            processing_time=processing_time,
        )

        return response

    except Exception as e:
        logger.error("Feature extraction failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
    finally:
        # Clean up temporary file
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink()