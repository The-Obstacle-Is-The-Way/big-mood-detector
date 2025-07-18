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
from big_mood_detector.infrastructure.di.container import get_container
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

        # Process the file
        container = get_container()
        
        # Parse the data
        parsing_service: DataParsingService = container.resolve(DataParsingService)
        parsed_data = parsing_service.parse_health_data(tmp_path)
        
        # Aggregate records
        aggregation_pipeline: AggregationPipeline = container.resolve(AggregationPipeline)
        daily_features = aggregation_pipeline.aggregate_daily(
            sleep_records=parsed_data.sleep_records,
            activity_records=parsed_data.activity_records,
            heart_rate_records=parsed_data.heart_rate_records,
        )
        
        # Extract clinical features for the most recent day
        if not daily_features:
            raise HTTPException(
                status_code=400, detail="No data found in the uploaded file"
            )
        
        # Get the most recent date with data
        latest_date = max(daily_features.keys())
        latest_features = daily_features[latest_date]
        
        # Extract clinical features
        clinical_extractor: ClinicalFeatureExtractor = container.resolve(ClinicalFeatureExtractor)
        feature_set = clinical_extractor.extract_clinical_features(
            sleep_records=parsed_data.sleep_records,
            activity_records=parsed_data.activity_records,
            heart_records=parsed_data.heart_rate_records,
            target_date=latest_date,
        )
        
        if not feature_set or not feature_set.seoul_features:
            raise HTTPException(
                status_code=500, detail="Failed to extract clinical features"
            )
        
        # Convert to XGBoost features
        feature_list = feature_set.seoul_features.to_xgboost_features()
        feature_names = [
            "sleep_duration_hours",
            "sleep_efficiency",
            "sleep_onset_hour",
            "wake_time_hour",
            "sleep_midpoint_hour",
            "sleep_regularity_index",
            "social_jet_lag_hours",
            "weekday_weekend_difference",
            "total_episodes",
            "fragmentation_index",
            "wake_after_sleep_onset",
            "longest_sleep_episode",
            "short_sleep_percent",
            "long_sleep_percent",
            "heart_rate_mean",
            "heart_rate_std",
            "hrv_mean",
            "hrv_std",
            "resting_heart_rate",
            "activity_calories",
            "basal_calories",
            "total_distance_km",
            "step_count",
            "flights_climbed",
            "stand_hours",
            "exercise_minutes",
            "high_intensity_minutes",
            "activity_level_sedentary",
            "activity_level_light",
            "activity_level_moderate",
            "activity_level_vigorous",
            "correlation_sleep_activity",
            "phase_alignment_score",
            "disruption_index",
            "stability_score",
            "quality_score",
        ]
        
        # Create feature dictionary
        features = dict(zip(feature_names, feature_list))
        
        processing_time = time.time() - start_time
        
        # Build response
        response = FeatureExtractionResponse(
            features=features,
            metadata={
                "filename": file.filename,
                "file_size_bytes": len(content),
                "file_type": file_ext[1:],  # Remove the dot
                "records_processed": len(parsed_data.sleep_records) + 
                                   len(parsed_data.activity_records) + 
                                   len(parsed_data.heart_rate_records),
                "target_date": str(latest_date),
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