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

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    ProcessHealthDataUseCase,
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
        use_case = container.resolve(ProcessHealthDataUseCase)

        result = use_case.execute(
            input_file=tmp_path,
            output_file=None,  # Don't save, just extract features
            extract_clinical_features=True,
            predict_mood=False,  # Just extract features, don't predict
        )

        # Extract features from the result
        if not result.clinical_features:
            raise HTTPException(
                status_code=500, detail="Failed to extract clinical features"
            )

        features = result.clinical_features
        processing_time = time.time() - start_time

        # Build response
        response = FeatureExtractionResponse(
            features=features,
            metadata={
                "filename": file.filename,
                "file_size_bytes": len(content),
                "file_type": file_ext[1:],  # Remove the dot
                "records_processed": result.metadata.get("total_records", 0),
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