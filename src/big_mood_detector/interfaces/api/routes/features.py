"""
Feature Extraction API Routes

Direct feature extraction from health data files.
"""

import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from pydantic import BaseModel

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.application.services.data_parsing_service import DataParsingService
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
)
from big_mood_detector.infrastructure.logging import get_module_logger
from big_mood_detector.interfaces.api.dependencies import get_mood_pipeline

logger = get_module_logger(__name__)

router = APIRouter(prefix="/api/v1/features", tags=["features"])


class FeatureExtractionResponse(BaseModel):
    """Response model for feature extraction."""

    features: dict[str, float]
    metadata: dict[str, Any]
    processing_time_seconds: float
    feature_count: int


@router.post("/extract", response_model=FeatureExtractionResponse)
async def extract_features(
    file: UploadFile = File(...),
    pipeline = Depends(get_mood_pipeline),
) -> FeatureExtractionResponse:
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
        
        # REAL feature extraction pipeline
        logger.info("Starting real feature extraction", filename=file.filename)
        
        # 1. Parse the XML file using DataParsingService
        from big_mood_detector.application.services.data_parsing_service import DataParsingService
        parsing_service = DataParsingService()
        parsed_data = parsing_service.parse_xml_export(tmp_path)
        
        logger.info(
            "Parsed health data",
            sleep_records=len(parsed_data.sleep_records),
            activity_records=len(parsed_data.activity_records),
            heart_records=len(parsed_data.heart_rate_records),
        )
        
        # Check if we have data
        if not parsed_data.sleep_records:
            raise HTTPException(
                status_code=422,
                detail="No sleep data found in the uploaded file. Please ensure the export contains sleep records.",
            )
        
        # 2. Determine date range from the data
        from datetime import date as dt_date
        all_dates = []
        for record in parsed_data.sleep_records:
            all_dates.append(record.start_date.date())
        for record in parsed_data.activity_records:
            all_dates.append(record.start_date.date())
        for record in parsed_data.heart_rate_records:
            all_dates.append(record.timestamp.date())
        
        if not all_dates:
            raise HTTPException(
                status_code=422,
                detail="No valid dates found in the health data.",
            )
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        logger.info(
            "Date range determined",
            start_date=str(start_date),
            end_date=str(end_date),
            days=(end_date - start_date).days + 1,
        )
        
        # 3. Use injected MoodPredictionPipeline for feature extraction
        
        # Extract features using the pipeline
        try:
            # Process the data and extract features
            # Create a temporary output file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_output:
                output_path = Path(tmp_output.name)
            
            feature_df = pipeline.process_health_export(
                export_path=tmp_path,
                output_path=output_path,
                start_date=start_date,
                end_date=end_date,
            )
            
            # Clean up temp file
            try:
                output_path.unlink()
            except:
                pass
            
            if feature_df.empty:
                raise HTTPException(
                    status_code=422,
                    detail="Could not extract features from the data. Ensure you have at least 7 days of continuous data.",
                )
            
            # Get the latest features
            latest_features = feature_df.iloc[-1].to_dict()
            
            # Return the raw 36-feature vector with proper naming
            # This matches the Seoul study exactly
            features = {
                # Sleep percentage features (3)
                "sleep_percentage_mean": latest_features.get("sleep_percentage_MN", 0),
                "sleep_percentage_std": latest_features.get("sleep_percentage_SD", 0),
                "sleep_percentage_zscore": latest_features.get("sleep_percentage_Z", 0),
                
                # Sleep amplitude features (3)
                "sleep_amplitude_mean": latest_features.get("sleep_amplitude_MN", 0),
                "sleep_amplitude_std": latest_features.get("sleep_amplitude_SD", 0),
                "sleep_amplitude_zscore": latest_features.get("sleep_amplitude_Z", 0),
                
                # Long sleep window features (12)
                "long_sleep_num_mean": latest_features.get("long_num_MN", 0),
                "long_sleep_num_std": latest_features.get("long_num_SD", 0),
                "long_sleep_num_zscore": latest_features.get("long_num_Z", 0),
                "long_sleep_len_mean": latest_features.get("long_len_MN", 0),
                "long_sleep_len_std": latest_features.get("long_len_SD", 0),
                "long_sleep_len_zscore": latest_features.get("long_len_Z", 0),
                "long_sleep_st_mean": latest_features.get("long_ST_MN", 0),
                "long_sleep_st_std": latest_features.get("long_ST_SD", 0),
                "long_sleep_st_zscore": latest_features.get("long_ST_Z", 0),
                "long_sleep_wt_mean": latest_features.get("long_WT_MN", 0),
                "long_sleep_wt_std": latest_features.get("long_WT_SD", 0),
                "long_sleep_wt_zscore": latest_features.get("long_WT_Z", 0),
                
                # Short sleep window features (12)
                "short_sleep_num_mean": latest_features.get("short_num_MN", 0),
                "short_sleep_num_std": latest_features.get("short_num_SD", 0),
                "short_sleep_num_zscore": latest_features.get("short_num_Z", 0),
                "short_sleep_len_mean": latest_features.get("short_len_MN", 0),
                "short_sleep_len_std": latest_features.get("short_len_SD", 0),
                "short_sleep_len_zscore": latest_features.get("short_len_Z", 0),
                "short_sleep_st_mean": latest_features.get("short_ST_MN", 0),
                "short_sleep_st_std": latest_features.get("short_ST_SD", 0),
                "short_sleep_st_zscore": latest_features.get("short_ST_Z", 0),
                "short_sleep_wt_mean": latest_features.get("short_WT_MN", 0),
                "short_sleep_wt_std": latest_features.get("short_WT_SD", 0),
                "short_sleep_wt_zscore": latest_features.get("short_WT_Z", 0),
                
                # Circadian features (6)
                "circadian_amplitude_mean": latest_features.get("circadian_amplitude_MN", 0),
                "circadian_amplitude_std": latest_features.get("circadian_amplitude_SD", 0),
                "circadian_amplitude_zscore": latest_features.get("circadian_amplitude_Z", 0),
                "circadian_phase_mean": latest_features.get("circadian_phase_MN", 0),
                "circadian_phase_std": latest_features.get("circadian_phase_SD", 0),
                "circadian_phase_zscore": latest_features.get("circadian_phase_Z", 0),
            }
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Feature extraction completed",
                features_extracted=len(features),
                processing_time=processing_time,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Feature extraction pipeline failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Feature extraction failed: {str(e)}",
            ) from e
        
        
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