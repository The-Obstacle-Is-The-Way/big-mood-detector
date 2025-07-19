"""
Feature Extraction API Routes

Direct feature extraction from health data files.
"""

import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.infrastructure.logging import get_module_logger
from big_mood_detector.infrastructure.settings import get_settings
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
    pipeline: MoodPredictionPipeline = Depends(get_mood_pipeline),
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
    if file_ext not in [".json", ".xml", ".zip"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload JSON, XML, or ZIP file.",
        )

    # Save uploaded file temporarily
    try:
        with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        # Parse the file using DataParsingService
        from big_mood_detector.application.services.data_parsing_service import (
            DataParsingService,
        )
        parsing_service = DataParsingService()

        logger.info("Starting feature extraction", filename=file.filename, file_type=file_ext)

        # Handle different file types
        if file_ext == ".zip":
            # Extract and process ZIP file
            import zipfile
            try:
                with zipfile.ZipFile(tmp_path, 'r') as zf:
                    # Check if ZIP contains JSON files
                    json_files = [f for f in zf.namelist() if f.endswith('.json')]

                    if not json_files:
                        raise HTTPException(
                            status_code=400,
                            detail="No JSON files found in ZIP archive."
                        )

                # Parse JSON files from ZIP
                parsed_health_data = parsing_service.parse_json_zip(tmp_path)
                # Convert to dict format
                parsed_data = parsing_service._format_result(parsed_health_data)

            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid ZIP file."
                ) from None

        elif file_ext == ".xml":
            # Process XML file
            parsed_health_data = parsing_service.parse_xml_export(tmp_path)
            # Convert to dict format
            parsed_data = parsing_service._format_result(parsed_health_data)

        else:
            # Single JSON file - create a clean directory for it
            import shutil
            json_dir = tmp_path.parent / "json_files"
            json_dir.mkdir(exist_ok=True)

            # Copy with original filename
            shutil.copy2(tmp_path, json_dir / file.filename)

            # Parse the directory
            parsed_health_data = parsing_service.parse_json_export(json_dir)
            # Convert to dict format for consistency
            parsed_data = parsing_service._format_result(parsed_health_data)

            # Clean up
            shutil.rmtree(json_dir)

        logger.info(
            "Parsed health data",
            sleep_records=len(parsed_data.get("sleep_records", [])),
            activity_records=len(parsed_data.get("activity_records", [])),
            heart_records=len(parsed_data.get("heart_rate_records", [])),
        )

        # Check if we have data
        if not parsed_data.get("sleep_records", []):
            raise HTTPException(
                status_code=422,
                detail="No sleep data found in the uploaded file. Please ensure the export contains sleep records.",
            )

        # 2. Determine date range from the data
        all_dates = []
        for sleep_record in parsed_data.get("sleep_records", []):
            all_dates.append(sleep_record.start_date.date())
        for activity_record in parsed_data.get("activity_records", []):
            all_dates.append(activity_record.start_date.date())
        for hr_record in parsed_data.get("heart_rate_records", []):
            all_dates.append(hr_record.timestamp.date())

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

            feature_df = pipeline.process_parsed_health_data(
                parsed_data=parsed_data,
                output_path=output_path,
                start_date=start_date,
                end_date=end_date,
            )

            # Clean up temp file
            try:
                output_path.unlink()
            except Exception:
                pass

            if feature_df.empty:
                settings = get_settings()
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not extract features from the data. Ensure you have at least {settings.MIN_OBSERVATION_DAYS} days of continuous data.",
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

                # Activity features (6)
                "daily_steps": latest_features.get("daily_steps", 0),
                "activity_variance": latest_features.get("activity_variance", 0),
                "sedentary_hours": latest_features.get("sedentary_hours", 24.0),
                "activity_fragmentation": latest_features.get("activity_fragmentation", 0),
                "sedentary_bout_mean": latest_features.get("sedentary_bout_mean", 24.0),
                "activity_intensity_ratio": latest_features.get("activity_intensity_ratio", 0),
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
                "date_range": {
                    "start": str(start_date),
                    "end": str(end_date),
                    "days": (end_date - start_date).days + 1
                },
                "records_processed": {
                    "sleep": len(parsed_data.get("sleep_records", [])),
                    "activity": len(parsed_data.get("activity_records", [])),
                    "heart_rate": len(parsed_data.get("heart_rate_records", []))
                },
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
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}") from e
    finally:
        # Clean up temporary file
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink()
