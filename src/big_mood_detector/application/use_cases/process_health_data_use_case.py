"""
Mood Prediction Pipeline

End-to-end integration that processes Apple Health data through
all domain services to generate the 36 features required by XGBoost.

This is the crown jewel - where everything comes together!

Design Principles:
- Orchestration layer (no business logic)
- Dependency injection for services
- Stream processing for large datasets
- Clinical validation at each step
"""

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
    DailyFeatures,
)
from big_mood_detector.application.services.data_parsing_service import (
    DataParsingService,
)
from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    EnsembleConfig,
    EnsembleOrchestrator,
)
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
)
from big_mood_detector.domain.services.circadian_rhythm_analyzer import (
    CircadianRhythmAnalyzer,
)
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
    ClinicalFeatureSet,
)
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator
from big_mood_detector.domain.services.mood_predictor import (
    MoodPredictor,
)
from big_mood_detector.domain.services.sleep_window_analyzer import SleepWindowAnalyzer
from big_mood_detector.domain.services.sparse_data_handler import (
    SparseDataHandler,
)
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for mood prediction pipeline."""

    min_days_required: int = 7
    include_pat_sequences: bool = False
    confidence_threshold: float = 0.7
    model_dir: Path | None = None
    enable_sparse_handling: bool = True
    max_interpolation_days: int = 3
    ensemble_config: EnsembleConfig | None = None
    enable_personal_calibration: bool = False
    personal_calibrator: Any | None = None  # PersonalCalibrator instance
    user_id: str | None = None


@dataclass
class PipelineResult:
    """Result of mood prediction pipeline processing."""

    daily_predictions: dict[date, dict[str, Any]]
    overall_summary: dict[str, Any]
    confidence_score: float
    processing_time_seconds: float
    records_processed: int = 0
    features_extracted: int = 0
    has_warnings: bool = False
    warnings: list[str] = field(default_factory=list)
    has_errors: bool = False
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MoodPredictionPipeline:
    """
    Orchestrates the complete mood prediction pipeline.

    This brings together all domain services to process
    raw Apple Health data into XGBoost-ready features.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        sleep_analyzer: SleepWindowAnalyzer | None = None,
        activity_extractor: ActivitySequenceExtractor | None = None,
        circadian_analyzer: CircadianRhythmAnalyzer | None = None,
        dlmo_calculator: DLMOCalculator | None = None,
        sparse_handler: SparseDataHandler | None = None,
        data_parsing_service: DataParsingService | None = None,
        aggregation_pipeline: AggregationPipeline | None = None,
    ):
        """
        Initialize with domain services.

        Uses dependency injection for testability.
        """
        self.config = config or PipelineConfig()
        self.sleep_analyzer = sleep_analyzer or SleepWindowAnalyzer()
        self.activity_extractor = activity_extractor or ActivitySequenceExtractor()
        self.circadian_analyzer = circadian_analyzer or CircadianRhythmAnalyzer()
        self.dlmo_calculator = dlmo_calculator or DLMOCalculator()
        self.sparse_handler = sparse_handler or SparseDataHandler()
        self.clinical_extractor = ClinicalFeatureExtractor()
        self.mood_predictor = MoodPredictor(model_dir=self.config.model_dir)
        self.xgboost_predictor = None  # Will be loaded separately for ensemble

        # Initialize ensemble orchestrator if PAT sequences are enabled
        self.ensemble_orchestrator = None
        if self.config.include_pat_sequences:
            from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE
            from big_mood_detector.infrastructure.ml_models.xgboost_models import (
                XGBoostMoodPredictor,
            )

            # Initialize XGBoost predictor for ensemble
            self.xgboost_predictor = XGBoostMoodPredictor()
            model_dir = self.config.model_dir or Path(
                "model_weights/xgboost/pretrained"
            )
            if self.xgboost_predictor.load_models(model_dir):
                logger.info("XGBoost models loaded for ensemble")

            # Initialize PAT model if available
            pat_model = None
            if PAT_AVAILABLE:
                from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
                pat_model = PATModel()
            else:
                logger.warning("PAT model not available - TensorFlow not installed")

            # Try to load PAT weights
            if pat_model is not None:
                if not pat_model.load_pretrained_weights():
                    logger.warning("Failed to load PAT model weights")
                    pat_model = None
                else:
                    logger.info("PAT model loaded successfully")

            if self.xgboost_predictor and self.xgboost_predictor.is_loaded:
                # Will be updated after personal calibrator is initialized
                self.ensemble_orchestrator = EnsembleOrchestrator(
                    xgboost_predictor=self.xgboost_predictor,
                    pat_model=pat_model,
                    config=self.config.ensemble_config or EnsembleConfig(),
                )

        # Data parsing service (extracted)
        self.data_parsing_service = data_parsing_service or DataParsingService()

        # Aggregation pipeline (extracted)
        self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline(
            sleep_analyzer=self.sleep_analyzer,
            activity_extractor=self.activity_extractor,
            circadian_analyzer=self.circadian_analyzer,
            dlmo_calculator=self.dlmo_calculator,
        )

        # Initialize personal calibrator
        self.personal_calibrator = None
        if self.config.enable_personal_calibration:
            if self.config.personal_calibrator:
                # Use provided calibrator
                self.personal_calibrator = self.config.personal_calibrator
            elif self.config.user_id and self.config.model_dir:
                # Try to load existing personal model
                try:
                    from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
                        PersonalCalibrator,
                    )

                    self.personal_calibrator = PersonalCalibrator.load(
                        user_id=self.config.user_id, model_dir=self.config.model_dir
                    )
                    logger.info(
                        f"Loaded personal model for user: {self.config.user_id}"
                    )
                except Exception as e:
                    logger.warning(f"Could not load personal model: {e}")
                    # Continue without personal calibration

        # Update ensemble orchestrator with personal calibrator if both exist
        if self.ensemble_orchestrator and self.personal_calibrator:
            self.ensemble_orchestrator.personal_calibrator = self.personal_calibrator

    def process_apple_health_file(
        self,
        file_path: Path,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> PipelineResult:
        """
        Process Apple Health export file and generate mood predictions.

        Args:
            file_path: Path to export.xml or JSON directory
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            PipelineResult with predictions and metadata
        """
        # Delegate parsing to DataParsingService
        parsed_data = self.data_parsing_service.parse_health_data(
            file_path=file_path,
            start_date=start_date,
            end_date=end_date,
            continue_on_error=True,
        )

        # Extract records from parsed data
        sleep_records = parsed_data.get("sleep_records", [])
        activity_records = parsed_data.get("activity_records", [])
        heart_records = parsed_data.get("heart_rate_records", [])
        errors = parsed_data.get("errors", [])

        # Process health data
        result = self.process_health_data(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=end_date or date.today(),
        )

        # Add any parsing errors to result
        if errors:
            result.errors.extend(errors)
            result.has_errors = True

        return result

    def process_health_data(
        self,
        sleep_records: list,
        activity_records: list,
        heart_records: list,
        target_date: date,
    ) -> PipelineResult:
        """
        Process health data and generate mood predictions.

        Args:
            sleep_records: List of sleep records
            activity_records: List of activity records
            heart_records: List of heart rate records
            target_date: Target date for analysis

        Returns:
            PipelineResult with predictions and metadata
        """
        start_time = time.time()
        warnings = []
        errors = []

        # Check if models are loaded
        if not self.mood_predictor.is_loaded:
            errors.append("Models not loaded")
            return PipelineResult(
                daily_predictions={},
                overall_summary={},
                confidence_score=0.0,
                processing_time_seconds=time.time() - start_time,
                has_errors=True,
                errors=errors,
            )

        # Check data sufficiency
        available_days = len({r.start_date.date() for r in sleep_records})
        if available_days < self.config.min_days_required:
            warnings.append(
                f"Insufficient data: {available_days} days available, {self.config.min_days_required} required"
            )

        # Check for sparse data
        if available_days > 0:
            date_range = (
                target_date - min(r.start_date.date() for r in sleep_records)
            ).days + 1
            density = available_days / date_range
            if density < 0.5:
                warnings.append(f"Sparse data detected: {density:.1%} density")

        # Extract features for date range
        start_date = target_date - timedelta(days=self.config.min_days_required - 1)
        features = self.extract_features_batch(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            start_date=start_date,
            end_date=target_date,
        )

        # Generate predictions
        daily_predictions = {}
        for feature_date, feature_set in features.items():
            if feature_set and feature_set.seoul_features:
                feature_vector = np.array(
                    feature_set.seoul_features.to_xgboost_features()
                )

                if self.ensemble_orchestrator:
                    # Use ensemble predictions
                    # Get activity records for the current date
                    date_activity_records = [
                        r
                        for r in activity_records
                        if r.start_date.date() <= feature_date <= r.end_date.date()
                    ]

                    ensemble_result = self.ensemble_orchestrator.predict(
                        statistical_features=feature_vector,
                        activity_records=date_activity_records,
                        prediction_date=np.datetime64(feature_date),
                    )

                    prediction = ensemble_result.ensemble_prediction

                    daily_predictions[feature_date] = {
                        "depression_risk": prediction.depression_risk,
                        "hypomanic_risk": prediction.hypomanic_risk,
                        "manic_risk": prediction.manic_risk,
                        "confidence": prediction.confidence,
                        "models_used": ensemble_result.models_used,
                        "confidence_scores": ensemble_result.confidence_scores,
                    }

                    # Add warning if PAT failed
                    if (
                        "pat" not in ensemble_result.models_used
                        or ensemble_result.pat_enhanced_prediction is None
                    ):
                        warnings.append("PAT model unavailable")
                else:
                    # Use XGBoost-only predictions
                    prediction = self.mood_predictor.predict(feature_vector)

                    daily_predictions[feature_date] = {
                        "depression_risk": prediction.depression_risk,
                        "hypomanic_risk": prediction.hypomanic_risk,
                        "manic_risk": prediction.manic_risk,
                        "confidence": prediction.confidence,
                    }

        # Calculate overall summary
        if daily_predictions:
            all_predictions = list(daily_predictions.values())
            overall_summary = {
                "avg_depression_risk": float(
                    np.mean(
                        [p["depression_risk"] for p in all_predictions]  # type: ignore[arg-type]
                    )
                ),
                "avg_hypomanic_risk": float(
                    np.mean(
                        [p["hypomanic_risk"] for p in all_predictions]  # type: ignore[arg-type]
                    )
                ),
                "avg_manic_risk": float(
                    np.mean([p["manic_risk"] for p in all_predictions])  # type: ignore[arg-type]
                ),
                "days_analyzed": len(daily_predictions),
            }
            confidence_score = float(
                np.mean([p["confidence"] for p in all_predictions])  # type: ignore[arg-type]
            )
            if np.isnan(confidence_score):
                confidence_score = 0.0
        else:
            overall_summary = {}
            confidence_score = 0.0

        # Adjust confidence based on data quality
        if warnings:
            confidence_score = float(
                confidence_score * 0.7
            )  # Reduce confidence for data issues

        # Build metadata
        metadata = {}
        if self.personal_calibrator:
            metadata["personal_calibration_used"] = True
            metadata["user_id"] = self.personal_calibrator.user_id
            metadata["baseline_available"] = bool(self.personal_calibrator.baseline)

        return PipelineResult(
            daily_predictions=daily_predictions,
            overall_summary=overall_summary,
            confidence_score=confidence_score,
            processing_time_seconds=time.time() - start_time,
            records_processed=len(sleep_records)
            + len(activity_records)
            + len(heart_records),
            features_extracted=len(features),
            has_warnings=bool(warnings),
            warnings=warnings,
            has_errors=bool(errors),
            errors=errors,
            metadata=metadata,
        )

    def extract_features_batch(
        self,
        sleep_records: list,
        activity_records: list,
        heart_records: list,
        start_date: date,
        end_date: date,
    ) -> dict[date, ClinicalFeatureSet | None]:
        """
        Extract features for multiple days efficiently.

        Args:
            sleep_records: List of sleep records
            activity_records: List of activity records
            heart_records: List of heart rate records
            start_date: Start date for extraction
            end_date: End date for extraction

        Returns:
            Dictionary mapping dates to ClinicalFeatureSet
        """
        features: dict[date, ClinicalFeatureSet | None] = {}

        current_date = start_date
        while current_date <= end_date:
            try:
                feature_set = self.clinical_extractor.extract_clinical_features(
                    sleep_records=sleep_records,
                    activity_records=activity_records,
                    heart_records=heart_records,
                    target_date=current_date,
                    include_pat_sequence=self.config.include_pat_sequences,
                )
                features[current_date] = feature_set
            except Exception as e:
                # Log error but continue processing other dates
                logger.error(
                    "feature_extraction_failed",
                    date=str(current_date),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                features[current_date] = None

            current_date += timedelta(days=1)

        return features

    def update_personal_model(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        """
        Update personal model with new labeled data.

        Args:
            features: Feature matrix
            labels: Ground truth labels
            sample_weight: Optional sample weights

        Returns:
            Dictionary of training metrics or None if no calibrator
        """
        if not self.personal_calibrator:
            logger.warning("No personal calibrator available for model update")
            return None

        # Calibrate the model
        metrics = self.personal_calibrator.calibrate(
            features=features,
            labels=labels,
            sample_weight=sample_weight or 1.0,
        )

        # Save the updated model
        self.personal_calibrator.save_model(metrics)

        return metrics  # type: ignore[no-any-return]

    def export_results(self, result: PipelineResult, output_path: Path) -> None:
        """
        Export pipeline results to CSV format.

        Args:
            result: PipelineResult to export
            output_path: Path to save CSV file
        """
        # Convert predictions to DataFrame
        rows = []
        for pred_date, prediction in result.daily_predictions.items():
            row: dict[str, Any] = {"date": pred_date}
            row.update(prediction)
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("date")

        # Save to CSV
        df.to_csv(output_path, index=False)

        # Also save summary
        summary_path = output_path.with_suffix(".summary.json")
        import json

        with open(summary_path, "w") as f:
            summary_data = {
                "overall_summary": result.overall_summary,
                "confidence_score": result.confidence_score,
                "processing_time_seconds": result.processing_time_seconds,
                "records_processed": result.records_processed,
                "warnings": result.warnings,
                "errors": result.errors,
            }

            # Add personal calibration info if available
            if result.metadata.get("personal_calibration_used"):
                summary_data["personal_calibration"] = {
                    "user_id": result.metadata.get("user_id"),
                    "baseline_available": result.metadata.get("baseline_available"),
                }

            json.dump(summary_data, f, indent=2, default=str)

    def process_health_export(
        self,
        export_path: Path,
        output_path: Path,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Process complete Apple Health export.

        Args:
            export_path: Path to export.xml or JSON directory
            output_path: Where to save the 36 features CSV
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with 36 features per day
        """
        # Use DataParsingService for all parsing operations
        parsed_data = self.data_parsing_service.parse_health_data(
            file_path=export_path,
            start_date=start_date,
            end_date=end_date,
            continue_on_error=True,
        )

        # Convert to old format for compatibility
        records = {
            "sleep": parsed_data.get("sleep_records", []),
            "activity": parsed_data.get("activity_records", []),
            "heart_rate": parsed_data.get("heart_rate_records", []),
        }

        # Validate parsed data
        validation_result = self.data_parsing_service.validate_parsed_data(parsed_data)
        if not validation_result.is_valid:
            logger.warning(
                "data_validation_failed",
                warnings=validation_result.warnings,
                warning_count=len(validation_result.warnings),
            )

        # Get data summary for analysis
        self.data_parsing_service.get_data_summary(
            parsed_data
        )  # Available for future use

        # First, analyze data density and quality
        sleep_dates = [r.start_date.date() for r in records.get("sleep", [])]
        activity_dates = [r.start_date.date() for r in records.get("activity", [])]

        logger.info("data_quality_analysis_started")
        if sleep_dates:
            sleep_density = self.sparse_handler.assess_density(sleep_dates)
            logger.info(
                "sleep_data_quality",
                days_count=len(sleep_dates),
                coverage_ratio=round(sleep_density.coverage_ratio, 3),
                max_gap_days=sleep_density.max_gap_days,
                quality=sleep_density.density_class.name,
            )

        if activity_dates:
            activity_density = self.sparse_handler.assess_density(activity_dates)
            logger.info(
                "activity_data_quality",
                days_count=len(activity_dates),
                coverage_ratio=round(activity_density.coverage_ratio, 3),
                max_gap_days=activity_density.max_gap_days,
                quality=activity_density.density_class.name,
            )

        # Find overlapping windows
        if sleep_dates and activity_dates:
            windows = self.sparse_handler.find_analysis_windows(
                sleep_dates, activity_dates
            )
            logger.info(
                "overlapping_windows_found",
                window_count=len(windows),
                sample_windows=[
                    {
                        "start": str(start),
                        "end": str(end),
                        "days": (end - start).days + 1,
                    }
                    for start, end in windows[:3]
                ],
            )

        # Extract features for each day using aggregation pipeline
        features = self._extract_daily_features(records, start_date, end_date)

        # Convert to DataFrame
        if not features:
            logger.warning(
                "no_features_extracted",
                message="Check date range and data availability",
            )
            df = pd.DataFrame()  # Empty dataframe
        else:
            df = pd.DataFrame([f.to_dict() for f in features])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # Add confidence scores based on data density
            logger.info(
                "features_extracted", days_count=len(df), adding_confidence_scores=True
            )

        # Save to CSV
        df.to_csv(output_path)
        logger.info("features_saved", days_count=len(df), output_path=str(output_path))

        return df

    def _extract_daily_features(
        self,
        records: dict[str, list],
        start_date: date | None,
        end_date: date | None,
    ) -> list[DailyFeatures]:
        """
        Extract 36 features for each day using the aggregation pipeline.

        This delegates to the AggregationPipeline service for cleaner separation of concerns.
        """
        sleep_records = records["sleep"]
        activity_records = records["activity"]
        heart_records = records.get("heart_rate", [])

        # Determine date range
        if not sleep_records:
            return []

        all_dates = [r.start_date.date() for r in sleep_records]
        min_date = start_date or min(all_dates)
        max_date = end_date or max(all_dates)

        # Use aggregation pipeline
        return self.aggregation_pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            start_date=min_date,
            end_date=max_date,
        )


# Convenience function for CLI usage
def process_health_data(
    input_path: str,
    output_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Process health data from command line.

    Args:
        input_path: Path to Apple Health export
        output_path: Path for output CSV
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        DataFrame with 36 features
    """
    pipeline = MoodPredictionPipeline()

    # Parse dates
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None

    return pipeline.process_health_export(
        Path(input_path), Path(output_path), start, end
    )
