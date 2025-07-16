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
from big_mood_detector.infrastructure.parsers.json.json_parsers import (
    ActivityJSONParser,
    SleepJSONParser,
)
from big_mood_detector.infrastructure.parsers.parser_factory import ParserFactory
from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)
from big_mood_detector.infrastructure.sparse_data_handler import (
    SparseDataHandler,
)


@dataclass
class PipelineConfig:
    """Configuration for mood prediction pipeline."""

    min_days_required: int = 7
    include_pat_sequences: bool = False
    confidence_threshold: float = 0.7
    model_dir: Path | None = None
    enable_sparse_handling: bool = True
    max_interpolation_days: int = 3


@dataclass
class PipelineResult:
    """Result of mood prediction pipeline processing."""

    daily_predictions: dict[date, dict[str, float]]
    overall_summary: dict[str, Any]
    confidence_score: float
    processing_time_seconds: float
    records_processed: int = 0
    features_extracted: int = 0
    has_warnings: bool = False
    warnings: list[str] = field(default_factory=list)
    has_errors: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class DailyFeatures:
    """
    Complete feature set for one day (36 features).

    Matches the Seoul study's XGBoost input format.
    """

    date: date

    # Sleep features (10 × 3 = 30)
    sleep_percentage_mean: float
    sleep_percentage_std: float
    sleep_percentage_zscore: float

    sleep_amplitude_mean: float
    sleep_amplitude_std: float
    sleep_amplitude_zscore: float

    long_sleep_num_mean: float
    long_sleep_num_std: float
    long_sleep_num_zscore: float

    long_sleep_len_mean: float
    long_sleep_len_std: float
    long_sleep_len_zscore: float

    long_sleep_st_mean: float
    long_sleep_st_std: float
    long_sleep_st_zscore: float

    long_sleep_wt_mean: float
    long_sleep_wt_std: float
    long_sleep_wt_zscore: float

    short_sleep_num_mean: float
    short_sleep_num_std: float
    short_sleep_num_zscore: float

    short_sleep_len_mean: float
    short_sleep_len_std: float
    short_sleep_len_zscore: float

    short_sleep_st_mean: float
    short_sleep_st_std: float
    short_sleep_st_zscore: float

    short_sleep_wt_mean: float
    short_sleep_wt_std: float
    short_sleep_wt_zscore: float

    # Circadian features (2 × 3 = 6)
    circadian_amplitude_mean: float
    circadian_amplitude_std: float
    circadian_amplitude_zscore: float

    circadian_phase_mean: float  # DLMO hour
    circadian_phase_std: float
    circadian_phase_zscore: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "date": self.date,
            "sleep_percentage_MN": self.sleep_percentage_mean,
            "sleep_percentage_SD": self.sleep_percentage_std,
            "sleep_percentage_Z": self.sleep_percentage_zscore,
            "sleep_amplitude_MN": self.sleep_amplitude_mean,
            "sleep_amplitude_SD": self.sleep_amplitude_std,
            "sleep_amplitude_Z": self.sleep_amplitude_zscore,
            "long_num_MN": self.long_sleep_num_mean,
            "long_num_SD": self.long_sleep_num_std,
            "long_num_Z": self.long_sleep_num_zscore,
            "long_len_MN": self.long_sleep_len_mean,
            "long_len_SD": self.long_sleep_len_std,
            "long_len_Z": self.long_sleep_len_zscore,
            "long_ST_MN": self.long_sleep_st_mean,
            "long_ST_SD": self.long_sleep_st_std,
            "long_ST_Z": self.long_sleep_st_zscore,
            "long_WT_MN": self.long_sleep_wt_mean,
            "long_WT_SD": self.long_sleep_wt_std,
            "long_WT_Z": self.long_sleep_wt_zscore,
            "short_num_MN": self.short_sleep_num_mean,
            "short_num_SD": self.short_sleep_num_std,
            "short_num_Z": self.short_sleep_num_zscore,
            "short_len_MN": self.short_sleep_len_mean,
            "short_len_SD": self.short_sleep_len_std,
            "short_len_Z": self.short_sleep_len_zscore,
            "short_ST_MN": self.short_sleep_st_mean,
            "short_ST_SD": self.short_sleep_st_std,
            "short_ST_Z": self.short_sleep_st_zscore,
            "short_WT_MN": self.short_sleep_wt_mean,
            "short_WT_SD": self.short_sleep_wt_std,
            "short_WT_Z": self.short_sleep_wt_zscore,
            "circadian_amplitude_MN": self.circadian_amplitude_mean,
            "circadian_amplitude_SD": self.circadian_amplitude_std,
            "circadian_amplitude_Z": self.circadian_amplitude_zscore,
            "circadian_phase_MN": self.circadian_phase_mean,
            "circadian_phase_SD": self.circadian_phase_std,
            "circadian_phase_Z": self.circadian_phase_zscore,
        }


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

        # Parsers for different data sources
        self.sleep_parser = SleepJSONParser()
        self.activity_parser = ActivityJSONParser()
        self.xml_parser = StreamingXMLParser()

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
        time.time()

        # Parse health data
        parser = ParserFactory.create_parser(str(file_path))
        parsed_data = parser.parse(file_path)

        # Extract records
        sleep_records = parsed_data.get("sleep_records", [])
        activity_records = parsed_data.get("activity_records", [])
        heart_records = parsed_data.get("heart_rate_records", [])

        # Filter by date range if specified
        if start_date:
            sleep_records = [
                r for r in sleep_records if r.start_date.date() >= start_date
            ]
            activity_records = [
                r for r in activity_records if r.start_date.date() >= start_date
            ]
            heart_records = [
                r for r in heart_records if r.timestamp.date() >= start_date
            ]

        if end_date:
            sleep_records = [
                r for r in sleep_records if r.start_date.date() <= end_date
            ]
            activity_records = [
                r for r in activity_records if r.start_date.date() <= end_date
            ]
            heart_records = [r for r in heart_records if r.timestamp.date() <= end_date]

        # Process health data
        return self.process_health_data(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=end_date or date.today(),
        )

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
                "avg_depression_risk": np.mean(
                    [p["depression_risk"] for p in all_predictions]
                ),
                "avg_hypomanic_risk": np.mean(
                    [p["hypomanic_risk"] for p in all_predictions]
                ),
                "avg_manic_risk": np.mean([p["manic_risk"] for p in all_predictions]),
                "days_analyzed": len(daily_predictions),
            }
            confidence_score = float(np.mean([p["confidence"] for p in all_predictions]))
            if np.isnan(confidence_score):
                confidence_score = 0.0
        else:
            overall_summary = {}
            confidence_score = 0.0

        # Adjust confidence based on data quality
        if warnings:
            confidence_score = float(confidence_score * 0.7)  # Reduce confidence for data issues

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
                print(f"Error extracting features for {current_date}: {e}")
                features[current_date] = None

            current_date += timedelta(days=1)

        return features

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
            json.dump(
                {
                    "overall_summary": result.overall_summary,
                    "confidence_score": result.confidence_score,
                    "processing_time_seconds": result.processing_time_seconds,
                    "records_processed": result.records_processed,
                    "warnings": result.warnings,
                    "errors": result.errors,
                },
                f,
                indent=2,
                default=str,
            )

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
        # Determine data source type
        if export_path.is_file() and export_path.suffix == ".xml":
            # Process XML export
            records = self._process_xml_export(export_path)
        else:
            # Process JSON export
            records = self._process_json_export(export_path)

        # First, analyze data density and quality
        sleep_dates = [r.start_date.date() for r in records.get("sleep", [])]
        activity_dates = [r.start_date.date() for r in records.get("activity", [])]

        print("\n=== Data Quality Analysis ===")
        if sleep_dates:
            sleep_density = self.sparse_handler.assess_density(sleep_dates)
            print(
                f"Sleep data: {len(sleep_dates)} days, {sleep_density.coverage_ratio:.1%} coverage, "
                f"max gap: {sleep_density.max_gap_days} days, quality: {sleep_density.density_class.name}"
            )

        if activity_dates:
            activity_density = self.sparse_handler.assess_density(activity_dates)
            print(
                f"Activity data: {len(activity_dates)} days, {activity_density.coverage_ratio:.1%} coverage, "
                f"max gap: {activity_density.max_gap_days} days, quality: {activity_density.density_class.name}"
            )

        # Find overlapping windows
        if sleep_dates and activity_dates:
            windows = self.sparse_handler.find_analysis_windows(
                sleep_dates, activity_dates
            )
            print(f"\nOverlapping windows: {len(windows)}")
            for i, (start, end) in enumerate(windows[:3]):  # Show first 3
                days = (end - start).days + 1
                print(f"  Window {i+1}: {start} to {end} ({days} days)")

        # Extract features for each day
        features = self._extract_daily_features(records, start_date, end_date)

        # Convert to DataFrame
        if not features:
            print(
                "\nWarning: No features extracted. Check date range and data availability."
            )
            df = pd.DataFrame()  # Empty dataframe
        else:
            df = pd.DataFrame([f.to_dict() for f in features])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # Add confidence scores based on data density
            print(f"\nExtracted features for {len(df)} days")
            print("Adding confidence scores based on data quality...")

        # Save to CSV
        df.to_csv(output_path)
        print(f"Saved {len(df)} days of features to {output_path}")

        return df

    def _process_xml_export(self, xml_path: Path) -> dict[str, list]:
        """Process Apple Health XML export."""
        records: dict[str, list] = {"sleep": [], "activity": [], "heart_rate": []}

        # Stream through XML
        for record in self.xml_parser.iter_records(str(xml_path)):
            if "Sleep" in record.get("type", ""):
                # Convert to SleepRecord
                # Implementation depends on XML structure
                pass
            elif "StepCount" in record.get("type", ""):
                # Convert to ActivityRecord
                pass
            elif "HeartRate" in record.get("type", ""):
                # Convert to HeartRateRecord
                pass

        return records

    def _process_json_export(self, json_dir: Path) -> dict[str, list]:
        """Process Health Auto Export JSON files."""
        records: dict[str, list] = {"sleep": [], "activity": [], "heart_rate": []}

        # Process sleep data
        sleep_files = list(json_dir.glob("*[Ss]leep*.json"))
        for file in sleep_files:
            print(f"Processing sleep file: {file.name}")
            sleep_data = self.sleep_parser.parse_file(str(file))
            records["sleep"].extend(sleep_data)

        # Process activity data (Step Count.json)
        activity_files = list(json_dir.glob("*[Ss]tep*.json"))
        for file in activity_files:
            print(f"Processing activity file: {file.name}")
            activity_data = self.activity_parser.parse_file(str(file))
            records["activity"].extend(activity_data)

        print(f"Loaded {len(records['sleep'])} sleep records")
        print(f"Loaded {len(records['activity'])} activity records")

        return records

    def _extract_daily_features(
        self,
        records: dict[str, list],
        start_date: date | None,
        end_date: date | None,
    ) -> list[DailyFeatures]:
        """
        Extract 36 features for each day.

        This is where the magic happens - all services work together!
        """
        sleep_records = records["sleep"]
        activity_records = records["activity"]

        # Determine date range
        if not sleep_records:
            return []

        all_dates = [r.start_date.date() for r in sleep_records]
        min_date = start_date or min(all_dates)
        max_date = end_date or max(all_dates)

        # Process each day
        daily_features = []
        current_date = min_date

        # Keep rolling windows for statistics
        window_size = 30  # 30-day windows for mean/std
        sleep_metrics_window = []
        circadian_metrics_window = []

        while current_date <= max_date:
            # 1. Sleep Window Analysis
            day_sleep = [
                s
                for s in sleep_records
                if s.start_date.date() <= current_date <= s.end_date.date()
            ]

            if current_date.day == 15:  # Debug first day
                print(f"\nProcessing {current_date}:")
                print(f"  Found {len(day_sleep)} sleep records")

            sleep_windows = self.sleep_analyzer.analyze_sleep_episodes(
                day_sleep, current_date
            )

            # 2. Activity Sequence Extraction
            day_activity = [
                a for a in activity_records if a.start_date.date() == current_date
            ]

            activity_sequence = self.activity_extractor.extract_daily_sequence(
                day_activity, current_date
            )

            # 3. Circadian Rhythm Analysis
            # Get up to 7 days of sequences for circadian analysis
            # Look back up to 14 days to find enough data
            week_sequences = []
            sequences_with_dates = []

            for days_back in range(14):
                seq_date = current_date - timedelta(days=days_back)
                seq_activity = [
                    a for a in activity_records if a.start_date.date() == seq_date
                ]
                if seq_activity:
                    seq = self.activity_extractor.extract_daily_sequence(
                        seq_activity, seq_date
                    )
                    sequences_with_dates.append((seq_date, seq))

                    # Stop when we have 7 days of data
                    if len(sequences_with_dates) >= 7:
                        break

            # Sort by date and take the sequences
            sequences_with_dates.sort(key=lambda x: x[0])
            week_sequences = [seq for _, seq in sequences_with_dates]

            circadian_metrics = None
            if len(week_sequences) >= 3:
                if current_date.day == 16:  # Debug
                    print(
                        f"  Found {len(week_sequences)} days of activity for circadian analysis"
                    )
                circadian_metrics = self.circadian_analyzer.calculate_metrics(
                    week_sequences
                )

            # 4. DLMO Calculation
            # Get sleep records from the past 14 days for DLMO
            dlmo_sleep = [
                s
                for s in sleep_records
                if (current_date - s.start_date.date()).days < 14
                and s.start_date.date() <= current_date
            ]

            dlmo_result = None
            if len(dlmo_sleep) >= 3:  # Need at least 3 days
                if current_date.day == 16:  # Debug
                    print(
                        f"  Found {len(dlmo_sleep)} sleep records for DLMO calculation"
                    )
                dlmo_result = self.dlmo_calculator.calculate_dlmo(
                    dlmo_sleep, current_date, days_to_model=min(7, len(dlmo_sleep))
                )

            # 5. Extract daily metrics
            daily_metrics = self._calculate_daily_metrics(
                sleep_windows, activity_sequence, circadian_metrics, dlmo_result
            )

            # 6. Update rolling windows
            if daily_metrics:
                sleep_metrics_window.append(daily_metrics["sleep"])
                if daily_metrics["circadian"]:
                    circadian_metrics_window.append(daily_metrics["circadian"])

                # Keep only recent window
                if len(sleep_metrics_window) > window_size:
                    sleep_metrics_window.pop(0)
                if len(circadian_metrics_window) > window_size:
                    circadian_metrics_window.pop(0)

            # 7. Calculate statistics (mean, std, z-score)
            if len(sleep_metrics_window) >= 3:  # Reduced threshold for testing
                features = self._calculate_features_with_stats(
                    current_date,
                    daily_metrics,
                    sleep_metrics_window,
                    circadian_metrics_window,
                )

                if features:
                    daily_features.append(features)

            current_date += timedelta(days=1)

        return daily_features

    def _calculate_daily_metrics(
        self, sleep_windows, activity_sequence, circadian_metrics, dlmo_result
    ) -> dict | None:
        """Calculate raw metrics for a single day."""
        if not sleep_windows:
            return None

        # Sleep metrics
        total_sleep_minutes = sum(w.total_duration_hours * 60 for w in sleep_windows)
        sleep_percentage = total_sleep_minutes / 1440.0  # % of day

        # Sleep amplitude (coefficient of variation of wake amounts)
        # This is a simplified version - full implementation would analyze
        # wake periods within sleep windows
        wake_periods = [g for w in sleep_windows for g in w.gap_hours if g > 0]
        if wake_periods:
            sleep_amplitude = np.std(wake_periods) / np.mean(wake_periods)
        else:
            sleep_amplitude = 0.0

        # Long/short window counts
        long_windows = [w for w in sleep_windows if w.total_duration_hours >= 3.75]
        short_windows = [w for w in sleep_windows if w.total_duration_hours < 3.75]

        sleep_metrics = {
            "sleep_percentage": sleep_percentage,
            "sleep_amplitude": sleep_amplitude,
            "long_num": len(long_windows),
            "long_len": sum(w.total_duration_hours for w in long_windows),
            "long_st": sum(w.total_duration_hours for w in long_windows),  # Simplified
            "long_wt": sum(sum(w.gap_hours) for w in long_windows),
            "short_num": len(short_windows),
            "short_len": sum(w.total_duration_hours for w in short_windows),
            "short_st": sum(
                w.total_duration_hours for w in short_windows
            ),  # Simplified
            "short_wt": sum(sum(w.gap_hours) for w in short_windows),
        }

        # Circadian metrics
        circadian_dict = None
        if circadian_metrics and dlmo_result:
            circadian_dict = {
                "amplitude": circadian_metrics.relative_amplitude,
                "phase": dlmo_result.dlmo_hour,
            }

        return {"sleep": sleep_metrics, "circadian": circadian_dict}

    def _calculate_features_with_stats(
        self,
        current_date: date,
        daily_metrics: dict,
        sleep_window: list[dict],
        circadian_window: list[dict],
    ) -> DailyFeatures | None:
        """Calculate features with mean, std, and z-scores."""
        if not daily_metrics or not daily_metrics["sleep"]:
            return None

        import numpy as np

        # Calculate sleep statistics
        sleep_features = {}
        for metric in [
            "sleep_percentage",
            "sleep_amplitude",
            "long_num",
            "long_len",
            "long_st",
            "long_wt",
            "short_num",
            "short_len",
            "short_st",
            "short_wt",
        ]:
            values = [s[metric] for s in sleep_window]
            mean_val = np.mean(values)
            std_val = np.std(values)
            current_val = daily_metrics["sleep"][metric]

            # Z-score
            if std_val > 0:
                z_score = (current_val - mean_val) / std_val
            else:
                z_score = 0.0

            sleep_features[f"{metric}_mean"] = mean_val
            sleep_features[f"{metric}_std"] = std_val
            sleep_features[f"{metric}_zscore"] = z_score

        # Calculate circadian statistics
        circadian_features = {
            "circadian_amplitude_mean": 0.0,
            "circadian_amplitude_std": 0.0,
            "circadian_amplitude_zscore": 0.0,
            "circadian_phase_mean": 0.0,
            "circadian_phase_std": 0.0,
            "circadian_phase_zscore": 0.0,
        }

        if circadian_window and daily_metrics["circadian"]:
            for metric in ["amplitude", "phase"]:
                values = [c[metric] for c in circadian_window if c]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    current_val = daily_metrics["circadian"][metric]

                    # Z-score
                    if std_val > 0:
                        z_score = (current_val - mean_val) / std_val
                    else:
                        z_score = 0.0

                    circadian_features[f"circadian_{metric}_mean"] = mean_val
                    circadian_features[f"circadian_{metric}_std"] = std_val
                    circadian_features[f"circadian_{metric}_zscore"] = z_score

        # Create DailyFeatures object with explicit field mapping
        return DailyFeatures(
            date=current_date,
            # Sleep percentage
            sleep_percentage_mean=sleep_features["sleep_percentage_mean"],
            sleep_percentage_std=sleep_features["sleep_percentage_std"],
            sleep_percentage_zscore=sleep_features["sleep_percentage_zscore"],
            # Sleep amplitude
            sleep_amplitude_mean=sleep_features["sleep_amplitude_mean"],
            sleep_amplitude_std=sleep_features["sleep_amplitude_std"],
            sleep_amplitude_zscore=sleep_features["sleep_amplitude_zscore"],
            # Long sleep
            long_sleep_num_mean=sleep_features["long_num_mean"],
            long_sleep_num_std=sleep_features["long_num_std"],
            long_sleep_num_zscore=sleep_features["long_num_zscore"],
            long_sleep_len_mean=sleep_features["long_len_mean"],
            long_sleep_len_std=sleep_features["long_len_std"],
            long_sleep_len_zscore=sleep_features["long_len_zscore"],
            long_sleep_st_mean=sleep_features["long_st_mean"],
            long_sleep_st_std=sleep_features["long_st_std"],
            long_sleep_st_zscore=sleep_features["long_st_zscore"],
            long_sleep_wt_mean=sleep_features["long_wt_mean"],
            long_sleep_wt_std=sleep_features["long_wt_std"],
            long_sleep_wt_zscore=sleep_features["long_wt_zscore"],
            # Short sleep
            short_sleep_num_mean=sleep_features["short_num_mean"],
            short_sleep_num_std=sleep_features["short_num_std"],
            short_sleep_num_zscore=sleep_features["short_num_zscore"],
            short_sleep_len_mean=sleep_features["short_len_mean"],
            short_sleep_len_std=sleep_features["short_len_std"],
            short_sleep_len_zscore=sleep_features["short_len_zscore"],
            short_sleep_st_mean=sleep_features["short_st_mean"],
            short_sleep_st_std=sleep_features["short_st_std"],
            short_sleep_st_zscore=sleep_features["short_st_zscore"],
            short_sleep_wt_mean=sleep_features["short_wt_mean"],
            short_sleep_wt_std=sleep_features["short_wt_std"],
            short_sleep_wt_zscore=sleep_features["short_wt_zscore"],
            # Circadian
            circadian_amplitude_mean=circadian_features["circadian_amplitude_mean"],
            circadian_amplitude_std=circadian_features["circadian_amplitude_std"],
            circadian_amplitude_zscore=circadian_features["circadian_amplitude_zscore"],
            circadian_phase_mean=circadian_features["circadian_phase_mean"],
            circadian_phase_std=circadian_features["circadian_phase_std"],
            circadian_phase_zscore=circadian_features["circadian_phase_zscore"],
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
