#!/usr/bin/env python3
"""
Command-line interface for making mood predictions.

This command processes Apple Health data and generates mood predictions
using pre-trained models, with optional personalization.

Usage:
    python -m big_mood_detector.cli.predict \
        --input /path/to/export.xml \
        --output predictions.json \
        --format json
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
    PipelineResult,
)
from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
    PersonalCalibrator,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def format_risk_level(risk_score: float) -> str:
    """Format risk score with level."""
    if risk_score >= 0.7:
        return f"{risk_score:.1%} [HIGH]"
    elif risk_score >= 0.4:
        return f"{risk_score:.1%} [MODERATE]"
    else:
        return f"{risk_score:.1%} [LOW]"


def print_summary(result: PipelineResult) -> None:
    """Print prediction summary to console."""
    print("\n" + "=" * 60)
    print("MOOD PREDICTION RESULTS".center(60))
    print("=" * 60)

    # Overall summary
    if result.overall_summary:
        print("\n📊 Overall Risk Summary:")
        print("-" * 30)

        # Use the actual field names from the pipeline
        dep_risk = result.overall_summary.get("avg_depression_risk", 0)
        hypo_risk = result.overall_summary.get("avg_hypomanic_risk", 0)
        manic_risk = result.overall_summary.get("avg_manic_risk", 0)

        print(f"Depression: {format_risk_level(dep_risk)}")
        print(f"Hypomanic: {format_risk_level(hypo_risk)}")
        print(f"Manic: {format_risk_level(manic_risk)}")

        # Count days at risk
        total_days = len(result.daily_predictions)
        if total_days > 0:
            days_at_risk = {
                "depression": sum(
                    1
                    for d in result.daily_predictions.values()
                    if d.get("depression_risk", 0) >= 0.4
                ),
                "hypomanic": sum(
                    1
                    for d in result.daily_predictions.values()
                    if d.get("hypomanic_risk", 0) >= 0.4
                ),
                "manic": sum(
                    1
                    for d in result.daily_predictions.values()
                    if d.get("manic_risk", 0) >= 0.3
                ),
            }
            print("\nDays at risk:")
            print(f"  Depression: {days_at_risk['depression']}/{total_days}")
            print(f"  Hypomanic: {days_at_risk['hypomanic']}/{total_days}")
            print(f"  Manic: {days_at_risk['manic']}/{total_days}")

    # Daily predictions (first 7 days)
    if result.daily_predictions:
        print("\n📅 Recent Daily Predictions:")
        print("-" * 30)

        sorted_dates = sorted(result.daily_predictions.keys(), reverse=True)
        for pred_date in sorted_dates[:7]:
            predictions = result.daily_predictions[pred_date]
            print(f"\n{pred_date}:")
            print(
                f"  Depression: {format_risk_level(predictions.get('depression_risk', 0))}"
            )
            print(f"  Hypomanic: {predictions.get('hypomanic_risk', 0):.1%}")
            print(f"  Manic: {predictions.get('manic_risk', 0):.1%}")

    # Metadata
    print("\n📈 Processing Metadata:")
    print("-" * 30)
    print(f"Confidence: {result.confidence_score:.1%}")
    print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
    print(f"Records processed: {result.records_processed:,}")
    print(f"Features extracted: {result.features_extracted}")

    # Warnings
    if result.warnings:
        print("\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")

    print("\n" + "=" * 60)


def save_json_output(result: PipelineResult, output_path: Path) -> None:
    """Save results as JSON."""
    import numpy as np

    def convert_to_serializable(obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.float32 | np.float64):
            return float(obj)
        elif isinstance(obj, np.int32 | np.int64):
            return int(obj)
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    # Convert dates to strings and handle numpy types
    data = {
        "daily_predictions": {
            str(date): convert_to_serializable(predictions)
            for date, predictions in result.daily_predictions.items()
        },
        "overall_summary": convert_to_serializable(result.overall_summary),
        "confidence_score": float(result.confidence_score),
        "processing_time_seconds": float(result.processing_time_seconds),
        "records_processed": int(result.records_processed),
        "features_extracted": int(result.features_extracted),
        "warnings": result.warnings,
        "errors": result.errors,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv_output(result: PipelineResult, output_path: Path) -> None:
    """Save results as CSV."""
    import pandas as pd

    # Create dataframe from daily predictions
    rows = []
    for pred_date, predictions in result.daily_predictions.items():
        row: dict[str, Any] = {"date": pred_date}
        row.update(predictions)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def generate_clinical_report(result: PipelineResult, output_path: Path) -> None:
    """Generate detailed clinical report."""
    with open(output_path, "w") as f:
        f.write("# Clinical Assessment Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Risk assessment
        f.write("## Risk Assessment\n\n")

        if result.overall_summary:
            dep_risk = result.overall_summary.get("avg_depression_risk", 0)
            hypo_risk = result.overall_summary.get("avg_hypomanic_risk", 0)
            manic_risk = result.overall_summary.get("avg_manic_risk", 0)

            # Determine primary concern
            if dep_risk >= 0.7:
                f.write("### ⚠️ Depression Risk: HIGH\n\n")
                f.write("**Immediate Actions Recommended:**\n")
                f.write("- Schedule clinical evaluation within 48 hours\n")
                f.write("- Assess for suicidal ideation\n")
                f.write("- Review current medications\n")
                f.write("- Evaluate sleep hygiene and daily routine\n\n")
            elif hypo_risk >= 0.5 or manic_risk >= 0.3:
                f.write("### ⚠️ Elevated Mood Episode Risk\n\n")
                f.write("**Clinical Considerations:**\n")
                f.write("- Monitor for decreased sleep need\n")
                f.write("- Track goal-directed activity\n")
                f.write("- Assess medication adherence\n")
                f.write("- Consider mood stabilizer adjustment\n\n")
            else:
                f.write("### ✓ Mood Stability\n\n")
                f.write("Current data suggests stable mood patterns.\n")
                f.write("Continue regular monitoring.\n\n")

        # Detailed metrics
        f.write("## Detailed Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Analysis Period | {len(result.daily_predictions)} days |\n")
        f.write(f"| Data Confidence | {result.confidence_score:.1%} |\n")
        f.write(f"| Records Processed | {result.records_processed:,} |\n")

        # Trend analysis
        if len(result.daily_predictions) >= 7:
            f.write("\n## 7-Day Trend Analysis\n\n")
            recent_dates = sorted(result.daily_predictions.keys())[-7:]

            f.write("| Date | Depression | Hypomanic | Manic |\n")
            f.write("|------|------------|-----------|-------|\n")

            for pred_date in recent_dates:
                preds = result.daily_predictions[pred_date]
                f.write(f"| {pred_date} | ")
                f.write(f"{preds.get('depression_risk', 0):.1%} | ")
                f.write(f"{preds.get('hypomanic_risk', 0):.1%} | ")
                f.write(f"{preds.get('manic_risk', 0):.1%} |\n")

        # Data quality
        if result.warnings:
            f.write("\n## Data Quality Warnings\n\n")
            for warning in result.warnings:
                f.write(f"- {warning}\n")

        f.write("\n---\n")
        f.write("*This report is for clinical decision support only. ")
        f.write("It should not replace professional medical judgment.*\n")


def main() -> None:
    """Main entry point for prediction CLI."""
    parser = argparse.ArgumentParser(
        description="Generate mood predictions from Apple Health data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  %(prog)s --input export.xml

  # Save predictions as JSON
  %(prog)s --input export.xml --output predictions.json --format json

  # Use ensemble models
  %(prog)s --input export.xml --ensemble

  # Use personalized model
  %(prog)s --input export.xml --user-id john_doe --model-dir models/

  # Generate clinical report
  %(prog)s --input export.xml --clinical-report report.md
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to Apple Health export (XML or JSON)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for predictions",
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )

    # Date filtering
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for analysis (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for analysis (YYYY-MM-DD)",
    )

    # Model options
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble of XGBoost + PAT models",
    )

    parser.add_argument(
        "--user-id",
        type=str,
        help="User ID for personalized model",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing model files",
    )

    # Report options
    parser.add_argument(
        "--clinical-report",
        type=str,
        help="Generate clinical report at specified path",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Parse dates
    start_date = None
    end_date = None

    if args.start_date:
        try:
            start_date = date.fromisoformat(args.start_date)
        except ValueError:
            logger.error(f"Invalid start date format: {args.start_date}")
            sys.exit(1)

    if args.end_date:
        try:
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            logger.error(f"Invalid end date format: {args.end_date}")
            sys.exit(1)

    try:
        # Configure pipeline
        config = PipelineConfig(
            model_dir=Path(args.model_dir) if args.model_dir else None,
            include_pat_sequences=args.ensemble,  # Enable ensemble when requested
        )

        # Create pipeline
        logger.info("Initializing mood prediction pipeline...")
        pipeline = MoodPredictionPipeline(config=config)

        # Load personalized model if specified
        if args.user_id:
            logger.info(f"Loading personalized model for user: {args.user_id}")
            try:
                _ = PersonalCalibrator.load(
                    user_id=args.user_id,
                    model_dir=Path(args.model_dir),
                )
                # TODO: Integrate personal calibrator into pipeline
                # TODO: Integrate calibrator with pipeline
                logger.info("Personalized model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load personalized model: {e}")
                logger.warning("Falling back to population model")

        # Process data
        logger.info(f"Processing health data from: {input_path}")
        result = pipeline.process_apple_health_file(
            file_path=input_path,
            start_date=start_date,
            end_date=end_date,
        )

        # Print summary
        print_summary(result)

        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.format == "json":
                save_json_output(result, output_path)
                logger.info(f"Predictions saved to: {output_path}")
            elif args.format == "csv":
                save_csv_output(result, output_path)
                logger.info(f"Predictions saved to: {output_path}")

        # Generate clinical report if requested
        if args.clinical_report:
            report_path = Path(args.clinical_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            generate_clinical_report(result, report_path)
            logger.info(f"Clinical report generated: {report_path}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
