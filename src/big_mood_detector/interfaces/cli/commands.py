"""
Unified CLI Commands

All CLI commands consolidated into the interfaces layer following Clean Architecture.
This module contains all command implementations for the Big Mood Detector.
"""

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import click

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
    PipelineResult,
)
from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
    PersonalCalibrator,
)


def format_risk_level(risk_score: float) -> str:
    """Format risk score with clinical level."""
    if risk_score >= 0.7:
        return f"{risk_score:.1%} [HIGH]"
    elif risk_score >= 0.4:
        return f"{risk_score:.1%} [MODERATE]"
    else:
        return f"{risk_score:.1%} [LOW]"


def print_summary(result: PipelineResult, verbose: bool = False) -> None:
    """Print analysis summary to console."""
    if result.overall_summary:
        click.echo("\n📊 Analysis Complete!")
        click.echo(
            f"Depression Risk: {format_risk_level(result.overall_summary.get('avg_depression_risk', 0))}"
        )
        click.echo(
            f"Hypomanic Risk: {format_risk_level(result.overall_summary.get('avg_hypomanic_risk', 0))}"
        )
        click.echo(
            f"Manic Risk: {format_risk_level(result.overall_summary.get('avg_manic_risk', 0))}"
        )
        click.echo(f"\nDays analyzed: {result.overall_summary.get('days_analyzed', 0)}")
        click.echo(f"Confidence: {result.confidence_score:.1%}")

        if verbose and result.metadata:
            click.echo("\nMetadata:")
            for key, value in result.metadata.items():
                click.echo(f"  {key}: {value}")

    if result.warnings and verbose:
        click.echo("\n⚠️  Warnings:")
        for warning in result.warnings:
            click.echo(f"  • {warning}")


def save_json_output(result: PipelineResult, output_path: Path) -> None:
    """Save results in JSON format."""
    import json

    output_data = {
        "summary": result.overall_summary,
        "confidence": result.confidence_score,
        "daily_predictions": {
            str(date): pred for date, pred in result.daily_predictions.items()
        },
        "metadata": {
            "records_processed": result.records_processed,
            "processing_time": result.processing_time_seconds,
            "warnings": result.warnings,
            "errors": result.errors,
        },
    }

    if result.metadata:
        output_data["metadata"].update(result.metadata)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)


def save_csv_output(result: PipelineResult, output_path: Path) -> None:
    """Save results in CSV format."""
    import pandas as pd

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


def generate_clinical_report(result: PipelineResult, output_path: Path) -> None:
    """Generate a detailed clinical report."""
    with open(output_path, "w") as f:
        f.write("CLINICAL DECISION SUPPORT (CDS) REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("PATIENT DATA SUMMARY\n")
        f.write(f"Analysis Period: {len(result.daily_predictions)} days\n")
        f.write(f"Total Records Processed: {result.records_processed}\n")
        f.write(f"Data Quality Score: {result.confidence_score:.1%}\n")

        if result.metadata.get("personal_calibration_used"):
            f.write(f"\nPersonalized Model: Active (User: {result.metadata.get('user_id')})\n")

        f.write("\nCLINICAL RISK ASSESSMENT\n")
        f.write("-" * 30 + "\n")

        if result.overall_summary:
            dep_risk = result.overall_summary.get("avg_depression_risk", 0)
            hypo_risk = result.overall_summary.get("avg_hypomanic_risk", 0)
            manic_risk = result.overall_summary.get("avg_manic_risk", 0)

            f.write(f"Depression Risk: {format_risk_level(dep_risk)}\n")
            f.write(f"Hypomanic Risk: {format_risk_level(hypo_risk)}\n")
            f.write(f"Manic Risk: {format_risk_level(manic_risk)}\n")

            f.write("\nCLINICAL RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")

            # Clinical decision logic based on DSM-5 criteria
            if dep_risk > 0.7:
                f.write("⚠️  HIGH DEPRESSION RISK DETECTED\n")
                f.write("• Consider immediate clinical evaluation\n")
                f.write("• Review sleep patterns and activity levels\n")
                f.write("• Monitor for suicidal ideation\n")
                f.write("• Assess functional impairment\n")
            elif dep_risk > 0.4:
                f.write("⚠️  MODERATE DEPRESSION RISK\n")
                f.write("• Schedule follow-up within 2 weeks\n")
                f.write("• Assess sleep hygiene and daily routines\n")
                f.write("• Consider therapy referral\n")
                f.write("• Monitor symptom progression\n")
            else:
                f.write("✓ Low depression risk\n")
                f.write("• Continue regular monitoring\n")
                f.write("• Maintain healthy sleep schedule\n")

            if hypo_risk > 0.5 or manic_risk > 0.3:
                f.write("\n⚠️  ELEVATED MOOD EPISODE RISK\n")
                f.write("• Monitor for decreased sleep need\n")
                f.write("• Track activity levels and goal-directed behavior\n")
                f.write("• Review medication compliance\n")
                f.write("• Assess for impulsive behaviors\n")

        if result.warnings:
            f.write("\nDATA QUALITY WARNINGS\n")
            f.write("-" * 30 + "\n")
            for warning in result.warnings:
                f.write(f"• {warning}\n")

        f.write("\nDETAILED DAILY ANALYSIS\n")
        f.write("-" * 30 + "\n")

        # Show first week of daily predictions
        for date, pred in list(result.daily_predictions.items())[:7]:
            f.write(f"\n{date}:\n")
            f.write(f"  Depression: {format_risk_level(pred['depression_risk'])}\n")
            f.write(f"  Hypomania: {format_risk_level(pred['hypomanic_risk'])}\n")
            f.write(f"  Mania: {format_risk_level(pred['manic_risk'])}\n")
            f.write(f"  Confidence: {pred['confidence']:.1%}\n")

            # Add model info if ensemble was used
            if "models_used" in pred:
                f.write(f"  Models: {', '.join(pred['models_used'])}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("Generated by Big Mood Detector\n")
        f.write("This report is for clinical decision support only.\n")
        f.write("Not a substitute for professional diagnosis.\n")


@click.command(name="process")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (format based on extension)",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for analysis (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for analysis (YYYY-MM-DD)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output"
)
def process_command(
    input_path: str,
    output: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    verbose: bool,
) -> None:
    """Process health data to extract features for mood prediction."""
    try:
        click.echo(f"Processing health data from: {input_path}")

        # Initialize pipeline
        pipeline = MoodPredictionPipeline()

        # Process health export
        output_path = Path(output) if output else Path("output/features.csv")
        df = pipeline.process_health_export(
            export_path=Path(input_path),
            output_path=output_path,
            start_date=start_date.date() if start_date else None,
            end_date=end_date.date() if end_date else None,
        )

        click.echo(f"\n✅ Features extracted: {len(df)} days")
        click.echo(f"Output saved to: {output_path}")

        if verbose and not df.empty:
            click.echo("\nFeature Summary:")
            click.echo(f"  Columns: {len(df.columns)}")
            click.echo(f"  Date range: {df.index.min()} to {df.index.max()}")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command(name="predict")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (.json, .csv, or .txt for report)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "csv", "summary"]),
    default="summary",
    help="Output format",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for analysis (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for analysis (YYYY-MM-DD)",
)
@click.option(
    "--ensemble/--no-ensemble",
    default=False,
    help="Use ensemble model (PAT + XGBoost)",
)
@click.option(
    "--user-id",
    help="User ID for personalized predictions",
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True),
    help="Directory containing model weights",
)
@click.option(
    "--report/--no-report",
    default=False,
    help="Generate clinical report",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output"
)
def predict_command(
    input_path: str,
    output: Optional[str],
    format: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    ensemble: bool,
    user_id: Optional[str],
    model_dir: Optional[str],
    report: bool,
    verbose: bool,
) -> None:
    """Generate mood predictions from health data."""
    try:
        click.echo(f"Processing health data from: {input_path}")

        # Configure pipeline
        config = PipelineConfig(
            include_pat_sequences=ensemble,
            model_dir=Path(model_dir) if model_dir else None,
            enable_personal_calibration=bool(user_id),
            user_id=user_id,
        )

        # Initialize pipeline
        pipeline = MoodPredictionPipeline(config=config)

        # Process data
        result = pipeline.process_apple_health_file(
            file_path=Path(input_path),
            start_date=start_date.date() if start_date else None,
            end_date=end_date.date() if end_date else None,
        )

        # Handle output based on format
        if format == "summary":
            print_summary(result, verbose=verbose)
        elif output:
            output_path = Path(output)
            if format == "json" or output_path.suffix == ".json":
                save_json_output(result, output_path)
                click.echo(f"✅ JSON output saved to: {output_path}")
            elif format == "csv" or output_path.suffix == ".csv":
                save_csv_output(result, output_path)
                click.echo(f"✅ CSV output saved to: {output_path}")

        # Generate clinical report if requested
        if report:
            report_path = Path(output).with_suffix(".txt") if output else Path("clinical_report.txt")
            generate_clinical_report(result, report_path)
            click.echo(f"✅ Clinical report saved to: {report_path}")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)