#!/usr/bin/env python3
"""
Command-line interface for processing Apple Health data.

This is the main entry point for users to generate the 36 features
required for mood prediction.

Usage:
    python -m big_mood_detector.cli.process_data \
        --input /path/to/health/export \
        --output features.csv \
        --start-date 2024-01-01 \
        --end-date 2024-12-31
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from big_mood_detector.application.mood_prediction_pipeline import (
    MoodPredictionPipeline,
)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Process Apple Health data for mood prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process XML export
  %(prog)s --input export.xml --output features.csv

  # Process JSON directory with date range
  %(prog)s --input health_data/ --output features.csv --start-date 2024-01-01

  # Verbose mode for debugging
  %(prog)s --input export.xml --output features.csv --verbose
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to Apple Health export (XML file or JSON directory)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output CSV file path for 36 features",
    )

    parser.add_argument(
        "--start-date", type=str, help="Start date for processing (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=str, help="End date for processing (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output against expected format",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input path
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

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process data
    logger.info(f"Processing health data from: {input_path}")
    logger.info(f"Output will be saved to: {output_path}")

    if start_date:
        logger.info(f"Start date: {start_date}")
    if end_date:
        logger.info(f"End date: {end_date}")

    try:
        # Create pipeline and process
        pipeline = MoodPredictionPipeline()
        df = pipeline.process_health_export(
            input_path, output_path, start_date, end_date
        )

        logger.info(f"Successfully processed {len(df)} days of data")

        # Validate output if requested
        if args.validate:
            validate_output(df)
            logger.info("Output validation passed")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


def validate_output(df):
    """
    Validate that output has correct format.

    Checks:
    - 36 columns with correct names
    - No missing values
    - Reasonable value ranges
    """
    expected_columns = [
        # Sleep features (10 × 3)
        "sleep_percentage_MN",
        "sleep_percentage_SD",
        "sleep_percentage_Z",
        "sleep_amplitude_MN",
        "sleep_amplitude_SD",
        "sleep_amplitude_Z",
        "long_num_MN",
        "long_num_SD",
        "long_num_Z",
        "long_len_MN",
        "long_len_SD",
        "long_len_Z",
        "long_ST_MN",
        "long_ST_SD",
        "long_ST_Z",
        "long_WT_MN",
        "long_WT_SD",
        "long_WT_Z",
        "short_num_MN",
        "short_num_SD",
        "short_num_Z",
        "short_len_MN",
        "short_len_SD",
        "short_len_Z",
        "short_ST_MN",
        "short_ST_SD",
        "short_ST_Z",
        "short_WT_MN",
        "short_WT_SD",
        "short_WT_Z",
        # Circadian features (2 × 3)
        "circadian_amplitude_MN",
        "circadian_amplitude_SD",
        "circadian_amplitude_Z",
        "circadian_phase_MN",
        "circadian_phase_SD",
        "circadian_phase_Z",
    ]

    # Check columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        logging.warning(f"Extra columns found: {extra_cols}")

    # Check for missing values
    missing_values = df[expected_columns].isnull().sum().sum()
    if missing_values > 0:
        logging.warning(f"Found {missing_values} missing values")

    # Check value ranges
    # Sleep percentage should be 0-1
    if not (0 <= df["sleep_percentage_MN"].min() <= 1):
        logging.warning("Sleep percentage values outside expected range")

    # Z-scores should be roughly -3 to 3
    z_cols = [c for c in expected_columns if c.endswith("_Z")]
    for col in z_cols:
        if df[col].abs().max() > 10:
            logging.warning(f"{col} has extreme z-scores")

    # DLMO (circadian phase) should be 0-24
    if not (0 <= df["circadian_phase_MN"].min() < 24):
        logging.warning("Circadian phase values outside 0-24 range")

    logging.info(
        f"Validation complete: {len(df)} rows, {len(expected_columns)} features"
    )


if __name__ == "__main__":
    main()
