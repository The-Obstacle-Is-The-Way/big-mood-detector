"""
Background Tasks

Specific task implementations for health data processing.
"""

import logging
from pathlib import Path
from typing import Any

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)

from .worker import TaskContext, TaskWorker

logger = logging.getLogger(__name__)


def process_health_file_task(payload: dict[str, Any], context: TaskContext) -> None:
    """Process a health data file.

    Args:
        payload: Task payload with file_path and upload_id
        context: Task context for progress updates
    """
    file_path = Path(payload["file_path"])
    payload.get("upload_id")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Update progress
        context.update_progress(0.1, "Starting health data processing")

        # Initialize pipeline
        pipeline = MoodPredictionPipeline()
        context.update_progress(0.2, "Pipeline initialized")

        # Process file
        context.update_progress(0.3, "Parsing health data...")
        result = pipeline.process_apple_health_file(file_path)

        # Update progress based on processing
        context.update_progress(0.7, "Generating predictions...")

        # Save results if output path provided
        output_path = payload.get("output_path")
        if output_path:
            pipeline.export_results(result, Path(output_path))
            context.update_progress(0.9, "Results exported")

        # Store results in payload for retrieval
        payload["result"] = {
            "depression_risk": float(
                result.overall_summary.get("avg_depression_risk", 0)
            ),
            "hypomanic_risk": float(
                result.overall_summary.get("avg_hypomanic_risk", 0)
            ),
            "manic_risk": float(result.overall_summary.get("avg_manic_risk", 0)),
            "confidence": float(result.confidence_score),
            "days_analyzed": int(result.overall_summary.get("days_analyzed", 0)),
            "records_processed": result.records_processed,
            "warnings": result.warnings,
        }

        context.update_progress(1.0, "Processing complete")
        logger.info(f"Successfully processed health file: {file_path}")

    except Exception as e:
        logger.error(f"Failed to process health file {file_path}: {e}")
        raise


def batch_process_files_task(payload: dict[str, Any], context: TaskContext) -> None:
    """Process multiple health data files.

    Args:
        payload: Task payload with file_paths list
        context: Task context for progress updates
    """
    file_paths = [Path(p) for p in payload["file_paths"]]
    results = []

    for i, file_path in enumerate(file_paths):
        progress = i / len(file_paths)
        context.update_progress(
            progress,
            f"Processing file {i + 1} of {len(file_paths)}: {file_path.name}",
        )

        try:
            # Process individual file
            file_payload = {"file_path": str(file_path)}
            process_health_file_task(file_payload, context)
            results.append(
                {
                    "file": str(file_path),
                    "status": "success",
                    "result": file_payload.get("result"),
                }
            )
        except Exception as e:
            results.append(
                {
                    "file": str(file_path),
                    "status": "failed",
                    "error": str(e),
                }
            )
            logger.error(f"Failed to process {file_path}: {e}")

    # Store batch results
    payload["results"] = results

    successful = sum(1 for r in results if r["status"] == "success")
    context.update_progress(
        1.0,
        f"Batch processing complete: {successful}/{len(file_paths)} files processed",
    )


def generate_clinical_report_task(
    payload: dict[str, Any], context: TaskContext
) -> None:
    """Generate a clinical report from processing results.

    Args:
        payload: Task payload with processing results
        context: Task context for progress updates
    """
    from big_mood_detector.interfaces.api.clinical_routes import (
        generate_clinical_report,
    )

    context.update_progress(0.1, "Starting report generation")

    # Extract results
    depression_risk = payload.get("depression_risk", 0)
    hypomanic_risk = payload.get("hypomanic_risk", 0)
    manic_risk = payload.get("manic_risk", 0)
    confidence = payload.get("confidence", 0)

    # Generate report
    context.update_progress(0.5, "Generating clinical interpretation")

    report = generate_clinical_report(
        depression_risk=depression_risk,
        hypomanic_risk=hypomanic_risk,
        manic_risk=manic_risk,
        confidence_score=confidence,
        days_analyzed=payload.get("days_analyzed", 0),
    )

    # Store report in payload
    payload["clinical_report"] = report

    # Save to file if path provided
    report_path = payload.get("report_path")
    if report_path:
        context.update_progress(0.8, "Saving report to file")
        Path(report_path).write_text(report)

    context.update_progress(1.0, "Report generation complete")


def register_health_processing_tasks(worker: TaskWorker) -> None:
    """Register all health processing tasks with the worker.

    Args:
        worker: Task worker to register handlers with
    """
    worker.register_handler("process_health_file", process_health_file_task)
    worker.register_handler("batch_process_files", batch_process_files_task)
    worker.register_handler("generate_clinical_report", generate_clinical_report_task)

    logger.info("Registered health processing tasks")
