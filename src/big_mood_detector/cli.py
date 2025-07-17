"""
Big Mood Detector CLI

Command-line interface for processing health data and running the API server.
"""

import sys
from pathlib import Path

import click

try:
    import uvicorn
except ImportError:
    uvicorn = None  # type: ignore[assignment]

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineResult,
)


@click.group()
def main() -> None:
    """Big Mood Detector CLI - Process health data for mood predictions."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default="output/mood_predictions.csv",
    help="Output CSV file path",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Generate clinical report",
)
def process(input_path: str, output: str, report: bool) -> None:
    """Process health data and generate mood predictions.

    INPUT_PATH can be either:
    - A directory containing Health Auto Export JSON files
    - An Apple Health export.xml file
    """
    try:
        click.echo(f"Processing health data from: {input_path}")

        # Initialize pipeline
        pipeline = MoodPredictionPipeline()

        # Process data
        result = pipeline.process_apple_health_file(file_path=Path(input_path))

        # Save results
        output_path = Path(output)
        pipeline.export_results(result, output_path)
        click.echo(f"Results saved to: {output_path}")

        # Print summary
        if result.overall_summary:
            click.echo("\n📊 Analysis Complete!")
            click.echo(
                f"Depression Risk: {result.overall_summary.get('avg_depression_risk', 0):.1%}"
            )
            click.echo(
                f"Hypomanic Risk: {result.overall_summary.get('avg_hypomanic_risk', 0):.1%}"
            )
            click.echo(
                f"Manic Risk: {result.overall_summary.get('avg_manic_risk', 0):.1%}"
            )
            click.echo(
                f"\nDays analyzed: {result.overall_summary.get('days_analyzed', 0)}"
            )
            click.echo(f"Confidence: {result.confidence_score:.1%}")

        # Generate clinical report if requested
        if report:
            report_path = output_path.with_suffix(".txt")
            _generate_clinical_report(result, report_path)
            click.echo(f"\nClinical report generated: {report_path}")

    except Exception as e:
        click.echo(f"Error processing health data: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(host: str, port: int) -> None:
    """Start the API server."""
    if uvicorn is None:
        click.echo(
            "Error: uvicorn not installed. Install with: pip install uvicorn", err=True
        )
        sys.exit(1)

    click.echo(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "big_mood_detector.interfaces.api.main:app",
        host=host,
        port=port,
        reload=True,
    )


@main.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "--poll-interval",
    default=60,
    help="Polling interval in seconds",
)
@click.option("--patterns", "-p", multiple=True, default=["*.xml", "*.json"], help="File patterns to watch")
@click.option("--recursive/--no-recursive", default=True, help="Watch subdirectories")
@click.option("--state-file", type=click.Path(), help="File to persist watcher state")
def watch(
    directory: str, 
    poll_interval: int,
    patterns: tuple[str, ...],
    recursive: bool,
    state_file: str | None,
) -> None:
    """Watch directory for new health data files.

    Monitors DIRECTORY for new JSON or XML files and automatically
    processes them when detected.
    """
    from big_mood_detector.infrastructure.background.task_queue import TaskQueue
    from big_mood_detector.infrastructure.background.tasks import (
        register_health_processing_tasks,
    )
    from big_mood_detector.infrastructure.background.worker import TaskWorker
    from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher
    
    watch_path = Path(directory)
    click.echo(f"Watching {watch_path} for health data files...")
    click.echo(f"Patterns: {', '.join(patterns)}")
    click.echo(f"Recursive: {recursive}")
    click.echo(f"Poll interval: {poll_interval} seconds")
    
    # Create task queue and worker
    task_queue = TaskQueue()
    task_worker = TaskWorker(task_queue)
    register_health_processing_tasks(task_worker)
    
    # Create file watcher
    watcher = FileWatcher(
        watch_path=watch_path,
        patterns=list(patterns),
        recursive=recursive,
        poll_interval=poll_interval,
        state_file=Path(state_file) if state_file else None,
        ignore_patterns=[".*", "*~", "*.tmp", "*.bak"],
    )
    
    # Track processed files
    processed_count = 0
    
    def process_new_file(file_path: Path):
        nonlocal processed_count
        
        click.echo(f"\n🔍 Found new file: {file_path}")
        
        # Add to task queue
        task_id = task_queue.add_task(
            task_type="process_health_file",
            payload={
                "file_path": str(file_path),
                "output_path": str(
                    file_path.parent / f"{file_path.stem}_predictions.csv"
                ),
            },
        )
        
        # Process immediately
        click.echo("Processing...")
        task_worker.process_one()
        
        # Check result
        status = task_queue.get_task_status(task_id)
        if status["status"] == "completed":
            processed_count += 1
            click.secho("✓ Processing complete!", fg="green")
            
            # Get task result
            task = next((t for t in task_queue._tasks.values() if t.id == task_id), None)
            if task and "result" in task.payload:
                result = task.payload["result"]
                click.echo(f"  Depression Risk: {result['depression_risk']:.1%}")
                click.echo(f"  Hypomanic Risk: {result['hypomanic_risk']:.1%}")
                click.echo(f"  Manic Risk: {result['manic_risk']:.1%}")
        else:
            click.secho(f"✗ Processing failed: {status.get('error', 'Unknown error')}", fg="red")
    
    # Register handlers
    watcher.on_created(process_new_file)
    
    # Start watching
    click.echo(f"\nWatching for new files (press Ctrl+C to stop)...")
    click.echo(f"Processed files: {processed_count}")
    
    try:
        watcher.watch()
    except KeyboardInterrupt:
        click.echo(f"\n\nStopping file watcher...")
        click.echo(f"Total files processed: {processed_count}")
        watcher.save_state()


def _generate_clinical_report(result: PipelineResult, output_path: Path) -> None:
    """Generate a clinical report from pipeline results.

    Args:
        result: Pipeline result containing predictions
        output_path: Path to save the report
    """
    with open(output_path, "w") as f:
        f.write("CLINICAL DECISION SUPPORT (CDS) REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("PATIENT DATA SUMMARY\n")
        f.write(f"Analysis Period: {len(result.daily_predictions)} days\n")
        f.write(f"Total Records Processed: {result.records_processed}\n")
        f.write(f"Data Quality Score: {result.confidence_score:.1%}\n")

        f.write("\nCLINICAL RISK ASSESSMENT\n")
        f.write("-" * 30 + "\n")

        if result.overall_summary:
            dep_risk = result.overall_summary.get("avg_depression_risk", 0)
            hypo_risk = result.overall_summary.get("avg_hypomanic_risk", 0)
            manic_risk = result.overall_summary.get("avg_manic_risk", 0)

            f.write(f"Depression Risk: {dep_risk:.1%}\n")
            f.write(f"Hypomanic Risk: {hypo_risk:.1%}\n")
            f.write(f"Manic Risk: {manic_risk:.1%}\n")

            f.write("\nCLINICAL RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")

            # Clinical decision logic
            if dep_risk > 0.7:
                f.write("⚠️  HIGH DEPRESSION RISK DETECTED\n")
                f.write("• Consider immediate clinical evaluation\n")
                f.write("• Review sleep patterns and activity levels\n")
                f.write("• Monitor for suicidal ideation\n")
            elif dep_risk > 0.4:
                f.write("⚠️  MODERATE DEPRESSION RISK\n")
                f.write("• Schedule follow-up within 2 weeks\n")
                f.write("• Assess sleep hygiene and daily routines\n")
                f.write("• Consider therapy referral\n")
            else:
                f.write("✓ Low depression risk\n")
                f.write("• Continue regular monitoring\n")

            if hypo_risk > 0.5 or manic_risk > 0.3:
                f.write("\n⚠️  ELEVATED MOOD EPISODE RISK\n")
                f.write("• Monitor for decreased sleep need\n")
                f.write("• Track activity levels and goal-directed behavior\n")
                f.write("• Review medication compliance\n")

        if result.warnings:
            f.write("\nDATA QUALITY WARNINGS\n")
            f.write("-" * 30 + "\n")
            for warning in result.warnings:
                f.write(f"• {warning}\n")

        f.write("\nDETAILED DAILY ANALYSIS\n")
        f.write("-" * 30 + "\n")

        # Show daily predictions
        for date, pred in list(result.daily_predictions.items())[:7]:  # First week
            f.write(f"\n{date}:\n")
            f.write(f"  Depression: {pred['depression_risk']:.1%}")

            # Add risk level
            if pred["depression_risk"] > 0.7:
                f.write(" [HIGH RISK]")
            elif pred["depression_risk"] > 0.4:
                f.write(" [MODERATE]")
            else:
                f.write(" [LOW]")

            f.write(f"\n  Hypomania: {pred['hypomanic_risk']:.1%}")
            f.write(f"\n  Mania: {pred['manic_risk']:.1%}")
            f.write(f"\n  Confidence: {pred['confidence']:.1%}\n")


if __name__ == "__main__":
    main()
