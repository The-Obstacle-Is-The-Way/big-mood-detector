"""
Watch Command

File monitoring and automatic processing for the Big Mood Detector.
"""

from pathlib import Path

import click

from big_mood_detector.infrastructure.background.task_queue import TaskQueue
from big_mood_detector.infrastructure.background.tasks import (
    register_health_processing_tasks,
)
from big_mood_detector.infrastructure.background.worker import TaskWorker
from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher


@click.command(name="watch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--poll-interval",
    default=60,
    type=int,
    help="Polling interval in seconds",
)
@click.option(
    "--patterns",
    "-p",
    multiple=True,
    default=["*.xml", "*.json"],
    help="File patterns to watch",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Watch subdirectories",
)
@click.option(
    "--state-file",
    type=click.Path(),
    help="File to persist watcher state",
)
@click.option(
    "--auto-process/--no-auto-process",
    default=True,
    help="Automatically process new files",
)
def watch_command(
    directory: str,
    poll_interval: int,
    patterns: tuple[str, ...],
    recursive: bool,
    state_file: str | None,
    auto_process: bool,
) -> None:
    """Watch directory for new health data files.

    Monitors DIRECTORY for new JSON or XML files and optionally
    processes them automatically when detected.
    """
    watch_path = Path(directory)
    click.echo(f"👁️  Watching {watch_path} for health data files...")
    click.echo(f"Patterns: {', '.join(patterns)}")
    click.echo(f"Recursive: {recursive}")
    click.echo(f"Poll interval: {poll_interval} seconds")
    click.echo(f"Auto-process: {auto_process}")

    # Create task queue and worker if auto-processing
    task_queue = None
    task_worker = None
    if auto_process:
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
        ignore_patterns=[".*", "*~", "*.tmp", "*.bak", "*.swp"],
    )

    # Track statistics
    stats = {"detected": 0, "processed": 0, "failed": 0}

    def handle_new_file(file_path: Path) -> None:
        """Handle detection of a new file."""
        stats["detected"] += 1
        click.echo(f"\n🔍 Found new file: {file_path}")
        click.echo(f"   Size: {file_path.stat().st_size / 1024:.1f} KB")

        if not auto_process:
            click.echo("   Status: Detected (auto-process disabled)")
            return

        if task_queue and task_worker:
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
            click.echo("   Processing...")
            task_worker.process_one()

            # Check result
            status = task_queue.get_task_status(task_id)
            if status["status"] == "completed":
                stats["processed"] += 1
                click.secho("   ✓ Processing complete!", fg="green")

                # Get task result
                task = next(
                    (t for t in task_queue._tasks.values() if t.id == task_id), None
                )
                if task and "result" in task.payload:
                    result = task.payload["result"]
                    click.echo(f"   Depression Risk: {result['depression_risk']:.1%}")
                    click.echo(f"   Hypomanic Risk: {result['hypomanic_risk']:.1%}")
                    click.echo(f"   Manic Risk: {result['manic_risk']:.1%}")
            else:
                stats["failed"] += 1
                error_msg = status.get("error", "Unknown error")
                click.secho(f"   ✗ Processing failed: {error_msg}", fg="red")

    def handle_file_modified(file_path: Path) -> None:
        """Handle file modification."""
        click.echo(f"\n📝 File modified: {file_path}")

    def handle_file_deleted(file_path: Path) -> None:
        """Handle file deletion."""
        click.echo(f"\n🗑️  File deleted: {file_path}")

    # Register handlers
    watcher.on_created(handle_new_file)
    watcher.on_modified(handle_file_modified)
    watcher.on_deleted(handle_file_deleted)

    # Status display
    click.echo("\n" + "=" * 50)
    click.echo("Watching for new files (press Ctrl+C to stop)...")
    click.echo(f"Files: {stats['detected']} detected")
    if auto_process:
        click.echo(f"       {stats['processed']} processed, {stats['failed']} failed")
    click.echo("=" * 50 + "\n")

    try:
        watcher.watch()
    except KeyboardInterrupt:
        click.echo("\n\n⏹️  Stopping file watcher...")
        click.echo("\nFinal Statistics:")
        click.echo(f"  Files detected: {stats['detected']}")
        if auto_process:
            click.echo(f"  Files processed: {stats['processed']}")
            click.echo(f"  Processing failures: {stats['failed']}")

        if state_file:
            watcher.save_state()
            click.echo(f"\n💾 State saved to: {state_file}")
