"""Background worker runner."""

import logging
import sys

from big_mood_detector.infrastructure.background.task_queue import TaskQueue
from big_mood_detector.infrastructure.background.worker import TaskWorker
from big_mood_detector.infrastructure.settings.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the background worker."""
    logger.info("Starting background task worker...")

    # Ensure directories exist
    settings = get_settings()
    settings.ensure_directories()

    # Initialize task queue
    task_queue = TaskQueue()

    # Initialize worker
    worker = TaskWorker(task_queue)

    # Register tasks (if needed)
    # from big_mood_detector.infrastructure.background.tasks import register_health_processing_tasks
    # register_health_processing_tasks(worker)

    # Start processing
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Worker interrupted, shutting down...")
        worker.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
