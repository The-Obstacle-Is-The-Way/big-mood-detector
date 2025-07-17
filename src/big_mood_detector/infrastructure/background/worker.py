"""
Task Worker Implementation

Processes tasks from the queue with error handling and retries.
"""

import asyncio
import builtins
import logging
import signal
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any

from .task_queue import TaskQueue

logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Context provided to task handlers."""

    task_id: str
    queue: TaskQueue

    def update_progress(self, progress: float, message: str | None = None) -> None:
        """Update task progress.

        Args:
            progress: Progress value (0.0 to 1.0)
            message: Optional status message
        """
        self.queue.update_task_status(
            self.task_id,
            status="processing",
            progress=progress,
            message=message,
        )


TaskHandler = (
    Callable[[dict[str, Any]], Any]
    | Callable[[dict[str, Any], TaskContext], Any]
)

AsyncTaskHandler = (
    Callable[[dict[str, Any]], Coroutine[Any, Any, Any]]
    | Callable[[dict[str, Any], TaskContext], Coroutine[Any, Any, Any]]
)


class TaskWorker:
    """Synchronous task worker."""

    def __init__(
        self,
        queue: TaskQueue,
        max_retries: int = 3,
        task_timeout: int | None = None,
    ):
        """Initialize the task worker.

        Args:
            queue: Task queue to process
            max_retries: Maximum number of retries for failed tasks
            task_timeout: Task timeout in seconds (None for no timeout)
        """
        self.queue = queue
        self.max_retries = max_retries
        self.task_timeout = task_timeout
        self._handlers: dict[str, TaskHandler] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._running = False

    def register_handler(self, task_type: str, handler: TaskHandler) -> None:
        """Register a task handler.

        Args:
            task_type: Type of task to handle
            handler: Handler function
        """
        self._handlers[task_type] = handler

    def process_one(self) -> bool:
        """Process one task from the queue.

        Returns:
            True if a task was processed, False if queue was empty
        """
        task = self.queue.get_next_task()
        if not task:
            return False

        handler = self._handlers.get(task.task_type)
        if not handler:
            logger.error(f"No handler registered for task type: {task.task_type}")
            self.queue.update_task_status(
                task.id,
                status="failed",
                error=f"No handler for task type: {task.task_type}",
            )
            return True

        try:
            # Create task context
            context = TaskContext(task.id, self.queue)

            # Check if handler accepts context
            sig = handler.__code__.co_argcount
            if sig == 2:
                # Handler accepts context - narrow to specific type
                context_handler: Callable[[dict[str, Any], TaskContext], Any] = handler  # type: ignore[assignment]
                if self.task_timeout:
                    future = self._executor.submit(context_handler, task.payload, context)
                    future.result(timeout=self.task_timeout)
                else:
                    context_handler(task.payload, context)
            else:
                # Handler only accepts payload - narrow to specific type
                payload_handler: Callable[[dict[str, Any]], Any] = handler  # type: ignore[assignment]
                if self.task_timeout:
                    future = self._executor.submit(payload_handler, task.payload)
                    future.result(timeout=self.task_timeout)
                else:
                    payload_handler(task.payload)

            # Mark as completed
            self.queue.update_task_status(
                task.id,
                status="completed",
                progress=1.0,
            )

            logger.info(f"Task {task.id} completed successfully")
            return True

        except TimeoutError:
            error_msg = f"Task timed out after {self.task_timeout} seconds"
            logger.error(f"Task {task.id} failed: {error_msg}")
            self._handle_task_failure(task, error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.id} failed: {error_msg}", exc_info=True)
            self._handle_task_failure(task, error_msg)

        return True

    def _handle_task_failure(self, task: Any, error: str) -> None:
        """Handle task failure with retry logic.

        Args:
            task: Failed task
            error: Error message
        """
        if task.retry_count < self.max_retries:
            # Requeue for retry
            self.queue.requeue_task(task.id)
            logger.info(
                f"Task {task.id} requeued for retry "
                f"({task.retry_count + 1}/{self.max_retries})"
            )
        else:
            # Max retries exceeded
            self.queue.update_task_status(
                task.id,
                status="failed",
                error=error,
            )
            logger.error(
                f"Task {task.id} failed after {self.max_retries} retries: {error}"
            )

    def run(self, poll_interval: float = 1.0) -> None:
        """Run the worker in a loop.

        Args:
            poll_interval: Seconds to wait between polls when queue is empty
        """
        self._running = True
        logger.info("Task worker started")

        # Set up signal handlers
        def stop_handler(signum, frame):
            logger.info("Received stop signal, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

        while self._running:
            try:
                if not self.process_one():
                    # Queue was empty, wait before polling again
                    time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(poll_interval)

        logger.info("Task worker stopped")

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        self._executor.shutdown(wait=True)


class AsyncTaskWorker:
    """Asynchronous task worker."""

    def __init__(
        self,
        queue: TaskQueue,
        max_retries: int = 3,
        task_timeout: int | None = None,
    ):
        """Initialize the async task worker.

        Args:
            queue: Task queue to process
            max_retries: Maximum number of retries for failed tasks
            task_timeout: Task timeout in seconds (None for no timeout)
        """
        self.queue = queue
        self.max_retries = max_retries
        self.task_timeout = task_timeout
        self._handlers: dict[str, AsyncTaskHandler] = {}
        self._running = False

    def register_handler(self, task_type: str, handler: AsyncTaskHandler) -> None:
        """Register an async task handler.

        Args:
            task_type: Type of task to handle
            handler: Async handler function
        """
        self._handlers[task_type] = handler

    async def process_one(self) -> bool:
        """Process one task from the queue.

        Returns:
            True if a task was processed, False if queue was empty
        """
        task = self.queue.get_next_task()
        if not task:
            return False

        handler = self._handlers.get(task.task_type)
        if not handler:
            logger.error(f"No handler registered for task type: {task.task_type}")
            self.queue.update_task_status(
                task.id,
                status="failed",
                error=f"No handler for task type: {task.task_type}",
            )
            return True

        try:
            # Create task context
            context = TaskContext(task.id, self.queue)

            # Check if handler accepts context
            if asyncio.iscoroutinefunction(handler):
                sig = handler.__code__.co_argcount
                if sig == 2:
                    # Handler accepts context
                    if self.task_timeout:
                        await asyncio.wait_for(
                            handler(task.payload, context),
                            timeout=self.task_timeout,
                        )
                    else:
                        await handler(task.payload, context)
                else:
                    # Handler only accepts payload
                    if self.task_timeout:
                        await asyncio.wait_for(
                            handler(task.payload),
                            timeout=self.task_timeout,
                        )
                    else:
                        await handler(task.payload)
            else:
                # Sync handler, run in executor
                loop = asyncio.get_event_loop()
                if self.task_timeout:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, handler, task.payload),
                        timeout=self.task_timeout,
                    )
                else:
                    await loop.run_in_executor(None, handler, task.payload)

            # Mark as completed
            self.queue.update_task_status(
                task.id,
                status="completed",
                progress=1.0,
            )

            logger.info(f"Task {task.id} completed successfully")
            return True

        except builtins.TimeoutError:
            error_msg = f"Task timed out after {self.task_timeout} seconds"
            logger.error(f"Task {task.id} failed: {error_msg}")
            self._handle_task_failure(task, error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.id} failed: {error_msg}", exc_info=True)
            self._handle_task_failure(task, error_msg)

        return True

    def _handle_task_failure(self, task: Any, error: str) -> None:
        """Handle task failure with retry logic.

        Args:
            task: Failed task
            error: Error message
        """
        if task.retry_count < self.max_retries:
            # Requeue for retry
            self.queue.requeue_task(task.id)
            logger.info(
                f"Task {task.id} requeued for retry "
                f"({task.retry_count + 1}/{self.max_retries})"
            )
        else:
            # Max retries exceeded
            self.queue.update_task_status(
                task.id,
                status="failed",
                error=error,
            )
            logger.error(
                f"Task {task.id} failed after {self.max_retries} retries: {error}"
            )

    async def run(self, poll_interval: float = 1.0) -> None:
        """Run the async worker in a loop.

        Args:
            poll_interval: Seconds to wait between polls when queue is empty
        """
        self._running = True
        logger.info("Async task worker started")

        while self._running:
            try:
                if not await self.process_one():
                    # Queue was empty, wait before polling again
                    await asyncio.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)

        logger.info("Async task worker stopped")

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
