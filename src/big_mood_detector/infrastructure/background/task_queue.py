"""
Task Queue Implementation

Simple in-memory task queue for background processing.
In production, this would be replaced with Redis, RabbitMQ, or similar.
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Task:
    """Represents a background task."""

    id: str
    task_type: str
    payload: dict[str, Any]
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    progress: float = 0.0
    message: str | None = None
    retry_count: int = 0


class TaskQueue:
    """In-memory task queue with thread safety."""

    def __init__(self) -> None:
        """Initialize the task queue."""
        self._tasks: dict[str, Task] = {}
        self._pending_queue: list[str] = []
        self._lock = threading.Lock()

    def add_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        task_id: str | None = None,
    ) -> str:
        """Add a task to the queue.

        Args:
            task_type: Type of task to execute
            payload: Task payload data
            task_id: Optional task ID (generated if not provided)

        Returns:
            Task ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = Task(
            id=task_id,
            task_type=task_type,
            payload=payload,
        )

        with self._lock:
            self._tasks[task_id] = task
            self._pending_queue.append(task_id)

        return task_id

    def get_next_task(self) -> Task | None:
        """Get the next pending task from the queue.

        Returns:
            Next task or None if queue is empty
        """
        with self._lock:
            while self._pending_queue:
                task_id = self._pending_queue.pop(0)
                task = self._tasks.get(task_id)

                if task and task.status == "pending":
                    task.status = "processing"
                    task.started_at = datetime.now()
                    return task

        return None

    def update_task_status(
        self,
        task_id: str,
        status: str,
        error: str | None = None,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        """Update task status.

        Args:
            task_id: Task ID
            status: New status
            error: Error message if failed
            progress: Task progress (0.0 to 1.0)
            message: Status message
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status

                if error is not None:
                    task.error = error

                if progress is not None:
                    task.progress = progress

                if message is not None:
                    task.message = message

                if status in ["completed", "failed"]:
                    task.completed_at = datetime.now()

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get task status information.

        Args:
            task_id: Task ID

        Returns:
            Task status dictionary
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {"status": "not_found"}

            return {
                "id": task.id,
                "task_type": task.task_type,
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
                "error": task.error,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": (
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                "retry_count": task.retry_count,
            }

    def get_pending_tasks(self) -> list[str]:
        """Get list of pending task IDs.

        Returns:
            List of pending task IDs
        """
        with self._lock:
            return [
                task_id
                for task_id in self._pending_queue
                if self._tasks.get(task_id, Task("", "", {})).status == "pending"
            ]

    def requeue_task(self, task_id: str) -> bool:
        """Requeue a task for retry.

        Args:
            task_id: Task ID

        Returns:
            True if task was requeued
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "pending"
                task.retry_count += 1
                task.started_at = None
                task.error = None
                self._pending_queue.append(task_id)
                return True
        return False

    def clear(self) -> None:
        """Clear all tasks from the queue."""
        with self._lock:
            self._tasks.clear()
            self._pending_queue.clear()

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics.

        Returns:
            Dictionary with queue stats
        """
        with self._lock:
            stats = {
                "total": len(self._tasks),
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
            }

            for task in self._tasks.values():
                if task.status in stats:
                    stats[task.status] += 1

            return stats
