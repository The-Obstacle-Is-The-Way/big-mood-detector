"""
Test Background Task Processing

TDD for enhanced background task handling.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


class TestBackgroundTasks:
    """Test background task processing."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from big_mood_detector.interfaces.api.main import app

        return TestClient(app)

    def test_background_task_queue(self):
        """Test that background tasks can be queued."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue

        queue = TaskQueue()
        assert queue is not None
        assert hasattr(queue, "add_task")
        assert hasattr(queue, "get_pending_tasks")

    def test_add_task_to_queue(self):
        """Test adding tasks to the queue."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue

        queue = TaskQueue()
        task_id = queue.add_task(
            task_type="process_health_file",
            payload={"file_path": "/tmp/test.json"},
        )

        assert task_id is not None
        assert len(queue.get_pending_tasks()) == 1

    def test_task_status_tracking(self):
        """Test tracking task status."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue

        queue = TaskQueue()
        task_id = queue.add_task(
            task_type="process_health_file",
            payload={"file_path": "/tmp/test.json"},
        )

        # Check initial status
        status = queue.get_task_status(task_id)
        assert status["status"] == "pending"

        # Update status
        queue.update_task_status(task_id, "processing", progress=0.5)
        status = queue.get_task_status(task_id)
        assert status["status"] == "processing"
        assert status["progress"] == 0.5

    def test_task_worker(self):
        """Test the task worker processes tasks."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.worker import TaskWorker

        queue = TaskQueue()
        worker = TaskWorker(queue)

        # Add a test task
        task_id = queue.add_task(
            task_type="test_task",
            payload={"value": 42},
        )

        # Mock task handler
        handler_called = False

        def test_handler(payload):
            nonlocal handler_called
            handler_called = True
            assert payload["value"] == 42

        worker.register_handler("test_task", test_handler)

        # Process one task
        worker.process_one()

        assert handler_called
        status = queue.get_task_status(task_id)
        assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_async_task_worker(self):
        """Test async task worker."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.worker import AsyncTaskWorker

        queue = TaskQueue()
        worker = AsyncTaskWorker(queue)

        # Add a test task
        task_id = queue.add_task(
            task_type="async_test_task",
            payload={"value": 42},
        )

        # Mock async task handler
        handler_called = False

        async def test_handler(payload):
            nonlocal handler_called
            handler_called = True
            assert payload["value"] == 42
            await asyncio.sleep(0.1)  # Simulate async work

        worker.register_handler("async_test_task", test_handler)

        # Process one task
        await worker.process_one()

        assert handler_called
        status = queue.get_task_status(task_id)
        assert status["status"] == "completed"

    def test_task_retry_on_failure(self):
        """Test task retry mechanism."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.worker import TaskWorker

        queue = TaskQueue()
        worker = TaskWorker(queue, max_retries=3)

        # Add a test task
        task_id = queue.add_task(
            task_type="failing_task",
            payload={"fail_times": 2},
        )

        # Mock task handler that fails initially
        attempt_count = 0

        def failing_handler(payload):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= payload["fail_times"]:
                raise Exception(f"Intentional failure {attempt_count}")
            return "success"

        worker.register_handler("failing_task", failing_handler)

        # Process task multiple times
        for _ in range(3):
            worker.process_one()

        assert attempt_count == 3
        status = queue.get_task_status(task_id)
        assert status["status"] == "completed"

    def test_task_failure_after_max_retries(self):
        """Test task fails after max retries."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.worker import TaskWorker

        queue = TaskQueue()
        worker = TaskWorker(queue, max_retries=2)

        # Add a test task
        task_id = queue.add_task(
            task_type="always_failing_task",
            payload={},
        )

        # Mock task handler that always fails
        def always_failing_handler(payload):
            raise Exception("Always fails")

        worker.register_handler("always_failing_task", always_failing_handler)

        # Process task until max retries
        for _ in range(3):  # Initial attempt + 2 retries
            worker.process_one()

        status = queue.get_task_status(task_id)
        assert status["status"] == "failed"
        assert "Always fails" in status.get("error", "")

    def test_concurrent_task_processing(self):
        """Test concurrent task processing."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.worker import TaskWorker

        queue = TaskQueue()

        # Add multiple tasks
        task_ids = []
        for i in range(5):
            task_id = queue.add_task(
                task_type="concurrent_task",
                payload={"index": i},
            )
            task_ids.append(task_id)

        # Track processed tasks
        processed = []

        def concurrent_handler(payload):
            processed.append(payload["index"])
            time.sleep(0.1)  # Simulate work

        # Create multiple workers
        workers = []
        for _ in range(3):
            worker = TaskWorker(queue)
            worker.register_handler("concurrent_task", concurrent_handler)
            workers.append(worker)

        # Process tasks concurrently
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for worker in workers:
                for _ in range(2):  # Each worker processes 2 tasks
                    future = executor.submit(worker.process_one)
                    futures.append(future)

            # Wait for completion
            concurrent.futures.wait(futures)

        # Verify all tasks were processed
        assert len(processed) == 5
        assert sorted(processed) == [0, 1, 2, 3, 4]

    def test_task_timeout(self):
        """Test task timeout handling."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.worker import TaskWorker

        queue = TaskQueue()
        worker = TaskWorker(
            queue, task_timeout=1, max_retries=0
        )  # 1 second timeout, no retries

        # Add a test task
        task_id = queue.add_task(
            task_type="slow_task",
            payload={},
        )

        # Mock task handler that takes too long
        def slow_handler(payload):
            time.sleep(2)  # Exceeds timeout

        worker.register_handler("slow_task", slow_handler)

        # Process task
        worker.process_one()

        status = queue.get_task_status(task_id)
        assert status["status"] == "failed"
        assert "timed out" in status.get("error", "").lower()

    def test_task_progress_updates(self):
        """Test task progress reporting."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.worker import TaskWorker

        queue = TaskQueue()
        worker = TaskWorker(queue)

        # Add a test task
        task_id = queue.add_task(
            task_type="progress_task",
            payload={},
        )

        # Mock task handler that reports progress
        def progress_handler(payload, task_context):
            task_context.update_progress(0.25, "Starting")
            time.sleep(0.1)
            task_context.update_progress(0.5, "Halfway")
            time.sleep(0.1)
            task_context.update_progress(0.75, "Almost done")
            time.sleep(0.1)
            task_context.update_progress(1.0, "Complete")

        worker.register_handler("progress_task", progress_handler)

        # Process task
        worker.process_one()

        status = queue.get_task_status(task_id)
        assert status["status"] == "completed"
        assert status["progress"] == 1.0
        assert status["message"] == "Complete"

    def test_health_file_processing_task(self):
        """Test health file processing as a background task."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.background.tasks import (
            register_health_processing_tasks,
        )
        from big_mood_detector.infrastructure.background.worker import TaskWorker

        queue = TaskQueue()
        worker = TaskWorker(queue)

        # Register health processing tasks
        register_health_processing_tasks(worker)

        # Create test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"data": []}')
            test_file = Path(f.name)

        try:
            # Add processing task
            task_id = queue.add_task(
                task_type="process_health_file",
                payload={
                    "file_path": str(test_file),
                    "upload_id": "test-upload-123",
                },
            )

            # Process task
            with patch(
                "big_mood_detector.application.use_cases.process_health_data_use_case.MoodPredictionPipeline"
            ) as mock_pipeline:
                mock_instance = Mock()
                mock_result = Mock()
                mock_result.overall_summary = {
                    "avg_depression_risk": 0.4,
                    "avg_hypomanic_risk": 0.05,
                    "avg_manic_risk": 0.01,
                    "days_analyzed": 7,
                }
                mock_result.confidence_score = 0.85
                mock_result.daily_predictions = {}
                mock_instance.process_apple_health_file.return_value = mock_result
                mock_pipeline.return_value = mock_instance

                worker.process_one()

            status = queue.get_task_status(task_id)
            assert status["status"] == "completed"

        finally:
            # Cleanup
            test_file.unlink(missing_ok=True)
