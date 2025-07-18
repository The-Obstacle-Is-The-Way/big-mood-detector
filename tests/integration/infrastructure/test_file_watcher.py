"""
Test File Watcher Service

TDD for file watching functionality.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest


class TestFileWatcher:
    """Test file watcher service."""

    def test_file_watcher_imports(self):
        """Test that file watcher can be imported."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        assert FileWatcher is not None

    def test_create_file_watcher(self):
        """Test creating a file watcher instance."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(Path(tmpdir))
            assert watcher is not None
            assert watcher.watch_path == Path(tmpdir)

    def test_file_watcher_configuration(self):
        """Test file watcher configuration options."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(
                watch_path=Path(tmpdir),
                patterns=["*.xml", "*.json"],
                recursive=True,
                poll_interval=2.0,
            )

            assert watcher.patterns == ["*.xml", "*.json"]
            assert watcher.recursive is True
            assert watcher.poll_interval == 2.0

    def test_register_handler(self):
        """Test registering file event handlers."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(Path(tmpdir))

            handler_called = False

            def on_file_created(file_path: Path):
                nonlocal handler_called
                handler_called = True

            watcher.on_created(on_file_created)

            # Simulate file creation event
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text('{"data": []}')

            # Process events
            watcher._check_for_changes()

            assert handler_called

    def test_file_pattern_matching(self):
        """Test that only files matching patterns are processed."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(
                Path(tmpdir),
                patterns=["*.json", "*.xml"],
            )

            processed_files = []

            def handler(file_path: Path):
                processed_files.append(file_path.name)

            watcher.on_created(handler)

            # Create various files
            (Path(tmpdir) / "data.json").write_text("{}")
            (Path(tmpdir) / "export.xml").write_text("<data/>")
            (Path(tmpdir) / "readme.txt").write_text("ignore me")
            (Path(tmpdir) / "script.py").write_text("print('hi')")

            # Process events
            watcher._check_for_changes()

            # Only JSON and XML files should be processed
            assert len(processed_files) == 2
            assert "data.json" in processed_files
            assert "export.xml" in processed_files
            assert "readme.txt" not in processed_files

    def test_recursive_watching(self):
        """Test recursive directory watching."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            watcher = FileWatcher(
                Path(tmpdir),
                patterns=["*.json"],
                recursive=True,
            )

            processed_files = []

            def handler(file_path: Path):
                processed_files.append(str(file_path.relative_to(tmpdir)))

            watcher.on_created(handler)

            # Create files in root and subdirectory
            (Path(tmpdir) / "root.json").write_text("{}")
            (subdir / "sub.json").write_text("{}")

            # Process events
            watcher._check_for_changes()

            assert len(processed_files) == 2
            assert "root.json" in processed_files
            assert "subdir/sub.json" in processed_files

    def test_file_modified_events(self):
        """Test detecting file modifications."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text('{"version": 1}')

            watcher = FileWatcher(Path(tmpdir))

            # Initialize watcher (records initial state)
            watcher._check_for_changes()

            modified_files = []

            def on_modified(file_path: Path):
                modified_files.append(file_path.name)

            watcher.on_modified(on_modified)

            # Modify file
            time.sleep(0.1)  # Ensure modification time changes
            test_file.write_text('{"version": 2}')

            # Process events
            watcher._check_for_changes()

            assert "test.json" in modified_files

    def test_file_deleted_events(self):
        """Test detecting file deletions."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text("{}")

            watcher = FileWatcher(Path(tmpdir))

            # Initialize watcher (records initial state)
            watcher._check_for_changes()

            deleted_files = []

            def on_deleted(file_path: Path):
                deleted_files.append(file_path.name)

            watcher.on_deleted(on_deleted)

            # Delete file
            test_file.unlink()

            # Process events
            watcher._check_for_changes()

            assert "test.json" in deleted_files

    @pytest.mark.asyncio
    async def test_async_file_watching(self):
        """Test async file watching."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(
                Path(tmpdir),
                poll_interval=0.1,  # Fast polling for test
            )

            created_files = []

            async def async_handler(file_path: Path):
                created_files.append(file_path.name)
                await asyncio.sleep(0.01)  # Simulate async work

            watcher.on_created_async(async_handler)

            # Start watching in background
            watch_task = asyncio.create_task(watcher.watch_async())

            # Give watcher time to start
            await asyncio.sleep(0.2)

            # Create files while watching
            (Path(tmpdir) / "file1.json").write_text("{}")
            await asyncio.sleep(0.2)

            (Path(tmpdir) / "file2.json").write_text("{}")
            await asyncio.sleep(0.2)

            # Stop watching
            watcher.stop()
            await watch_task

            assert "file1.json" in created_files
            assert "file2.json" in created_files

    def test_error_handling(self):
        """Test error handling in file handlers."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(Path(tmpdir))

            processed = []
            errors = []

            def good_handler(file_path: Path):
                processed.append(file_path.name)

            def bad_handler(file_path: Path):
                raise Exception(f"Failed to process {file_path.name}")

            # Register both handlers
            watcher.on_created(bad_handler)  # This will fail
            watcher.on_created(good_handler)  # This should still run

            # Set error handler
            watcher.on_error(lambda e, f: errors.append((str(e), f.name)))

            # Create file
            (Path(tmpdir) / "test.json").write_text("{}")

            # Process events
            watcher._check_for_changes()

            # Good handler should have run despite bad handler failing
            assert "test.json" in processed
            assert len(errors) == 1
            assert "Failed to process test.json" in errors[0][0]

    def test_integration_with_task_queue(self):
        """Test integration with background task queue."""
        from big_mood_detector.infrastructure.background.task_queue import TaskQueue
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create task queue
            task_queue = TaskQueue()

            # Create watcher
            watcher = FileWatcher(Path(tmpdir))

            # Handler that adds processing tasks
            def process_file(file_path: Path):
                task_queue.add_task(
                    task_type="process_health_file",
                    payload={
                        "file_path": str(file_path),
                        "auto_processed": True,
                    },
                )

            watcher.on_created(process_file)

            # Create health data files
            (Path(tmpdir) / "export.xml").write_text("<HealthData/>")
            (Path(tmpdir) / "sleep.json").write_text('{"data": []}')

            # Process events
            watcher._check_for_changes()

            # Check tasks were queued
            assert len(task_queue.get_pending_tasks()) == 2

    def test_watcher_with_ignore_patterns(self):
        """Test ignoring certain file patterns."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(
                Path(tmpdir),
                patterns=["*.json"],
                ignore_patterns=["*temp*", "*backup*", ".*"],
            )

            processed = []

            def handler(file_path: Path):
                processed.append(file_path.name)

            watcher.on_created(handler)

            # Create various files
            (Path(tmpdir) / "data.json").write_text("{}")
            (Path(tmpdir) / "temp.json").write_text("{}")
            (Path(tmpdir) / "backup.json").write_text("{}")
            (Path(tmpdir) / ".hidden.json").write_text("{}")
            (Path(tmpdir) / "important.json").write_text("{}")

            # Process events
            watcher._check_for_changes()

            # Only non-ignored files should be processed
            assert len(processed) == 2
            assert "data.json" in processed
            assert "important.json" in processed
            assert "temp.json" not in processed
            assert "backup.json" not in processed

    def test_watcher_state_persistence(self):
        """Test that watcher can persist and restore state."""
        from big_mood_detector.infrastructure.monitoring.file_watcher import FileWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".watcher_state.json"

            # Create initial files
            (Path(tmpdir) / "existing1.json").write_text("{}")
            (Path(tmpdir) / "existing2.json").write_text("{}")

            # First watcher session
            watcher1 = FileWatcher(
                Path(tmpdir),
                state_file=state_file,
            )

            # Initialize (should save state)
            watcher1._check_for_changes()
            watcher1.save_state()

            # Create new watcher with same state file
            watcher2 = FileWatcher(
                Path(tmpdir),
                state_file=state_file,
            )

            created = []
            watcher2.on_created(lambda f: created.append(f.name))

            # Should not detect existing files as new
            watcher2._check_for_changes()
            assert len(created) == 0

            # Create new file
            (Path(tmpdir) / "new.json").write_text("{}")
            watcher2._check_for_changes()

            # Should only detect the new file
            assert len(created) == 1
            assert "new.json" in created
