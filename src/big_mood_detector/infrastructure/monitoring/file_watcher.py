"""
File Watcher Service

Monitors directories for new health data files and triggers processing.
"""

import asyncio
import fnmatch
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a monitored file."""

    path: Path
    size: int
    mtime: float
    
    def has_changed(self, other: "FileInfo") -> bool:
        """Check if file has changed compared to another FileInfo."""
        return self.size != other.size or self.mtime != other.mtime


FileHandler = Callable[[Path], None]
AsyncFileHandler = Callable[[Path], Coroutine[Any, Any, None]]
ErrorHandler = Callable[[Exception, Path], None]


class FileWatcher:
    """Watches directories for file changes."""

    def __init__(
        self,
        watch_path: Path,
        patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        recursive: bool = False,
        poll_interval: float = 1.0,
        state_file: Optional[Path] = None,
    ):
        """Initialize file watcher.
        
        Args:
            watch_path: Directory to watch
            patterns: File patterns to watch (e.g., ["*.xml", "*.json"])
            ignore_patterns: Patterns to ignore (e.g., ["*temp*", ".*"])
            recursive: Watch subdirectories recursively
            poll_interval: Seconds between polls
            state_file: Optional file to persist state between runs
        """
        self.watch_path = watch_path
        self.patterns = patterns or ["*"]
        self.ignore_patterns = ignore_patterns or []
        self.recursive = recursive
        self.poll_interval = poll_interval
        self.state_file = state_file
        
        # Always ignore the state file if provided
        if state_file:
            self.ignore_patterns.append(state_file.name)
        
        self._running = False
        self._file_state: dict[str, FileInfo] = {}
        self._created_handlers: list[FileHandler] = []
        self._modified_handlers: list[FileHandler] = []
        self._deleted_handlers: list[FileHandler] = []
        self._created_async_handlers: list[AsyncFileHandler] = []
        self._modified_async_handlers: list[AsyncFileHandler] = []
        self._deleted_async_handlers: list[AsyncFileHandler] = []
        self._error_handlers: list[ErrorHandler] = []
        
        # Load previous state if available
        if self.state_file and self.state_file.exists():
            self._load_state()

    def on_created(self, handler: FileHandler) -> None:
        """Register handler for file creation events."""
        self._created_handlers.append(handler)

    def on_modified(self, handler: FileHandler) -> None:
        """Register handler for file modification events."""
        self._modified_handlers.append(handler)

    def on_deleted(self, handler: FileHandler) -> None:
        """Register handler for file deletion events."""
        self._deleted_handlers.append(handler)

    def on_created_async(self, handler: AsyncFileHandler) -> None:
        """Register async handler for file creation events."""
        self._created_async_handlers.append(handler)

    def on_modified_async(self, handler: AsyncFileHandler) -> None:
        """Register async handler for file modification events."""
        self._modified_async_handlers.append(handler)

    def on_deleted_async(self, handler: AsyncFileHandler) -> None:
        """Register async handler for file deletion events."""
        self._deleted_async_handlers.append(handler)

    def on_error(self, handler: ErrorHandler) -> None:
        """Register error handler."""
        self._error_handlers.append(handler)

    def _matches_patterns(self, file_path: Path) -> bool:
        """Check if file matches watch patterns and not ignore patterns."""
        filename = file_path.name
        
        # Check ignore patterns first
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return False
        
        # Check include patterns
        for pattern in self.patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        
        return False

    def _get_files(self) -> dict[str, FileInfo]:
        """Get current state of files in watch directory."""
        current_files = {}
        
        if self.recursive:
            # Walk directory tree
            for root, dirs, files in os.walk(self.watch_path):
                # Filter out ignored directories
                dirs[:] = [
                    d for d in dirs
                    if not any(
                        fnmatch.fnmatch(d, pattern) 
                        for pattern in self.ignore_patterns
                    )
                ]
                
                root_path = Path(root)
                for filename in files:
                    file_path = root_path / filename
                    if self._matches_patterns(file_path):
                        try:
                            stat = file_path.stat()
                            current_files[str(file_path)] = FileInfo(
                                path=file_path,
                                size=stat.st_size,
                                mtime=stat.st_mtime,
                            )
                        except (OSError, IOError) as e:
                            logger.warning(f"Failed to stat {file_path}: {e}")
        else:
            # Only watch top-level directory
            try:
                for file_path in self.watch_path.iterdir():
                    if file_path.is_file() and self._matches_patterns(file_path):
                        try:
                            stat = file_path.stat()
                            current_files[str(file_path)] = FileInfo(
                                path=file_path,
                                size=stat.st_size,
                                mtime=stat.st_mtime,
                            )
                        except (OSError, IOError) as e:
                            logger.warning(f"Failed to stat {file_path}: {e}")
            except (OSError, IOError) as e:
                logger.error(f"Failed to list directory {self.watch_path}: {e}")
        
        return current_files

    def _check_for_changes(self) -> None:
        """Check for file changes and trigger handlers."""
        current_files = self._get_files()
        
        # Find created files
        for path_str, file_info in current_files.items():
            if path_str not in self._file_state:
                self._trigger_created(file_info.path)
        
        # Find modified and deleted files
        for path_str, old_info in self._file_state.items():
            if path_str in current_files:
                new_info = current_files[path_str]
                if old_info.has_changed(new_info):
                    self._trigger_modified(new_info.path)
            else:
                self._trigger_deleted(old_info.path)
        
        # Update state
        self._file_state = current_files

    def _trigger_created(self, file_path: Path) -> None:
        """Trigger file created handlers."""
        logger.info(f"File created: {file_path}")
        
        for handler in self._created_handlers:
            try:
                handler(file_path)
            except Exception as e:
                logger.error(f"Error in created handler: {e}", exc_info=True)
                self._handle_error(e, file_path)

    def _trigger_modified(self, file_path: Path) -> None:
        """Trigger file modified handlers."""
        logger.info(f"File modified: {file_path}")
        
        for handler in self._modified_handlers:
            try:
                handler(file_path)
            except Exception as e:
                logger.error(f"Error in modified handler: {e}", exc_info=True)
                self._handle_error(e, file_path)

    def _trigger_deleted(self, file_path: Path) -> None:
        """Trigger file deleted handlers."""
        logger.info(f"File deleted: {file_path}")
        
        for handler in self._deleted_handlers:
            try:
                handler(file_path)
            except Exception as e:
                logger.error(f"Error in deleted handler: {e}", exc_info=True)
                self._handle_error(e, file_path)

    async def _trigger_created_async(self, file_path: Path) -> None:
        """Trigger async file created handlers."""
        for handler in self._created_async_handlers:
            try:
                await handler(file_path)
            except Exception as e:
                logger.error(f"Error in async created handler: {e}", exc_info=True)
                self._handle_error(e, file_path)

    async def _trigger_modified_async(self, file_path: Path) -> None:
        """Trigger async file modified handlers."""
        for handler in self._modified_async_handlers:
            try:
                await handler(file_path)
            except Exception as e:
                logger.error(f"Error in async modified handler: {e}", exc_info=True)
                self._handle_error(e, file_path)

    async def _trigger_deleted_async(self, file_path: Path) -> None:
        """Trigger async file deleted handlers."""
        for handler in self._deleted_async_handlers:
            try:
                await handler(file_path)
            except Exception as e:
                logger.error(f"Error in async deleted handler: {e}", exc_info=True)
                self._handle_error(e, file_path)

    def _handle_error(self, error: Exception, file_path: Path) -> None:
        """Handle errors from file handlers."""
        for handler in self._error_handlers:
            try:
                handler(error, file_path)
            except Exception as e:
                logger.error(f"Error in error handler: {e}", exc_info=True)

    async def _check_for_changes_async(self) -> None:
        """Check for file changes and trigger async handlers."""
        current_files = self._get_files()
        
        # Find created files
        created_tasks = []
        for path_str, file_info in current_files.items():
            if path_str not in self._file_state:
                self._trigger_created(file_info.path)
                if self._created_async_handlers:
                    created_tasks.append(
                        self._trigger_created_async(file_info.path)
                    )
        
        # Find modified and deleted files
        modified_tasks = []
        deleted_tasks = []
        for path_str, old_info in self._file_state.items():
            if path_str in current_files:
                new_info = current_files[path_str]
                if old_info.has_changed(new_info):
                    self._trigger_modified(new_info.path)
                    if self._modified_async_handlers:
                        modified_tasks.append(
                            self._trigger_modified_async(new_info.path)
                        )
            else:
                self._trigger_deleted(old_info.path)
                if self._deleted_async_handlers:
                    deleted_tasks.append(
                        self._trigger_deleted_async(old_info.path)
                    )
        
        # Wait for all async handlers
        all_tasks = created_tasks + modified_tasks + deleted_tasks
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Update state
        self._file_state = current_files

    def watch(self) -> None:
        """Start watching for file changes (blocking)."""
        self._running = True
        logger.info(f"Starting file watcher on {self.watch_path}")
        
        try:
            while self._running:
                self._check_for_changes()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("File watcher interrupted")
        finally:
            self.save_state()
            logger.info("File watcher stopped")

    async def watch_async(self) -> None:
        """Start watching for file changes (async)."""
        self._running = True
        logger.info(f"Starting async file watcher on {self.watch_path}")
        
        try:
            while self._running:
                await self._check_for_changes_async()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Async file watcher cancelled")
        finally:
            self.save_state()
            logger.info("Async file watcher stopped")

    def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False

    def save_state(self) -> None:
        """Save current file state to disk."""
        if not self.state_file:
            return
        
        try:
            state_data = {
                path_str: {
                    "size": info.size,
                    "mtime": info.mtime,
                }
                for path_str, info in self._file_state.items()
            }
            
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state_data, indent=2))
            logger.debug(f"Saved watcher state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save watcher state: {e}")

    def _load_state(self) -> None:
        """Load file state from disk."""
        if not self.state_file or not self.state_file.exists():
            return
        
        try:
            state_data = json.loads(self.state_file.read_text())
            self._file_state = {
                path_str: FileInfo(
                    path=Path(path_str),
                    size=info["size"],
                    mtime=info["mtime"],
                )
                for path_str, info in state_data.items()
            }
            logger.debug(f"Loaded watcher state from {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to load watcher state: {e}")
            self._file_state = {}