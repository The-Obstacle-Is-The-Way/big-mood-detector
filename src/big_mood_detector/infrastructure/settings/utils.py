"""Settings utilities for proper initialization."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from big_mood_detector.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


def ensure_directory(path: Path, mode: int = 0o755) -> None:
    """Ensure a directory exists with proper permissions.

    Args:
        path: Directory path to create
        mode: Unix permissions mode (default: 755)
    """
    try:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        logger.debug(f"Ensured directory exists: {path}")
    except PermissionError:
        # In Docker, we might not have permissions to create at runtime
        # Check if parent exists and is writable
        parent = path.parent
        if parent.exists() and os.access(parent, os.W_OK):
            # Parent is writable, try again
            try:
                path.mkdir(exist_ok=True, mode=mode)
            except Exception as e:
                logger.warning(f"Could not create directory {path}: {e}")
        else:
            logger.warning(f"No write permission for directory {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")


def initialize_directories(settings: "Settings") -> None:
    """Initialize all required directories from settings.

    This should be called at application startup, not during import.
    """
    directories = [
        settings.OUTPUT_DIR,
        settings.UPLOAD_DIR,
        settings.TEMP_DIR,
    ]

    # Also ensure DATA_DIR exists
    ensure_directory(settings.DATA_DIR)

    for directory in directories:
        ensure_directory(directory)

    # Set appropriate permissions for upload directory
    if settings.UPLOAD_DIR.exists():
        try:
            os.chmod(settings.UPLOAD_DIR, 0o775)
        except Exception:
            pass  # Ignore permission errors in Docker


def validate_model_paths(settings: "Settings") -> str | None:
    """Validate that model paths exist and are accessible.

    Returns:
        Error message if validation fails, None if successful
    """
    errors = []

    # Check model weights path
    if not settings.MODEL_WEIGHTS_PATH.exists():
        errors.append(f"Model weights path not found: {settings.MODEL_WEIGHTS_PATH}")

    # Check for specific model files
    expected_models = ["depression_risk.json", "hypomanic_risk.json", "manic_risk.json"]
    for model_file in expected_models:
        model_path = settings.MODEL_WEIGHTS_PATH / model_file
        if not model_path.exists():
            errors.append(f"Model file not found: {model_path}")

    return "\n".join(errors) if errors else None
