"""
File Upload API Routes

Handles health data file uploads and processing status tracking.
"""

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from big_mood_detector.infrastructure.background.task_queue import TaskQueue
from big_mood_detector.infrastructure.background.tasks import (
    register_health_processing_tasks,
)
from big_mood_detector.infrastructure.background.worker import TaskWorker

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

# In-memory storage for demo (use Redis/DB in production)
upload_status_store: dict[str, dict[str, Any]] = {}
upload_results_store: dict[str, Any] = {}

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Task queue and worker (singleton instances)
task_queue = TaskQueue()
task_worker = TaskWorker(task_queue)
register_health_processing_tasks(task_worker)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class UploadResponse(BaseModel):
    """Response model for file upload."""

    upload_id: str
    filename: str
    status: str
    metadata: dict[str, Any] | None = None


class BatchUploadResponse(BaseModel):
    """Response model for batch upload."""

    batch_id: str
    files: list[UploadResponse]


class UploadStatusResponse(BaseModel):
    """Response model for upload status."""

    upload_id: str
    status: str
    progress: float | None = None
    message: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


async def process_upload(
    upload_id: str, file_path: Path, metadata: dict[str, Any] | None = None
) -> None:
    """Process uploaded file using task queue.

    Args:
        upload_id: Unique upload identifier
        file_path: Path to uploaded file
        metadata: Optional metadata from upload
    """
    # Create task payload
    payload: dict[str, Any] = {
        "file_path": str(file_path),
        "upload_id": upload_id,
        "output_path": str(file_path.parent / f"{upload_id}_results.csv"),
    }

    if metadata:
        payload["metadata"] = metadata

    # Add task to queue
    task_id = task_queue.add_task(
        task_type="process_health_file",
        payload=payload,
        task_id=upload_id,  # Use upload_id as task_id for easy tracking
    )

    # Start background processing
    # In production, this would be handled by separate worker processes
    import threading

    def process_in_background() -> None:
        task_worker.process_one()

        # Update upload status based on task status
        task_status = task_queue.get_task_status(task_id)

        if task_status["status"] == "completed":
            # Extract results from task payload
            task = next(
                (t for t in task_queue._tasks.values() if t.id == task_id), None
            )
            if task and "result" in task.payload:
                upload_results_store[upload_id] = convert_numpy_types(
                    task.payload["result"]
                )
                upload_status_store[upload_id] = {
                    "status": "completed",
                    "progress": 1.0,
                    "message": "Processing complete",
                    "result": upload_results_store[upload_id],
                }
        elif task_status["status"] == "failed":
            upload_status_store[upload_id] = {
                "status": "failed",
                "progress": 0,
                "error": task_status.get("error", "Unknown error"),
            }
        else:
            # Update progress from task
            upload_status_store[upload_id] = {
                "status": task_status["status"],
                "progress": task_status.get("progress", 0),
                "message": task_status.get("message", "Processing..."),
            }

    # Run in background thread
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()


@router.post("/file", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: str | None = Form(None),
    date_range: str | None = Form(None),
    processing_options: str | None = Form(None),
) -> UploadResponse:
    """Upload a health data file for processing.

    Accepts:
    - Apple Health export XML files
    - Health Auto Export JSON files

    File size limit: 10MB
    """
    # Validate file type
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".xml", ".json"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Only .xml and .json files are allowed.",
            )

    # Check file size
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    # Generate upload ID
    upload_id = str(uuid.uuid4())

    # Save file
    upload_path = UPLOAD_DIR / upload_id
    upload_path.mkdir(exist_ok=True)
    file_path = upload_path / (file.filename or "upload.json")

    with open(file_path, "wb") as f:
        f.write(contents)

    # Prepare metadata
    metadata = {
        "patient_id": patient_id,
        "date_range": date_range,
        "processing_options": processing_options,
    }

    # Initialize status
    upload_status_store[upload_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Upload queued for processing",
    }

    # Queue processing
    background_tasks.add_task(process_upload, upload_id, file_path, metadata)

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename or "unknown",
        status="queued",
        metadata=metadata if any(metadata.values()) else None,
    )


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> BatchUploadResponse:
    """Upload multiple health data files.

    Useful for uploading separate JSON files from Health Auto Export.
    """
    batch_id = str(uuid.uuid4())
    upload_responses = []

    for file in files:
        # Process each file using the same logic as upload_file
        # Validate file type
        if file.filename:
            ext = Path(file.filename).suffix.lower()
            if ext not in [".xml", ".json"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported file type. Only .xml and .json files are allowed.",
                )

        # Check file size
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file",
            )

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
            )

        # Generate upload ID
        upload_id = str(uuid.uuid4())

        # Save file
        upload_path = UPLOAD_DIR / upload_id
        upload_path.mkdir(exist_ok=True)
        file_path = upload_path / (file.filename or "upload.json")

        with open(file_path, "wb") as f:
            f.write(contents)

        # Initialize status
        upload_status_store[upload_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Upload queued for processing",
        }

        # Queue processing
        background_tasks.add_task(process_upload, upload_id, file_path, None)

        upload_responses.append(
            UploadResponse(
                upload_id=upload_id,
                filename=file.filename or "unknown",
                status="queued",
                metadata=None,
            )
        )

    return BatchUploadResponse(
        batch_id=batch_id,
        files=upload_responses,
    )


@router.get("/status/{upload_id}", response_model=UploadStatusResponse)
async def get_upload_status(upload_id: str) -> UploadStatusResponse:
    """Get processing status for an upload."""
    if upload_id not in upload_status_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found",
        )

    status_data = upload_status_store[upload_id]

    return UploadStatusResponse(
        upload_id=upload_id,
        **status_data,
    )


@router.get("/result/{upload_id}")
async def get_upload_result(upload_id: str) -> dict[str, Any]:
    """Get processing results for a completed upload."""
    if upload_id not in upload_status_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found",
        )

    status_data = upload_status_store[upload_id]

    if status_data["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload is {status_data['status']}, not completed",
        )

    return {
        "upload_id": upload_id,
        "status": "completed",
        "result": status_data.get("result", {}),
    }


@router.get("/download/{upload_id}")
async def download_processed_file(upload_id: str) -> StreamingResponse:
    """Download processed results as CSV."""
    csv_path = UPLOAD_DIR / upload_id / f"{upload_id}_results.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed file not found",
        )

    def iterfile() -> Any:
        with open(csv_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={upload_id}_results.csv"
        },
    )


@router.get("/queue/stats")
async def get_queue_stats() -> dict[str, Any]:
    """Get task queue statistics."""
    return {
        "queue_stats": task_queue.get_stats(),
        "upload_count": len(upload_status_store),
    }


# Helper functions for testing
def get_processed_file(upload_id: str) -> bytes:
    """Get processed CSV content (for testing)."""
    csv_path = UPLOAD_DIR / upload_id / f"{upload_id}_results.csv"
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            return f.read()
    return b""
