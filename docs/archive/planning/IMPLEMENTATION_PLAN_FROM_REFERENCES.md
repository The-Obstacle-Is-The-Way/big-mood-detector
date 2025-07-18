# Implementation Plan from Reference Repos

Based on analysis of `fastapi-fullstack` and `fastapi-users`, here's what we need to implement:

## 1. Settings/Configuration Pattern (HIGHEST PRIORITY)

### Current State
- Settings are hardcoded throughout the codebase
- No environment-specific configuration
- No validation of settings

### Implementation Needed

Create `src/big_mood_detector/core/config.py`:

```python
from typing import Literal
from pathlib import Path
from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Big Mood Detector"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    
    # Paths
    MODEL_WEIGHTS_PATH: Path = Path("model_weights/xgboost/converted")
    OUTPUT_DIR: Path = Path("output")
    UPLOAD_DIR: Path = Path("uploads")
    TEMP_DIR: Path = Path("temp")
    
    # File Processing
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list[str] = [".xml", ".json"]
    
    # ML Configuration
    CONFIDENCE_THRESHOLD: float = 0.7
    USE_PAT_MODEL: bool = False  # Until we implement PAT
    
    # Background Tasks
    TASK_TIMEOUT: int = 300  # 5 minutes
    MAX_RETRIES: int = 3
    
    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: str = "json"  # or "text"
    
    # Clinical Thresholds
    DEPRESSION_THRESHOLD: float = 0.5
    HYPOMANIC_THRESHOLD: float = 0.3
    MANIC_THRESHOLD: float = 0.3
    
    @field_validator("MODEL_WEIGHTS_PATH", "OUTPUT_DIR", "UPLOAD_DIR", "TEMP_DIR")
    def validate_paths(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @computed_field
    @property
    def log_config(self) -> dict:
        """Generate logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(timestamp)s %(level)s %(name)s %(message)s"
                },
                "text": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": self.LOG_FORMAT,
                    "level": self.LOG_LEVEL
                }
            },
            "root": {
                "level": self.LOG_LEVEL,
                "handlers": ["console"]
            }
        }


settings = Settings()
```

### Usage Pattern

```python
# In any module
from big_mood_detector.core.config import settings

# Use settings
pipeline = MoodPredictionPipeline(
    model_path=settings.MODEL_WEIGHTS_PATH,
    confidence_threshold=settings.CONFIDENCE_THRESHOLD
)
```

## 2. Logging Pattern

### Current State
- Using `print()` statements everywhere
- No structured logging
- No request tracing

### Implementation Needed

Create `src/big_mood_detector/core/logging.py`:

```python
import logging
import sys
from functools import lru_cache
from typing import Any

import structlog
from structlog.types import FilteringBoundLogger

from .config import settings


@lru_cache()
def get_logger() -> FilteringBoundLogger:
    """Get configured logger instance."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


# Module-level logger
logger = get_logger()
```

### Usage Pattern

Replace all `print()` statements:

```python
# OLD
print(f"Processing {len(records)} records...")

# NEW
from big_mood_detector.core.logging import logger
logger.info("processing_records", count=len(records))
```

## 3. Dependency Injection Pattern

### Current State
- Direct instantiation of services
- No dependency management
- Hard to test

### Implementation Needed

Create `src/big_mood_detector/api/deps.py`:

```python
from functools import lru_cache
from typing import Generator

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.core.config import settings
from big_mood_detector.infrastructure.background.task_queue import TaskQueue
from big_mood_detector.infrastructure.background.worker import TaskWorker


@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return settings


@lru_cache()
def get_task_queue() -> TaskQueue:
    """Get singleton task queue."""
    return TaskQueue()


@lru_cache()
def get_task_worker() -> TaskWorker:
    """Get singleton task worker."""
    from big_mood_detector.infrastructure.background.tasks import (
        register_health_processing_tasks,
    )
    
    queue = get_task_queue()
    worker = TaskWorker(queue)
    register_health_processing_tasks(worker)
    return worker


@lru_cache()
def get_pipeline() -> MoodPredictionPipeline:
    """Get configured pipeline instance."""
    return MoodPredictionPipeline()
```

### Usage in API

```python
from fastapi import Depends
from big_mood_detector.api.deps import get_pipeline, get_task_queue

@router.post("/process")
async def process_file(
    file: UploadFile,
    pipeline: MoodPredictionPipeline = Depends(get_pipeline),
    task_queue: TaskQueue = Depends(get_task_queue),
):
    # Use injected dependencies
    task_id = task_queue.add_task(...)
```

## 4. Error Handling Pattern

### Current State
- Basic try/except blocks
- No consistent error responses
- No error tracking

### Implementation Needed

Create `src/big_mood_detector/core/exceptions.py`:

```python
from typing import Any, Optional

from fastapi import HTTPException, status


class BigMoodException(Exception):
    """Base exception for all custom exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(message)


class FileProcessingError(BigMoodException):
    """Raised when file processing fails."""
    pass


class ModelNotFoundError(BigMoodException):
    """Raised when ML model files are missing."""
    pass


class InsufficientDataError(BigMoodException):
    """Raised when not enough data for predictions."""
    pass


class ValidationError(BigMoodException):
    """Raised when data validation fails."""
    pass


# HTTP exceptions
class BadRequestError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class NotFoundError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
```

## 5. API Structure Improvements

### Current State
- Basic router structure
- No middleware
- No request ID tracking

### Implementation Needed

Update `src/big_mood_detector/interfaces/api/main.py`:

```python
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from big_mood_detector.core.config import settings
from big_mood_detector.core.exceptions import BigMoodException
from big_mood_detector.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("starting_application", environment=settings.ENVIRONMENT)
    yield
    # Shutdown
    logger.info("shutting_down_application")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(
        "request_processed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        process_time=process_time,
        status_code=response.status_code,
    )
    
    return response


@app.exception_handler(BigMoodException)
async def bigmood_exception_handler(request: Request, exc: BigMoodException):
    """Handle custom exceptions."""
    logger.error(
        "application_error",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        request_id=getattr(request.state, "request_id", None),
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )
```

## 6. Testing Improvements

### Current State
- Basic test structure
- No async test utilities
- No shared fixtures

### Implementation Needed

Create `tests/conftest.py`:

```python
import asyncio
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from big_mood_detector.core.config import Settings, settings
from big_mood_detector.interfaces.api.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Override settings for tests."""
    return Settings(
        ENVIRONMENT="testing",
        LOG_LEVEL="DEBUG",
        OUTPUT_DIR=Path(tempfile.mkdtemp()),
        UPLOAD_DIR=Path(tempfile.mkdtemp()),
    )


@pytest.fixture
def client(test_settings) -> Generator:
    """Create test client with overridden settings."""
    # Override settings
    app.dependency_overrides[get_settings] = lambda: test_settings
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_health_data() -> dict:
    """Sample health data for testing."""
    return {
        "data": {
            "metrics": [
                {
                    "name": "sleep_analysis",
                    "units": "hr",
                    "data": [
                        {"date": "2024-01-01", "qty": 7.5},
                        {"date": "2024-01-02", "qty": 8.0},
                    ],
                }
            ]
        }
    }
```

## Implementation Order

1. **Settings/Config** (2 hours)
   - Create config.py
   - Create .env.example
   - Update all hardcoded values

2. **Logging** (3 hours)
   - Create logging.py
   - Replace all print statements
   - Add structlog

3. **Dependency Injection** (2 hours)
   - Create deps.py
   - Update API routes
   - Update CLI commands

4. **Error Handling** (2 hours)
   - Create exceptions.py
   - Add exception handlers
   - Update error responses

5. **API Improvements** (2 hours)
   - Add middleware
   - Add request tracking
   - Improve startup/shutdown

6. **Testing** (2 hours)
   - Create conftest.py
   - Add async fixtures
   - Improve test organization

## Benefits

- **Production Ready**: Proper configuration, logging, and error handling
- **Maintainable**: Clear patterns, dependency injection
- **Observable**: Structured logging, request tracking
- **Testable**: Better fixtures, overridable dependencies
- **Scalable**: Ready for Redis, Celery, etc.

## Next Steps

After implementing these patterns:
1. Add Redis for caching and task queue
2. Add Celery for distributed task processing
3. Add OpenTelemetry for distributed tracing
4. Add Prometheus metrics
5. Add health check endpoints