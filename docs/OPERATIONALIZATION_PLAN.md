# Big Mood Detector - Operationalization Plan

## Executive Summary
Transform the working data processing pipeline into a production-ready application with proper entry points, APIs, and operational infrastructure.

## Current State
- ✅ Core processing pipeline works via scripts
- ✅ ML models integrated and functional
- ✅ Data parsers handle both XML and JSON
- ❌ No API endpoints
- ❌ No automated ingestion
- ❌ No background processing
- ❌ No state persistence

## Architecture Overview

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Sources     │     │  Entry Points    │     │   Processing    │
├─────────────────────┤     ├──────────────────┤     ├─────────────────┤
│ • File Upload (API) │────▶│ • FastAPI Server │────▶│ • Job Queue     │
│ • Folder Watch      │     │ • CLI Commands   │     │ • Async Tasks   │
│ • Direct API Call   │     │ • File Watcher   │     │ • State Machine │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
                                      │                        │
                                      ▼                        ▼
                            ┌──────────────────┐     ┌─────────────────┐
                            │   Data Storage   │     │    Results      │
                            ├──────────────────┤     ├─────────────────┤
                            │ • PostgreSQL     │◀────│ • Predictions   │
                            │ • File Storage   │     │ • Reports       │
                            │ • Redis Cache    │     │ • Metrics       │
                            └──────────────────┘     └─────────────────┘
```

## Phase 6: Application Operationalization

### Phase 6.1: FastAPI Application Structure (Day 1)

#### 6.1.1 Core API Structure
```python
src/big_mood_detector/interfaces/api/
├── __init__.py
├── main.py                 # FastAPI app instance
├── dependencies.py         # Dependency injection
├── middleware.py          # CORS, auth, logging
├── routers/
│   ├── __init__.py
│   ├── health.py          # Health check endpoints
│   ├── upload.py          # File upload endpoints
│   ├── processing.py      # Processing control
│   └── results.py         # Results retrieval
├── models/
│   ├── __init__.py
│   ├── requests.py        # Pydantic request models
│   └── responses.py       # Pydantic response models
└── services/
    ├── __init__.py
    ├── file_service.py    # File handling
    └── job_service.py     # Job management
```

#### 6.1.2 Main Application Entry
```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, upload, processing, results

app = FastAPI(
    title="Big Mood Detector API",
    description="Clinical mood prediction from health data",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, tags=["health"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(processing.router, prefix="/api/v1/process", tags=["processing"])
app.include_router(results.router, prefix="/api/v1/results", tags=["results"])
```

#### 6.1.3 TDD Tests Required
```python
tests/integration/api/
├── test_health_endpoints.py
├── test_upload_endpoints.py
├── test_processing_endpoints.py
└── test_results_endpoints.py
```

### Phase 6.2: File Upload and Processing Endpoints (Day 2)

#### 6.2.1 Upload Endpoints
```python
POST /api/v1/upload/file
- Accept: multipart/form-data
- File types: .xml, .json
- Response: { "upload_id": "uuid", "status": "uploaded" }

POST /api/v1/upload/batch
- Accept: multipart/form-data (multiple files)
- Response: { "batch_id": "uuid", "files": [...] }
```

#### 6.2.2 Processing Endpoints
```python
POST /api/v1/process/start
- Body: { "upload_id": "uuid", "options": {...} }
- Response: { "job_id": "uuid", "status": "queued" }

GET /api/v1/process/status/{job_id}
- Response: { "status": "processing", "progress": 0.75 }

POST /api/v1/process/cancel/{job_id}
- Response: { "status": "cancelled" }
```

#### 6.2.3 Implementation Details
- Use aiofiles for async file operations
- Store uploads in configurable directory
- Generate unique IDs for tracking
- Validate file formats before accepting

### Phase 6.3: Background Task Processing (Day 3)

#### 6.3.1 Celery Integration
```python
src/big_mood_detector/infrastructure/tasks/
├── __init__.py
├── celery_app.py          # Celery configuration
├── processing_tasks.py    # Main processing tasks
└── cleanup_tasks.py       # Maintenance tasks
```

#### 6.3.2 Task Definitions
```python
@celery.task(bind=True, max_retries=3)
def process_health_data(self, upload_id: str, options: dict):
    """Process uploaded health data asynchronously."""
    try:
        # 1. Load file from storage
        # 2. Parse data
        # 3. Run pipeline
        # 4. Store results
        # 5. Update job status
    except Exception as exc:
        self.retry(exc=exc, countdown=60)
```

#### 6.3.3 Redis Integration
- Job status tracking
- Progress updates
- Result caching
- Rate limiting

### Phase 6.4: File Watcher Service (Day 4)

#### 6.4.1 Watchdog Integration
```python
src/big_mood_detector/application/services/file_watcher_service.py
- Monitor configured directories
- Detect new/modified files
- Queue processing automatically
- Handle file patterns (*.json, *.xml)
```

#### 6.4.2 Configuration
```yaml
# config/watcher.yaml
watch_directories:
  - path: /data/health_auto_export
    patterns: ["*.json"]
    recursive: false
  - path: /data/apple_export
    patterns: ["export.xml"]
    recursive: true
    
processing:
  debounce_seconds: 5
  batch_size: 10
  auto_process: true
```

### Phase 6.5: CLI Entry Points (Day 5)

#### 6.5.1 Click CLI Structure
```python
# src/big_mood_detector/__main__.py
import click

@click.group()
def cli():
    """Big Mood Detector CLI"""
    pass

@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
def serve(host, port):
    """Start the API server"""
    import uvicorn
    uvicorn.run("big_mood_detector.interfaces.api.main:app", 
                host=host, port=port, reload=True)

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path')
def process(path, output):
    """Process health data file"""
    # Implementation

@cli.command()
@click.option('--config', '-c', help='Watcher config file')
def watch(config):
    """Start file watcher service"""
    # Implementation

if __name__ == '__main__':
    cli()
```

#### 6.5.2 Entry Point Configuration
```toml
# pyproject.toml
[project.scripts]
big-mood = "big_mood_detector.__main__:cli"
```

### Phase 6.6: State Persistence Layer (Day 6)

#### 6.6.1 Database Models
```python
src/big_mood_detector/infrastructure/database/
├── __init__.py
├── models.py              # SQLAlchemy models
├── repositories.py        # Repository pattern
└── migrations/            # Alembic migrations
```

#### 6.6.2 Core Models
```python
class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id = Column(UUID, primary_key=True)
    upload_id = Column(UUID, nullable=False)
    status = Column(Enum(JobStatus))
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
class PredictionResult(Base):
    __tablename__ = "prediction_results"
    
    id = Column(UUID, primary_key=True)
    job_id = Column(UUID, ForeignKey("processing_jobs.id"))
    date = Column(Date)
    depression_risk = Column(Float)
    hypomanic_risk = Column(Float)
    manic_risk = Column(Float)
    confidence = Column(Float)
```

### Phase 6.7: Deployment Configuration (Day 7)

#### 6.7.1 Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[prod]"
CMD ["big-mood", "serve"]
```

#### 6.7.2 Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db/bigmood
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
      
  worker:
    build: .
    command: celery -A big_mood_detector.infrastructure.tasks worker
    environment:
      - DATABASE_URL=postgresql://user:pass@db/bigmood
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
      
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=bigmood
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      
  redis:
    image: redis:7
```

## Implementation Strategy

### Week 1: Core Infrastructure
1. **Monday**: FastAPI structure + health endpoints
2. **Tuesday**: Upload endpoints + file handling
3. **Wednesday**: Background tasks + Celery
4. **Thursday**: File watcher service
5. **Friday**: CLI commands

### Week 2: Persistence & Deployment
1. **Monday**: Database models + migrations
2. **Tuesday**: Repository pattern implementation
3. **Wednesday**: Docker configuration
4. **Thursday**: Integration testing
5. **Friday**: Documentation + deployment

## Testing Strategy

### Unit Tests
- Each service class
- Each API endpoint
- Each background task
- Each CLI command

### Integration Tests
- End-to-end file upload → processing → results
- File watcher → automatic processing
- API → background tasks → database

### Load Tests
- Concurrent file uploads
- Multiple processing jobs
- Large file handling

## Success Criteria

1. **API Availability**: 99.9% uptime
2. **Processing Time**: < 30s for typical dataset
3. **Concurrent Jobs**: Support 10+ simultaneous
4. **File Size**: Handle up to 1GB XML files
5. **Response Time**: < 200ms for API calls

## Risk Mitigation

1. **File Storage**: Implement cleanup jobs
2. **Memory Usage**: Stream large files
3. **Job Failures**: Retry mechanism + DLQ
4. **Security**: Input validation + rate limiting
5. **Monitoring**: Prometheus metrics + alerts

## Next Steps

1. Review and approve plan
2. Set up development environment
3. Create GitHub issues for each phase
4. Begin TDD implementation
5. Daily progress reviews

---

*This plan provides a comprehensive roadmap for operationalizing the Big Mood Detector application with proper entry points, APIs, and production infrastructure.*