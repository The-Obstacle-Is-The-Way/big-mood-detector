# Big Mood Detector - Tailored Operationalization Plan

## Executive Summary
Transform our working pipeline into a production-ready application by building on existing infrastructure, leveraging reference implementations, and focusing on pragmatic solutions.

## Current State Audit

### ✅ What We Have
1. **Working Core Pipeline**
   - Data parsing (XML/JSON) ✓
   - Feature extraction ✓
   - ML predictions ✓
   - Clinical decisions ✓

2. **Existing API Structure**
   - FastAPI app skeleton at `interfaces/api/main.py`
   - Clinical routes with Pydantic models
   - Health check endpoint
   - Proper project structure

3. **CLI Entry Point**
   - Defined in pyproject.toml: `mood-detector`
   - Points to `big_mood_detector.cli:main`
   - Not yet implemented

4. **Reference Assets**
   - fastapi-fullstack: Production patterns, auth, deployment
   - fastapi-users: User management patterns
   - fhir-client: Healthcare data standards

### ❌ What's Missing
1. CLI implementation
2. File upload endpoints
3. Background processing
4. State persistence
5. File monitoring
6. Docker configuration

## Pragmatic Implementation Plan

### Phase 1: CLI Entry Point (4 hours)
**Goal**: Get the app running end-to-end with minimal complexity

#### 1.1 Create CLI Module
```python
# src/big_mood_detector/cli.py
import click
from pathlib import Path
from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline
)

@click.group()
def main():
    """Big Mood Detector CLI"""
    pass

@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='mood_predictions.csv')
@click.option('--report/--no-report', default=True)
def process(input_path, output, report):
    """Process health data and generate predictions"""
    pipeline = MoodPredictionPipeline()
    result = pipeline.process_apple_health_file(
        file_path=Path(input_path)
    )
    
    # Save results
    pipeline.export_results(result, Path(output))
    
    # Print summary
    if result.overall_summary:
        click.echo(f"\n📊 Analysis Complete!")
        click.echo(f"Depression Risk: {result.overall_summary.get('avg_depression_risk', 0):.1%}")
        click.echo(f"Hypomanic Risk: {result.overall_summary.get('avg_hypomanic_risk', 0):.1%}")
        click.echo(f"Manic Risk: {result.overall_summary.get('avg_manic_risk', 0):.1%}")
    
    if report:
        # Generate clinical report
        _generate_clinical_report(result, Path(output).with_suffix('.txt'))

@main.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
def serve(host, port):
    """Start the API server"""
    import uvicorn
    uvicorn.run(
        "big_mood_detector.interfaces.api.main:app",
        host=host,
        port=port,
        reload=True
    )

@main.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--poll-interval', default=60)
def watch(directory, poll_interval):
    """Watch directory for new health data files"""
    from big_mood_detector.application.services.file_watcher import FileWatcher
    watcher = FileWatcher(Path(directory), poll_interval)
    watcher.start()
```

#### 1.2 TDD Tests
```python
# tests/unit/test_cli.py
def test_cli_process_command():
    """Test CLI process command"""
    runner = CliRunner()
    result = runner.invoke(main, ['process', 'test_data/'])
    assert result.exit_code == 0
    assert "Analysis Complete" in result.output
```

### Phase 2: Enhanced API Endpoints (4 hours)
**Goal**: Add file upload and processing endpoints to existing API

#### 2.1 Upload Endpoints
```python
# src/big_mood_detector/interfaces/api/routes/upload.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pathlib import Path
import uuid

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

@router.post("/file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload health data file for processing"""
    # Save file
    upload_id = str(uuid.uuid4())
    file_path = Path(f"uploads/{upload_id}/{file.filename}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Queue processing
    background_tasks.add_task(process_upload, upload_id, file_path)
    
    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "status": "queued"
    }

@router.get("/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get processing status for upload"""
    # Check Redis or DB for status
    return {
        "upload_id": upload_id,
        "status": "processing",
        "progress": 0.75
    }
```

#### 2.2 Processing Endpoints
```python
# src/big_mood_detector/interfaces/api/routes/processing.py
@router.post("/process")
async def process_data(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Process health data with options"""
    job_id = str(uuid.uuid4())
    
    # Queue the job
    background_tasks.add_task(
        run_processing,
        job_id,
        request.input_path,
        request.options
    )
    
    return {"job_id": job_id, "status": "queued"}
```

### Phase 3: Simple Background Processing (2 hours)
**Goal**: Use FastAPI BackgroundTasks first, add Celery later if needed

#### 3.1 Background Task Functions
```python
# src/big_mood_detector/application/services/background_tasks.py
async def process_upload(upload_id: str, file_path: Path):
    """Process uploaded file in background"""
    try:
        # Update status
        await update_job_status(upload_id, "processing")
        
        # Run pipeline
        pipeline = MoodPredictionPipeline()
        result = pipeline.process_apple_health_file(file_path)
        
        # Store results
        await store_results(upload_id, result)
        
        # Update status
        await update_job_status(upload_id, "completed")
        
    except Exception as e:
        await update_job_status(upload_id, "failed", str(e))
```

### Phase 4: File Watcher Service (3 hours)
**Goal**: Simple folder monitoring without over-engineering

#### 4.1 Basic File Watcher
```python
# src/big_mood_detector/application/services/file_watcher.py
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class HealthDataHandler(FileSystemEventHandler):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.processed_files = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix in ['.json', '.xml']:
            if file_path not in self.processed_files:
                self.process_file(file_path)
                self.processed_files.add(file_path)
    
    def process_file(self, file_path: Path):
        """Process new health data file"""
        print(f"Processing: {file_path}")
        result = self.pipeline.process_apple_health_file(file_path)
        # Save results...
```

### Phase 5: Minimal State Persistence (2 hours)
**Goal**: SQLite for MVP, PostgreSQL for production

#### 5.1 Simple Models
```python
# src/big_mood_detector/infrastructure/database/models.py
from sqlalchemy import Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True)
    file_path = Column(String)
    status = Column(String)
    created_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
```

### Phase 6: Docker Configuration (2 hours)
**Goal**: Simple containerization for easy deployment

#### 6.1 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy application
COPY src/ src/
COPY models/ models/

# Run
CMD ["mood-detector", "serve"]
```

#### 6.2 Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./health_auto_export:/data
      - ./uploads:/app/uploads
    environment:
      - DATABASE_URL=sqlite:///./big_mood.db
    command: mood-detector serve
    
  watcher:
    build: .
    volumes:
      - ./health_auto_export:/data
      - ./uploads:/app/uploads
    environment:
      - DATABASE_URL=sqlite:///./big_mood.db
    command: mood-detector watch /data
```

## Implementation Timeline

### Day 1 (Today)
1. **Morning**: Implement CLI commands (2h)
2. **Afternoon**: Add upload endpoints (2h)

### Day 2
1. **Morning**: Background processing (2h)
2. **Afternoon**: File watcher (2h)

### Day 3
1. **Morning**: State persistence (2h)
2. **Afternoon**: Docker setup + testing (2h)

## Key Decisions

### 1. Start Simple
- Use FastAPI BackgroundTasks before Celery
- SQLite before PostgreSQL
- File storage before S3

### 2. Leverage Existing Code
- Build on existing FastAPI structure
- Use existing clinical routes as template
- Extend MoodPredictionPipeline

### 3. Reference Implementation Usage
- Copy authentication patterns from fastapi-fullstack when needed
- Use their Docker setup as template
- Adapt their testing patterns

### 4. Avoid Over-Engineering
- No microservices yet
- No Kubernetes yet
- No complex message queues yet

## Testing Strategy

### For Each Component
1. Write failing test first
2. Implement minimal solution
3. Refactor if needed
4. Run `make test`, `make lint`, `make type-check`

### Integration Tests
```python
# tests/integration/test_cli_integration.py
def test_cli_process_real_data():
    """Test CLI with real health data"""
    runner = CliRunner()
    result = runner.invoke(main, [
        'process',
        'health_auto_export/',
        '--output', 'test_output.csv'
    ])
    assert result.exit_code == 0
    assert Path('test_output.csv').exists()
```

## Success Metrics

1. **CLI Works**: `mood-detector process health_auto_export/`
2. **API Works**: Upload file → Get results
3. **Watcher Works**: Drop file → Auto-process
4. **Tests Pass**: All 526+ tests green
5. **Docker Works**: `docker-compose up`

## Next Steps After MVP

1. Add user authentication (from fastapi-fullstack)
2. Add FHIR compliance (from fhir-client)
3. Add Celery for scaling
4. Add PostgreSQL for production
5. Add monitoring/observability

---

This plan focuses on getting a working application TODAY while maintaining quality and preparing for future growth.