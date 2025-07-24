# 🐳 Docker Setup Guide - Big Mood Detector

Complete dockerization guide for cross-platform deployment and development.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector

# Build and run
docker compose up --build

# Or for GPU support
docker compose -f docker-compose.gpu.yml up --build
```

## 📋 Prerequisites

- Docker Desktop 4.0+ (Mac/Windows) or Docker Engine 20.10+ (Linux)
- 8GB RAM allocated to Docker
- For GPU: NVIDIA GPU with CUDA 12.0+ and nvidia-docker

## 🏗️ Docker Architecture

```
├── Dockerfile              # Production image
├── Dockerfile.dev          # Development image with tools
├── docker-compose.yml      # Standard CPU setup
├── docker-compose.gpu.yml  # GPU-enabled setup
└── .dockerignore          # Exclude unnecessary files
```

## 📦 Production Dockerfile

Create `Dockerfile.prod`:

```dockerfile
# Multi-stage build for minimal production image
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir --user -e ".[ml]"

# Production image
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application
WORKDIR /app
COPY --from=builder /build/src ./src
COPY model_weights/ ./model_weights/
COPY scripts/verify_setup.py ./scripts/

# Create required directories
RUN mkdir -p data/input data/cache data/baselines logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import big_mood_detector; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "big_mood_detector", "serve", "--host", "0.0.0.0"]
```

## 🧑‍💻 Development Dockerfile

Create `Dockerfile.dev`:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .

# Install all dependencies including dev
RUN pip install --no-cache-dir -e ".[dev,ml,monitoring]"

# Copy source code
COPY . .

# Install pre-commit hooks
RUN git init && pre-commit install || true

# Create directories
RUN mkdir -p data/input data/cache data/baselines logs

# Development command
CMD ["bash"]
```

## 🐋 Docker Compose Configuration

### CPU Version (`docker-compose.yml`):

```yaml
version: '3.8'

services:
  big-mood:
    build:
      context: .
      dockerfile: Dockerfile.prod
    image: big-mood-detector:latest
    container_name: big-mood-detector
    
    ports:
      - "8000:8000"
    
    volumes:
      # Data persistence
      - ./data:/app/data
      - ./model_weights:/app/model_weights
      - ./logs:/app/logs
      
      # For development - hot reload
      - ./src:/app/src:ro
    
    environment:
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - PYTHONUNBUFFERED=1
      - BIG_MOOD_ENV=docker
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    restart: unless-stopped

  # Development container
  big-mood-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    image: big-mood-detector:dev
    container_name: big-mood-dev
    
    volumes:
      - .:/app
      - /app/.venv  # Exclude venv from mount
    
    environment:
      - PYTHONPATH=/app
    
    command: bash
    profiles: ["dev"]

# Volumes for data persistence
volumes:
  model_weights:
  data:
  logs:
```

### GPU Version (`docker-compose.gpu.yml`):

```yaml
version: '3.8'

services:
  big-mood-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    image: big-mood-detector:gpu
    
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - TORCH_DEVICE=cuda
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # Inherit other settings from main compose
    extends:
      file: docker-compose.yml
      service: big-mood
```

## 🎯 GPU Dockerfile

Create `Dockerfile.gpu`:

```dockerfile
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[ml]"

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/input data/cache data/baselines logs

CMD ["python3", "-m", "big_mood_detector", "serve", "--host", "0.0.0.0"]
```

## 📁 Data Volume Management

### Prepare data before first run:

```bash
# Create directory structure
mkdir -p data/input/{apple_export,health_auto_export}
mkdir -p model_weights/{xgboost/converted,pat/pretrained}

# Copy your data
cp ~/Downloads/export.xml data/input/apple_export/
cp ~/Downloads/XGBoost_*.json model_weights/xgboost/converted/
```

### Use Docker volumes for persistence:

```yaml
# docker-compose.override.yml (for local overrides)
version: '3.8'

services:
  big-mood:
    volumes:
      # Use absolute paths for large datasets
      - /mnt/bigdata/nhanes:/app/data/nhanes:ro
      - ${HOME}/health_exports:/app/data/input:ro
```

## 🔧 Common Docker Commands

```bash
# Build and start
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f big-mood

# Run CLI commands
docker compose run --rm big-mood python -m big_mood_detector process /app/data/input/

# Interactive shell
docker compose run --rm big-mood bash

# Run tests
docker compose run --rm big-mood-dev make test

# Clean up
docker compose down -v  # -v removes volumes
```

## 🌐 Environment Variables

Create `.env` file:

```bash
# .env
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Model paths (optional overrides)
BIG_MOOD_PAT_WEIGHTS_DIR=/app/model_weights/pat/pretrained
BIGMOOD_DATA_DIR=/app/data

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# For GPU
CUDA_VISIBLE_DEVICES=0
TORCH_DEVICE=cuda
```

## 🚪 Entry Points

Create `docker-entrypoint.sh`:

```bash
#!/bin/bash
set -e

# Verify setup on first run
if [ ! -f "/app/.docker-initialized" ]; then
    echo "🔍 First run - verifying setup..."
    python scripts/verify_setup.py || {
        echo "❌ Setup verification failed!"
        echo "📖 See DATA_SETUP_GUIDE.md for instructions"
        exit 1
    }
    touch /app/.docker-initialized
fi

# Run command
exec "$@"
```

## 🏃 Running Different Services

```bash
# API server
docker compose run --rm -p 8000:8000 big-mood serve

# Process data
docker compose run --rm big-mood process /app/data/input/

# Generate predictions
docker compose run --rm big-mood predict /app/data/input/ --report

# Train PAT model
docker compose run --rm big-mood-gpu python scripts/pat_training/train_pat_canonical.py
```

## 🔍 Debugging Docker Issues

```bash
# Check if data is mounted correctly
docker compose run --rm big-mood ls -la /app/data/

# Verify model weights
docker compose run --rm big-mood python scripts/verify_setup.py

# Test imports
docker compose run --rm big-mood python -c "import big_mood_detector; print('OK')"

# GPU check
docker compose run --rm big-mood-gpu nvidia-smi
docker compose run --rm big-mood-gpu python -c "import torch; print(torch.cuda.is_available())"
```

## 🔐 Security Best Practices

1. **Never include sensitive data in images**
   ```dockerfile
   # Bad
   COPY data/ /app/data/
   
   # Good - use volumes
   VOLUME ["/app/data"]
   ```

2. **Use .dockerignore**
   ```
   # .dockerignore
   data/
   *.xml
   *.csv
   .git/
   .env
   __pycache__/
   *.pyc
   ```

3. **Run as non-root user**
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

## 🚀 Deployment Examples

### Local Development
```bash
docker compose --profile dev up
```

### Production API
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Batch Processing
```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/model_weights:/model_weights \
  big-mood-detector:latest \
  process /data/input/
```

## 📊 Monitoring

Add Prometheus metrics:

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## ✅ Verification Checklist

- [ ] Docker Desktop/Engine installed
- [ ] Model weights downloaded to `model_weights/`
- [ ] Health data in `data/input/`
- [ ] `.env` file created
- [ ] Ports 8000 available
- [ ] For GPU: nvidia-docker installed

---

With this setup, you can develop on Mac and deploy to Windows/Linux seamlessly!