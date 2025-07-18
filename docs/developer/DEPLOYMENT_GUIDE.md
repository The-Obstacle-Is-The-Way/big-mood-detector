# 🚀 Big Mood Detector - Deployment Guide

## 📋 Deployment Options

1. **Local Development** - Single machine deployment
2. **Docker** - Containerized deployment
3. **Docker Compose** - Full stack with dependencies
4. **Kubernetes** - Scalable cloud deployment
5. **Serverless** - AWS Lambda / Google Cloud Functions

## 🏠 Local Deployment

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (optional, SQLite by default)
- Redis 7+ (optional, for background tasks)
- 8GB RAM minimum
- 10GB disk space

### Basic Setup

```bash
# Clone repository
git clone https://github.com/yourusername/big-mood-detector.git
cd big-mood-detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install production dependencies
pip install -e ".[prod]"

# Set environment variables
cp .env.example .env
# Edit .env with your settings

# Initialize database
alembic upgrade head

# Start the server
python src/big_mood_detector/main.py serve --host 0.0.0.0 --port 8000
```

### Production Configuration

Create `.env.production`:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_KEY=your-secure-api-key

# Database
DATABASE_URL=postgresql://user:password@localhost/bigmood

# Redis (for background tasks)
REDIS_URL=redis://localhost:6379

# Model Paths
MODEL_WEIGHTS_PATH=/app/model_weights
DATA_DIR=/app/data

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=https://your-domain.com

# Monitoring
LOG_LEVEL=INFO
SENTRY_DSN=https://your-sentry-dsn
```

### Systemd Service

Create `/etc/systemd/system/bigmood.service`:
```ini
[Unit]
Description=Big Mood Detector API
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=bigmood
Group=bigmood
WorkingDirectory=/opt/bigmood
Environment="PATH=/opt/bigmood/.venv/bin"
EnvironmentFile=/opt/bigmood/.env.production
ExecStart=/opt/bigmood/.venv/bin/python /opt/bigmood/src/big_mood_detector/main.py serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable bigmood
sudo systemctl start bigmood
sudo systemctl status bigmood
```

## 🐳 Docker Deployment

### Dockerfile (Already Included)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml setup.py ./
COPY src/big_mood_detector/__init__.py src/big_mood_detector/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[prod]"

# Copy application code
COPY . .

# Copy model weights
COPY model_weights/ ./model_weights/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["python", "src/big_mood_detector/main.py", "serve"]
```

### Build and Run

```bash
# Build image
docker build -t big-mood-detector:latest .

# Run container
docker run -d \
  --name bigmood-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_weights:/app/model_weights \
  --env-file .env.production \
  big-mood-detector:latest
```

## 🎼 Docker Compose Deployment

### docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    image: big-mood-detector:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://bigmood:password@postgres/bigmood
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./model_weights:/app/model_weights
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    build: .
    image: big-mood-detector:latest
    command: celery -A big_mood_detector.infrastructure.background.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://bigmood:password@postgres/bigmood
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./model_weights:/app/model_weights
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=bigmood
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=bigmood
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bigmood"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

### Nginx Configuration

Create `nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Scale workers
docker-compose up -d --scale worker=3
```

## ☸️ Kubernetes Deployment

### Kubernetes Manifests

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bigmood-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bigmood-api
  template:
    metadata:
      labels:
        app: bigmood-api
    spec:
      containers:
      - name: api
        image: your-registry/big-mood-detector:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: bigmood-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: bigmood-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: bigmood-api
spec:
  selector:
    app: bigmood-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Helm Chart

Create `helm/bigmood/values.yaml`:
```yaml
replicaCount: 3

image:
  repository: your-registry/big-mood-detector
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
  hosts:
    - host: api.bigmood.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: bigmood-tls
      hosts:
        - api.bigmood.example.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    username: bigmood
    database: bigmood

redis:
  enabled: true
  auth:
    enabled: false
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace bigmood

# Create secrets
kubectl create secret generic bigmood-secrets \
  --from-literal=database-url=postgresql://user:pass@postgres/bigmood \
  --from-literal=redis-url=redis://redis:6379 \
  -n bigmood

# Deploy with Helm
helm install bigmood ./helm/bigmood -n bigmood

# Or with kubectl
kubectl apply -f k8s/ -n bigmood

# Check deployment
kubectl get all -n bigmood
```

## ⚡ Serverless Deployment

### AWS Lambda

Create `serverless.yml`:
```yaml
service: bigmood-detector

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  environment:
    DATABASE_URL: ${env:DATABASE_URL}
    MODEL_BUCKET: ${self:custom.modelBucket}

functions:
  api:
    handler: lambda_handler.handler
    events:
      - httpApi: '*'
    timeout: 30
    memorySize: 3008
    layers:
      - arn:aws:lambda:${aws:region}:${aws:accountId}:layer:bigmood-models:1

  processor:
    handler: lambda_processor.handler
    events:
      - sqs:
          arn:
            Fn::GetAtt:
              - ProcessingQueue
              - Arn
    timeout: 900
    memorySize: 3008

resources:
  Resources:
    ProcessingQueue:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: bigmood-processing
        VisibilityTimeout: 960

custom:
  modelBucket: bigmood-models-${aws:accountId}
```

### Lambda Handler

Create `lambda_handler.py`:
```python
import json
from mangum import Mangum
from big_mood_detector.interfaces.api.main import app

# Create Lambda handler
handler = Mangum(app)
```

### Deploy to AWS

```bash
# Install serverless
npm install -g serverless

# Deploy
serverless deploy --stage production
```

## 📊 Monitoring and Logging

### Prometheus Metrics

Add to `docker-compose.yml`:
```yaml
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus_data:/prometheus
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  volumes:
    - grafana_data:/var/lib/grafana
```

### Application Metrics

```python
# src/big_mood_detector/infrastructure/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prediction_counter = Counter(
    'bigmood_predictions_total',
    'Total number of predictions',
    ['mood_type', 'risk_level']
)

processing_time = Histogram(
    'bigmood_processing_seconds',
    'Time spent processing health data'
)

active_jobs = Gauge(
    'bigmood_active_jobs',
    'Number of active processing jobs'
)
```

### Logging Configuration

```python
# src/big_mood_detector/infrastructure/logging/config.py
import logging
import json

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
            
        if hasattr(record, 'job_id'):
            log_obj['job_id'] = record.job_id
            
        return json.dumps(log_obj)
```

## 🔒 Security Hardening

### SSL/TLS Configuration

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# For production, use Let's Encrypt
certbot certonly --webroot -w /var/www/html -d your-domain.com
```

### Environment Security

```bash
# Secure environment variables
chmod 600 .env.production

# Use secrets management
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name bigmood/production \
  --secret-string file://.env.production

# Kubernetes secrets
kubectl create secret generic bigmood-env \
  --from-env-file=.env.production
```

### API Security

```python
# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

# CORS configuration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🔄 Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups/postgres

pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/bigmood_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
```

### Model Weights Backup

```bash
# Sync to S3
aws s3 sync model_weights/ s3://bigmood-backups/model_weights/

# Or use rclone for multiple clouds
rclone sync model_weights/ remote:bigmood-backups/model_weights/
```

## 🎯 Performance Tuning

### API Performance

```python
# Gunicorn configuration
bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
keepalive = 5
```

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_episodes_user_date ON episodes(user_id, start_date);
CREATE INDEX idx_predictions_job_id ON predictions(job_id);
CREATE INDEX idx_features_date_range ON features(user_id, date);

-- Vacuum and analyze
VACUUM ANALYZE;
```

### Caching Strategy

```python
# Redis caching
from functools import lru_cache
import redis

redis_client = redis.from_url(settings.REDIS_URL)

@lru_cache(maxsize=1000)
def get_cached_features(user_id: str, date_range: tuple):
    key = f"features:{user_id}:{date_range[0]}:{date_range[1]}"
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None
```

## ✅ Deployment Checklist

- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] Model weights in place
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backups scheduled
- [ ] Health checks passing
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated

---

For troubleshooting, see [Operations Guide](./OPERATIONS_GUIDE.md)