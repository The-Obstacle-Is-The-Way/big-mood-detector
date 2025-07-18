# Health Check Endpoints

## GET /health

Basic health check endpoint for container orchestration.

### Response

```json
{
  "status": "healthy",
  "service": "big-mood-detector",
  "version": "0.1.0"
}
```

### Status Codes

- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

## GET /healthz

Kubernetes-style health check endpoint.

### Response

```json
{
  "status": "ok"
}
```

### Usage Example

```bash
# Check service health
curl http://localhost:8000/health

# Kubernetes probe
curl http://localhost:8000/healthz
```

## Metrics Endpoint

### GET /metrics

Prometheus-compatible metrics endpoint.

### Response

```text
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health",status_code="200"} 42.0

# HELP model_loaded Model load status
# TYPE model_loaded gauge
model_loaded{model_name="xgboost_depression"} 1.0
model_loaded{model_name="pat"} 1.0
```

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'big-mood-detector'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```