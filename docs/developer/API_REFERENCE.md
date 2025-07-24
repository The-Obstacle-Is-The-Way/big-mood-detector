# 🌐 Big Mood Detector - API Reference

## 📋 Overview

The Big Mood Detector API provides RESTful endpoints for health data processing, mood predictions, and clinical insights.

**Base URL**: `http://localhost:8000`  
**API Version**: `v1`  
**Documentation**: Available at `/docs` (Swagger UI) and `/redoc` (ReDoc)

## 🔐 Authentication

Currently supports API key authentication. JWT support coming soon.

```bash
# Include API key in header
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/health
```

## 📍 Endpoints

### Health Check

#### `GET /health`
Check if the API is running and healthy.

**Response**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": true,
  "timestamp": "2025-07-18T10:30:00Z"
}
```

---

### File Upload

#### `POST /api/v1/upload/file`
Upload a single health data file for processing.

**Request**
- Content-Type: `multipart/form-data`
- Body: File upload

**Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | file | Yes | Health data file (.json or .xml) |
| user_id | string | No | User identifier for tracking |

**Example**
```bash
curl -X POST http://localhost:8000/api/v1/upload/file \
  -F "file=@sleep_data.json" \
  -F "user_id=patient_123"
```

**Response**
```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "sleep_data.json",
  "file_type": "json",
  "size_bytes": 125840,
  "status": "uploaded",
  "created_at": "2025-07-18T10:30:00Z"
}
```

#### `POST /api/v1/upload/batch`
Upload multiple files at once.

**Request**
- Content-Type: `multipart/form-data`
- Body: Multiple file uploads

**Example**
```bash
curl -X POST http://localhost:8000/api/v1/upload/batch \
  -F "files=@sleep.json" \
  -F "files=@activity.json" \
  -F "files=@heart_rate.json"
```

**Response**
```json
{
  "batch_id": "660e8400-e29b-41d4-a716-446655440000",
  "files": [
    {
      "upload_id": "550e8400-e29b-41d4-a716-446655440001",
      "filename": "sleep.json",
      "status": "uploaded"
    },
    {
      "upload_id": "550e8400-e29b-41d4-a716-446655440002",
      "filename": "activity.json",
      "status": "uploaded"
    }
  ],
  "total_files": 3,
  "total_size_bytes": 384920
}
```

---

### Processing

#### `POST /api/v1/process/start`
Start processing uploaded health data.

**Request Body**
```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "options": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
    "ensemble": true,
    "generate_report": true,
    "user_id": "patient_123"
  }
}
```

**Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| upload_id | string | Yes* | ID from upload endpoint |
| batch_id | string | Yes* | ID from batch upload |
| options.start_date | date | No | Start date for analysis |
| options.end_date | date | No | End date for analysis |
| options.ensemble | boolean | No | Use temporal separation (default: true) |
| options.generate_report | boolean | No | Generate clinical report |
| options.user_id | string | No | User ID for personalized predictions |

*Either upload_id or batch_id required

**Response**
```json
{
  "job_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "estimated_time_seconds": 30,
  "created_at": "2025-07-18T10:31:00Z"
}
```

#### `GET /api/v1/process/status/{job_id}`
Check the status of a processing job.

**Response**
```json
{
  "job_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.75,
  "current_step": "Extracting features",
  "started_at": "2025-07-18T10:31:00Z",
  "updated_at": "2025-07-18T10:31:15Z"
}
```

**Status Values**
- `queued`: Job is waiting to be processed
- `processing`: Job is currently being processed
- `completed`: Job finished successfully
- `failed`: Job encountered an error
- `cancelled`: Job was cancelled

#### `POST /api/v1/process/cancel/{job_id}`
Cancel a processing job.

**Response**
```json
{
  "job_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "cancelled_at": "2025-07-18T10:32:00Z"
}
```

---

### Predictions

#### `POST /api/v1/predictions/predict`
Generate mood predictions from pre-computed features.

**Request Body**
```json
{
  "sleep_duration": 7.5,
  "sleep_efficiency": 0.85,
  "sleep_timing_variance": 30.0,
  "daily_steps": 8000,
  "activity_variance": 150.0,
  "sedentary_hours": 8.0,
  "interdaily_stability": 0.75,
  "intradaily_variability": 0.45,
  "relative_amplitude": 0.82,
  "resting_hr": 65.0,
  "hrv_rmssd": 35.0
}
```

**Response**
```json
{
  "depression_risk": 0.32,
  "hypomanic_risk": 0.15,
  "manic_risk": 0.08,
  "confidence": 0.87,
  "risk_level": "low",
  "interpretation": "Low mood episode risk"
}
```

#### `POST /api/v1/predictions/predict/ensemble`
Generate temporal predictions using both XGBoost and PAT models.

**Note**: This endpoint requires PyTorch to be installed for PAT model support. Without PyTorch, it will return a 501 status code.

**Request Body**
Same as `/predict` endpoint.

**Response**
```json
{
  "xgboost_prediction": {
    "depression_risk": 0.32,
    "hypomanic_risk": 0.15,
    "manic_risk": 0.08,
    "confidence": 0.85
  },
  "pat_prediction": {
    "depression_risk": 0.28,
    "hypomanic_risk": 0.18,
    "manic_risk": 0.06,
    "confidence": 0.89
  },
  "ensemble_prediction": {
    "depression_risk": 0.30,
    "hypomanic_risk": 0.16,
    "manic_risk": 0.07,
    "confidence": 0.88
  },
  "models_used": ["xgboost", "pat_enhanced"],
  "confidence_scores": {
    "xgboost": 0.85,
    "pat_enhanced": 0.89,
    "ensemble": 0.88
  },
  "clinical_summary": "Low risk - maintain healthy lifestyle",
  "recommendations": [
    "Continue current habits",
    "Maintain consistent sleep-wake schedule"
  ]
}
```

**Ensemble Weighting**
- XGBoost: 60% weight
- PAT: 40% weight

**Performance Targets**
- P99 latency < 200ms (with PAT)
- P99 latency < 100ms (XGBoost only)

#### `GET /api/v1/predictions/status`
Check available prediction models and their status.

**Response**
```json
{
  "xgboost_available": true,
  "pat_available": true,
  "ensemble_available": true,
  "models_loaded": ["depression", "hypomania", "mania"],
  "model_info": {
    "depression": {
      "type": "XGBoost",
      "version": "1.0.0",
      "features": 36,
      "trained_on": "Seoul National Hospital Dataset"
    }
  },
  "pat_info": {
    "model_size": "medium",
    "patch_size": 18,
    "num_patches": 560,
    "embed_dim": 96,
    "encoder_layers": 2,
    "encoder_heads": 12,
    "parameters": 1000000,
    "is_loaded": true
  },
  "ensemble_config": {
    "xgboost_weight": 0.6,
    "pat_weight": 0.4
  }
}
```

---

### Results

#### `GET /api/v1/results/{job_id}`
Retrieve results from a completed processing job.

**Response**
```json
{
  "job_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "results": {
    "predictions": {
      "depression_risk": 0.65,
      "mania_risk": 0.12,
      "hypomania_risk": 0.18,
      "confidence": 0.89
    },
    "features": {
      "sleep_duration_mean": 6.2,
      "sleep_efficiency": 0.85,
      "activity_level_mean": 5800,
      "circadian_phase": -1.2
    },
    "clinical_flags": [
      {
        "type": "sleep_duration_low",
        "severity": "moderate",
        "message": "Average sleep duration below recommended range"
      }
    ],
    "report_url": "/api/v1/results/770e8400-e29b-41d4-a716-446655440000/report"
  },
  "completed_at": "2025-07-18T10:31:30Z"
}
```

#### `GET /api/v1/results/{job_id}/report`
Get the clinical report for a job (if generated).

**Response**
```text
Big Mood Detector - Clinical Report
Generated: 2025-07-18 10:31:30

Patient ID: patient_123
Period: 2024-01-01 to 2024-03-31
Days analyzed: 90

RISK SUMMARY:
- Depression Risk: MODERATE (0.65)
- Mania Risk: LOW (0.12)
- Hypomania Risk: LOW (0.18)

[... full report content ...]
```

#### `GET /api/v1/results/latest`
Get the most recent results for the authenticated user.

**Query Parameters**
| Name | Type | Description |
|------|------|-------------|
| user_id | string | Filter by user ID |
| limit | integer | Number of results (default: 10) |

---

### Features Extraction

#### `POST /api/v1/features/extract`
Extract features from health data without predictions.

**Request Body**
```json
{
  "upload_id": "550e8400-e29b-41d4-a716-446655440000",
  "feature_sets": ["sleep", "activity", "circadian", "clinical"]
}
```

**Response**
```json
{
  "features": {
    "sleep": {
      "duration_mean": 6.2,
      "duration_std": 1.1,
      "efficiency_mean": 0.85,
      "fragmentation_index": 0.12,
      "window_count_mean": 1.2
    },
    "activity": {
      "steps_mean": 5800,
      "steps_std": 2100,
      "sedentary_minutes": 480,
      "active_minutes": 45
    },
    "circadian": {
      "phase": -1.2,
      "amplitude": 0.8,
      "interdaily_stability": 0.75,
      "intradaily_variability": 0.45
    },
    "clinical": {
      "short_sleep_days": 12,
      "long_sleep_days": 3,
      "irregular_schedule_score": 0.3
    }
  }
}
```

---

### Clinical Endpoints

#### `POST /api/v1/clinical/assess`
Perform clinical assessment based on features.

**Request Body**
```json
{
  "features": { ... },
  "patient_info": {
    "age": 35,
    "gender": "female",
    "medication": ["lithium", "quetiapine"],
    "diagnosis_date": "2020-05-15"
  }
}
```

**Response**
```json
{
  "assessment": {
    "current_state": "euthymic_with_risk",
    "risk_factors": [
      "sleep_disruption",
      "circadian_misalignment"
    ],
    "recommendations": [
      "Monitor sleep patterns closely",
      "Consider sleep hygiene intervention",
      "Schedule follow-up in 2 weeks"
    ],
    "alert_level": "moderate"
  }
}
```

---

### Labels Management

#### `POST /api/v1/labels/episode`
Create a new episode label.

**Request Body**
```json
{
  "start_date": "2024-01-15",
  "end_date": "2024-01-29",
  "mood_type": "depressive",
  "severity": "moderate",
  "confidence": 0.8,
  "rater_id": "clinician_1",
  "notes": "Patient reported low mood and anhedonia"
}
```

#### `GET /api/v1/labels/episodes`
List all episode labels.

**Query Parameters**
| Name | Type | Description |
|------|------|-------------|
| user_id | string | Filter by user |
| mood_type | string | Filter by mood type |
| start_date | date | Episodes after this date |
| end_date | date | Episodes before this date |

---

### Webhooks

#### `POST /api/v1/webhooks/configure`
Configure webhook notifications.

**Request Body**
```json
{
  "url": "https://your-server.com/mood-alerts",
  "events": ["high_risk_detected", "processing_complete"],
  "filters": {
    "min_risk_level": 0.7,
    "mood_types": ["mania", "depression"]
  },
  "secret": "your-webhook-secret"
}
```

---

## 🔄 WebSocket Endpoints

### Real-time Updates

#### `WS /ws/jobs/{job_id}`
Subscribe to real-time updates for a processing job.

**Message Format**
```json
{
  "type": "progress",
  "data": {
    "progress": 0.75,
    "current_step": "Extracting features"
  }
}
```

---

## 📊 Response Formats

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "req_123",
    "timestamp": "2025-07-18T10:30:00Z"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "File must be JSON or XML format",
    "details": {
      "received_format": "csv",
      "allowed_formats": ["json", "xml"]
    }
  },
  "meta": {
    "request_id": "req_123",
    "timestamp": "2025-07-18T10:30:00Z"
  }
}
```

## 📜 Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_FILE_FORMAT | 400 | Unsupported file format |
| FILE_TOO_LARGE | 400 | File exceeds size limit |
| MISSING_REQUIRED_FIELD | 400 | Required field not provided |
| INVALID_DATE_RANGE | 400 | Invalid or impossible date range |
| UPLOAD_NOT_FOUND | 404 | Upload ID not found |
| JOB_NOT_FOUND | 404 | Job ID not found |
| INSUFFICIENT_DATA | 422 | Not enough data for analysis |
| MODEL_ERROR | 500 | ML model prediction failed |
| PROCESSING_ERROR | 500 | General processing error |

## 🚀 Rate Limits

| Endpoint | Rate Limit | Window |
|----------|------------|---------|
| Upload | 100 requests | 1 hour |
| Process | 50 requests | 1 hour |
| Results | 500 requests | 1 hour |
| All others | 1000 requests | 1 hour |

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## 📝 Examples

### Complete Workflow Example

```python
import requests
import time

# 1. Upload file
upload_response = requests.post(
    "http://localhost:8000/api/v1/upload/file",
    files={"file": open("health_data.json", "rb")}
)
upload_id = upload_response.json()["upload_id"]

# 2. Start processing
process_response = requests.post(
    "http://localhost:8000/api/v1/process/start",
    json={
        "upload_id": upload_id,
        "options": {
            "ensemble": True,
            "generate_report": True
        }
    }
)
job_id = process_response.json()["job_id"]

# 3. Poll for completion
while True:
    status_response = requests.get(
        f"http://localhost:8000/api/v1/process/status/{job_id}"
    )
    status = status_response.json()["status"]
    
    if status == "completed":
        break
    elif status == "failed":
        raise Exception("Processing failed")
    
    time.sleep(2)

# 4. Get results
results_response = requests.get(
    f"http://localhost:8000/api/v1/results/{job_id}"
)
predictions = results_response.json()["results"]["predictions"]

print(f"Depression risk: {predictions['depression_risk']}")
print(f"Mania risk: {predictions['mania_risk']}")
```

### Batch Processing Example

```python
# Upload multiple files
files = [
    ("files", open("sleep.json", "rb")),
    ("files", open("activity.json", "rb")),
    ("files", open("heart_rate.json", "rb"))
]

batch_response = requests.post(
    "http://localhost:8000/api/v1/upload/batch",
    files=files
)

batch_id = batch_response.json()["batch_id"]

# Process batch
process_response = requests.post(
    "http://localhost:8000/api/v1/process/start",
    json={
        "batch_id": batch_id,
        "options": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31"
        }
    }
)
```

---

## 🔗 SDK Support

Official SDKs coming soon:
- Python SDK
- JavaScript/TypeScript SDK
- Go SDK

For now, use standard HTTP clients or generate from OpenAPI spec at `/openapi.json`.