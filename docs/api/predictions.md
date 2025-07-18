# Predictions API

## POST /api/v1/predict

Get mood episode risk predictions from processed health data.

### Request Body

```json
{
  "features": [8.2, 1.1, 0.89, ...],  // 36-element feature vector
  "metadata": {
    "user_id": "optional-user-id",
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

### Response

```json
{
  "predictions": {
    "depression": {
      "risk_score": 0.72,
      "risk_level": "high",
      "confidence": 0.85
    },
    "hypomanic": {
      "risk_score": 0.23,
      "risk_level": "low", 
      "confidence": 0.91
    },
    "manic": {
      "risk_score": 0.15,
      "risk_level": "low",
      "confidence": 0.88
    }
  },
  "metadata": {
    "model_version": "xgboost-v1.0",
    "prediction_timestamp": "2024-01-15T12:00:05Z",
    "processing_time_ms": 23
  }
}
```

## POST /api/v1/predict/ensemble

Get ensemble predictions combining XGBoost and PAT models.

### Request Body

```json
{
  "features": [8.2, 1.1, 0.89, ...],  // 36-element feature vector
  "pat_sequence": {
    "end_date": "2024-01-15",
    "activity_values": [0, 0, 0, 150, 200, ...],  // 10080 values (7 days × 1440 min/day)
    "missing_days": 0,
    "data_quality_score": 0.95
  }
}
```

### Response

```json
{
  "predictions": {
    "xgboost": {
      "depression": 0.72,
      "hypomanic": 0.23,
      "manic": 0.15
    },
    "pat": {
      "depression": 0.68,
      "hypomanic": 0.21,
      "manic": 0.11
    },
    "ensemble": {
      "depression": {
        "risk_score": 0.70,
        "risk_level": "high",
        "confidence": 0.87
      },
      "hypomanic": {
        "risk_score": 0.22,
        "risk_level": "low",
        "confidence": 0.92
      },
      "manic": {
        "risk_score": 0.13,
        "risk_level": "low",
        "confidence": 0.89
      }
    }
  },
  "weights": {
    "xgboost": 0.6,
    "pat": 0.4
  },
  "metadata": {
    "models_used": ["xgboost-v1.0", "pat-v1.0"],
    "ensemble_method": "weighted_average",
    "processing_time_ms": 87
  }
}
```

### Status Codes

- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid feature vector
- `422 Unprocessable Entity` - Validation failed
- `501 Not Implemented` - PAT model not available (TensorFlow not installed)

## Risk Levels

Risk scores are mapped to levels based on clinical thresholds:

- **Low**: < 0.3
- **Moderate**: 0.3 - 0.5  
- **High**: > 0.5

These thresholds can be adjusted in the clinical configuration.