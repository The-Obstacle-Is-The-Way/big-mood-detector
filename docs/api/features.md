# Feature Extraction API

## POST /api/v1/features/extract

Extract clinical features from processed health data.

### Request Body

```json
{
  "sleep_records": [
    {
      "start_time": "2024-01-15T22:00:00Z",
      "end_time": "2024-01-16T06:30:00Z",
      "sleep_duration_hours": 8.5,
      "deep_sleep_ratio": 0.25,
      "rem_sleep_ratio": 0.20,
      "awake_ratio": 0.10
    }
  ],
  "activity_records": [
    {
      "date": "2024-01-15",
      "hour": 14,
      "steps": 523,
      "active_calories": 45.2,
      "heart_rate_avg": 72
    }
  ]
}
```

### Response

```json
{
  "features": {
    "basic_sleep": {
      "mean_sleep_duration": 8.2,
      "std_sleep_duration": 1.1,
      "mean_sleep_efficiency": 0.89
    },
    "circadian": {
      "interdaily_stability": 0.72,
      "intradaily_variability": 0.45,
      "relative_amplitude": 0.81,
      "L5": 42.3,
      "M10": 285.7
    },
    "activity": {
      "mean_daily_steps": 8234,
      "activity_acrophase": 14.5,
      "sedentary_hours": 9.2
    }
  },
  "feature_vector": [8.2, 1.1, 0.89, ...],
  "metadata": {
    "feature_count": 36,
    "processing_time_ms": 45,
    "data_quality_score": 0.92
  }
}
```

### Status Codes

- `200 OK` - Features extracted successfully
- `400 Bad Request` - Invalid input data
- `422 Unprocessable Entity` - Data validation failed

## Feature Definitions

### Basic Sleep Features
- `mean_sleep_duration` - Average sleep duration in hours
- `std_sleep_duration` - Standard deviation of sleep duration
- `mean_sleep_efficiency` - Average sleep efficiency (0-1)

### Circadian Rhythm Features
- `interdaily_stability` (IS) - Regularity of sleep-wake patterns (0-1)
- `intradaily_variability` (IV) - Fragmentation of activity (0-2)
- `relative_amplitude` (RA) - Difference between active/rest periods (0-1)
- `L5` - Average activity during 5 least active hours
- `M10` - Average activity during 10 most active hours

### Activity Features
- `mean_daily_steps` - Average daily step count
- `activity_acrophase` - Peak activity time (0-24 hours)
- `sedentary_hours` - Hours with minimal activity