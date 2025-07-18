# Clinical Interpretation API

The clinical API provides DSM-5 compliant interpretation of mood assessments and digital biomarkers.

## POST /api/v1/clinical/interpret/depression

Interpret depression assessment scores and biomarkers.

### Request Body

```json
{
  "phq9_score": 18,
  "sleep_hours": 10.5,
  "activity_steps": 1200,
  "suicidal_ideation": false
}
```

### Response

```json
{
  "risk_level": "high",
  "episode_type": "major_depressive",
  "confidence": 0.87,
  "clinical_summary": "PHQ-9 score of 18 indicates moderately severe depression. Hypersomnia (10.5 hours) and low activity (1200 steps) support major depressive episode.",
  "recommendations": [
    "Immediate clinical evaluation recommended",
    "Monitor sleep patterns - hypersomnia present",
    "Encourage gradual increase in physical activity"
  ],
  "dsm5_criteria_met": true,
  "supporting_evidence": [
    "PHQ-9 score > 15 (moderately severe range)",
    "Sleep duration > 9 hours (hypersomnia)",
    "Activity level < 3000 steps (psychomotor retardation)"
  ]
}
```

## POST /api/v1/clinical/interpret/mania

Interpret mania/hypomania assessment scores and biomarkers.

### Request Body

```json
{
  "asrm_score": 12,
  "sleep_hours": 3.5,
  "activity_steps": 18000,
  "psychotic_features": false
}
```

### Response

```json
{
  "risk_level": "high",
  "episode_type": "hypomanic",
  "confidence": 0.91,
  "clinical_summary": "ASRM score of 12 with decreased sleep (3.5 hours) and increased activity (18000 steps) indicates hypomanic episode.",
  "recommendations": [
    "Clinical evaluation within 24-48 hours",
    "Monitor for progression to mania",
    "Sleep hygiene intervention needed"
  ],
  "dsm5_criteria_met": true,
  "supporting_evidence": [
    "ASRM score > 5 (clinically significant)",
    "Sleep duration < 4 hours",
    "Activity level > 15000 steps (hyperactivity)"
  ]
}
```

## POST /api/v1/clinical/interpret/mixed

Interpret mixed state features based on DSM-5 criteria.

### Request Body

```json
{
  "phq9_score": 15,
  "asrm_score": 8,
  "sleep_hours": 5.0,
  "activity_steps": 12000,
  "racing_thoughts": true,
  "psychomotor_agitation": true,
  "decreased_need_for_sleep": true
}
```

### Response

```json
{
  "risk_level": "high",
  "episode_type": "mixed_features",
  "confidence": 0.85,
  "clinical_summary": "Concurrent depressive (PHQ-9: 15) and manic symptoms (ASRM: 8) with mixed features specifier.",
  "recommendations": [
    "Urgent psychiatric evaluation needed",
    "Mixed states carry increased suicide risk",
    "Mood stabilizer may be indicated"
  ],
  "dsm5_criteria_met": true,
  "mixed_features_present": [
    "racing_thoughts",
    "psychomotor_agitation",
    "decreased_need_for_sleep"
  ]
}
```

## POST /api/v1/clinical/evaluate/duration

Evaluate if episode duration meets DSM-5 criteria.

### Request Body

```json
{
  "episode_type": "major_depressive",
  "symptom_days": 18,
  "hospitalization": false
}
```

### Response

```json
{
  "meets_duration_criteria": true,
  "minimum_days_required": 14,
  "actual_days": 18,
  "clinical_note": "Major depressive episode requires 14+ days of symptoms. Current duration of 18 days meets DSM-5 criteria."
}
```

## POST /api/v1/clinical/interpret/biomarkers

Interpret digital biomarkers for mood episode risk.

### Request Body

```json
{
  "sleep": {
    "mean_duration": 9.2,
    "std_duration": 2.1,
    "efficiency": 0.72
  },
  "activity": {
    "mean_steps": 3200,
    "std_steps": 1800,
    "sedentary_hours": 18.5
  },
  "circadian": {
    "interdaily_stability": 0.42,
    "intradaily_variability": 1.3,
    "relative_amplitude": 0.55
  }
}
```

### Response

```json
{
  "mood_risk": {
    "depression": 0.78,
    "mania": 0.15,
    "overall": "high"
  },
  "risk_factors": [
    "Low interdaily stability (0.42) - irregular sleep-wake patterns",
    "High intradaily variability (1.3) - fragmented activity",
    "Extended sedentary time (18.5 hours)",
    "High sleep duration variability (STD: 2.1 hours)"
  ],
  "clinical_notes": "Digital biomarkers suggest depressive episode with significant circadian disruption. Low activity and irregular patterns support clinical assessment.",
  "recommendations": [
    "Establish regular sleep-wake schedule",
    "Gradually increase daytime activity",
    "Consider light therapy for circadian regulation",
    "Clinical mood assessment recommended"
  ]
}
```

## GET /api/v1/clinical/thresholds

Get clinical thresholds used for interpretation.

### Response

```json
{
  "depression": {
    "phq9_minimal": [0, 4],
    "phq9_mild": [5, 9],
    "phq9_moderate": [10, 14],
    "phq9_moderately_severe": [15, 19],
    "phq9_severe": [20, 27]
  },
  "mania": {
    "asrm_threshold": 5,
    "ymrs_hypomania": 12,
    "ymrs_mania": 20
  },
  "biomarkers": {
    "sleep_depression_min": 9,
    "sleep_mania_max": 4,
    "activity_depression_max": 3000,
    "activity_mania_min": 15000
  },
  "dsm5_duration": {
    "major_depressive": 14,
    "manic": 7,
    "hypomanic": 4
  }
}
```

## Clinical Guidelines

All interpretations follow:
- **DSM-5-TR** criteria for mood episodes
- **PHQ-9** validated scoring for depression
- **ASRM/YMRS** validated scoring for mania/hypomania
- Evidence-based digital biomarker thresholds from peer-reviewed research

## Risk Levels

- **Low**: Minimal symptoms, no immediate concern
- **Moderate**: Clinical symptoms present, monitoring recommended
- **High**: Significant symptoms, clinical evaluation needed
- **Critical**: Severe symptoms or safety concerns, urgent care required

## Important Notes

⚠️ **Medical Disclaimer**: This API provides clinical decision support but does not replace professional medical judgment. All recommendations should be reviewed by qualified healthcare providers.

🔒 **Privacy**: No patient data is stored. All processing happens in-memory and responses contain no identifying information.