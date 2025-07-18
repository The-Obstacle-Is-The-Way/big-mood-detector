#!/usr/bin/env python3
"""Test the API ensemble endpoint."""

import requests
import json

# Test data
test_features = {
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

# Check model status first
print("Checking model status...")
response = requests.get("http://localhost:8000/api/v1/predictions/status")
if response.status_code == 200:
    status = response.json()
    print(f"✓ XGBoost available: {status['xgboost_available']}")
    print(f"✓ PAT available: {status['pat_available']}")
    print(f"✓ Ensemble available: {status['ensemble_available']}")
    if status.get('pat_info'):
        print(f"  PAT model: {status['pat_info']['model_size']} ({status['pat_info']['parameters']} params)")
    if status.get('ensemble_config'):
        print(f"  Weights: XGBoost={status['ensemble_config']['xgboost_weight']}, PAT={status['ensemble_config']['pat_weight']}")
else:
    print(f"✗ Status check failed: {response.status_code}")
    print(response.text)

print("\nTesting ensemble prediction...")
response = requests.post(
    "http://localhost:8000/api/v1/predictions/predict/ensemble",
    json=test_features
)

if response.status_code == 200:
    result = response.json()
    print("✓ Ensemble prediction successful!")
    print(f"\nModels used: {', '.join(result['models_used'])}")
    print(f"\nEnsemble prediction:")
    print(f"  Depression risk: {result['ensemble_prediction']['depression_risk']:.1%}")
    print(f"  Hypomanic risk: {result['ensemble_prediction']['hypomanic_risk']:.1%}")
    print(f"  Manic risk: {result['ensemble_prediction']['manic_risk']:.1%}")
    print(f"  Confidence: {result['ensemble_prediction']['confidence']:.1%}")
    print(f"\nClinical summary: {result['clinical_summary']}")
    print(f"Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
else:
    print(f"✗ Prediction failed: {response.status_code}")
    print(response.text)