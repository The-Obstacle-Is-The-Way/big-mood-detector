# 🔬 Big Mood Detector - Advanced Usage Guide

This guide covers advanced features and workflows for power users and clinicians.

## 📊 Advanced Data Processing

### Batch Processing Multiple Users

```bash
# Process multiple user directories
for user in data/users/*; do
    echo "Processing $user..."
    python src/big_mood_detector/main.py process "$user" \
        -o "output/$(basename $user)_features.json" \
        -v
done
```

### Custom Date Ranges and Filtering

```bash
# Process only last 30 days
python src/big_mood_detector/main.py process data/health_auto_export/ \
    --start-date $(date -d "30 days ago" +%Y-%m-%d) \
    --end-date $(date +%Y-%m-%d) \
    -o recent_features.json

# Process specific months
python src/big_mood_detector/main.py process data/health_auto_export/ \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    -v
```

### Combining Multiple Data Sources

```python
# combine_sources.py
from big_mood_detector.infrastructure.parsers.parser_factory import UnifiedHealthDataParser

parser = UnifiedHealthDataParser()

# Add JSON sources
parser.add_json_source("data/health_auto_export/Sleep Analysis.json", "sleep")
parser.add_json_source("data/health_auto_export/Heart Rate.json", "heart_rate")
parser.add_json_source("data/health_auto_export/Step Count.json", "activity")

# Add XML export
parser.add_xml_export("data/apple_export/export.xml")

# Get combined records
all_records = parser.get_all_records()
print(f"Total records: {len(all_records['sleep']) + len(all_records['activity']) + len(all_records['heart_rate'])}")
```

## 🧠 Advanced Prediction Features

### Ensemble Model Configuration

```bash
# Use only XGBoost models
python src/big_mood_detector/main.py predict data/health_auto_export/ \
    --no-ensemble \
    -o xgboost_only.json

# Use PAT transformer with custom weights
python src/big_mood_detector/main.py predict data/health_auto_export/ \
    --ensemble \
    --model-dir /path/to/custom/models/ \
    -o custom_predictions.json
```

### Personalized Predictions

```bash
# Use personalized model for user
python src/big_mood_detector/main.py predict data/health_auto_export/ \
    --user-id "patient_123" \
    --report \
    -o personalized_report.txt
```

### Continuous Monitoring Script

```python
# continuous_monitor.py
import time
import subprocess
from datetime import datetime, timedelta

def monitor_mood(data_dir, user_id):
    """Run predictions daily and alert on changes"""
    
    last_risk = {"depression": 0, "mania": 0, "hypomania": 0}
    
    while True:
        # Run prediction
        result = subprocess.run([
            "python", "src/big_mood_detector/main.py", "predict",
            data_dir,
            "--user-id", user_id,
            "--start-date", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "-o", f"monitoring/{user_id}_{datetime.now().strftime('%Y%m%d')}.json"
        ], capture_output=True)
        
        # Parse results and check for significant changes
        # (Add your alerting logic here)
        
        # Wait 24 hours
        time.sleep(86400)

# Run monitoring
monitor_mood("data/health_auto_export/", "patient_123")
```

## 🏷️ Advanced Labeling Workflows

### Multi-Rater Labeling

```bash
# Rater 1 labels
python src/big_mood_detector/main.py label episode \
    --rater "clinician_1" \
    --date-range 2024-01-15:2024-01-29 \
    --mood depressive \
    --severity moderate \
    --confidence high

# Rater 2 labels same period
python src/big_mood_detector/main.py label episode \
    --rater "clinician_2" \
    --date-range 2024-01-15:2024-01-29 \
    --mood depressive \
    --severity mild \
    --confidence medium

# Export inter-rater comparison
python src/big_mood_detector/main.py label export \
    --format csv \
    --include-raters \
    -o inter_rater_comparison.csv
```

### Importing Clinical Labels

```bash
# Import labels from clinical records
python src/big_mood_detector/main.py label import \
    clinical_episodes.csv \
    --format clinical \
    --validate
```

### Label Statistics and Analysis

```bash
# Detailed statistics
python src/big_mood_detector/main.py label stats \
    --detailed \
    --by-mood \
    --by-severity

# Export for analysis
python src/big_mood_detector/main.py label export \
    --format json \
    --include-features \
    -o labels_with_features.json
```

## 🎓 Model Training and Fine-Tuning

### Population Model Training

```python
# train_population_model.py
from big_mood_detector.infrastructure.fine_tuning.population_trainer import PopulationTrainer

trainer = PopulationTrainer(
    data_path="data/nhanes_processed/",
    output_dir="models/population/"
)

# Train on population data
metrics = trainer.train(
    epochs=100,
    learning_rate=0.01,
    early_stopping=True
)

print(f"Population model AUC: {metrics['auc']}")
```

### Personal Model Calibration

```bash
# Fine-tune for specific user
python src/big_mood_detector/main.py train \
    --model-type xgboost \
    --user-id "patient_123" \
    --data personal_features.csv \
    --labels personal_labels.csv \
    --base-model population \
    --calibration-method isotonic
```

### PAT Model Fine-Tuning

```python
# finetune_pat.py
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

# Load pretrained PAT
pat = PATModel(model_size="medium")
pat.load_pretrained()

# Fine-tune on user data
pat.finetune(
    train_data="data/user_activity_sequences.npy",
    labels="data/user_labels.npy",
    epochs=10,
    batch_size=32
)

# Save fine-tuned model
pat.save_finetuned(f"models/pat_user_123.h5")
```

## 🌐 API Advanced Features

### Batch Processing via API

```python
import requests
import json

# Batch upload
files = [
    ("files", open("sleep_january.json", "rb")),
    ("files", open("sleep_february.json", "rb")),
    ("files", open("sleep_march.json", "rb"))
]

response = requests.post(
    "http://localhost:8000/api/v1/upload/batch",
    files=files
)

batch_id = response.json()["batch_id"]

# Start batch processing
process_response = requests.post(
    "http://localhost:8000/api/v1/process/batch",
    json={
        "batch_id": batch_id,
        "options": {
            "ensemble": True,
            "generate_report": True
        }
    }
)

# Monitor progress
job_id = process_response.json()["job_id"]
while True:
    status = requests.get(f"http://localhost:8000/api/v1/process/status/{job_id}")
    if status.json()["status"] == "completed":
        break
    time.sleep(5)
```

### Webhook Integration

```python
# Configure webhooks for automatic alerts
webhook_config = {
    "url": "https://your-server.com/mood-alerts",
    "events": ["high_risk_detected", "processing_complete"],
    "filters": {
        "min_risk_level": 0.7,
        "mood_types": ["mania", "depression"]
    }
}

requests.post(
    "http://localhost:8000/api/v1/webhooks/configure",
    json=webhook_config
)
```

## 📈 Data Analysis and Visualization

### Export for Research

```bash
# Export features for statistical analysis
python src/big_mood_detector/main.py process data/health_auto_export/ \
    --output-format research \
    -o research_features.csv

# Include raw time series
python src/big_mood_detector/main.py process data/health_auto_export/ \
    --include-raw \
    --output-format hdf5 \
    -o time_series_data.h5
```

### Longitudinal Analysis

```python
# longitudinal_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from big_mood_detector.application.services.longitudinal_analyzer import LongitudinalAnalyzer

analyzer = LongitudinalAnalyzer()

# Load historical predictions
predictions = analyzer.load_predictions("output/predictions/user_123/")

# Analyze trends
trends = analyzer.analyze_trends(
    predictions,
    window_size=30,  # 30-day rolling average
    seasonality=True
)

# Generate report
analyzer.generate_report(
    trends,
    output_path="reports/longitudinal_user_123.pdf"
)
```

## 🔧 System Configuration

### Environment Variables

```bash
# .env file
BIG_MOOD_LOG_LEVEL=DEBUG
BIG_MOOD_MODEL_DIR=/path/to/models
BIG_MOOD_CACHE_DIR=/path/to/cache
BIG_MOOD_MAX_WORKERS=4
BIG_MOOD_MEMORY_LIMIT=8GB
```

### Custom Configuration

```python
# config/custom_settings.py
from big_mood_detector.core.config import Settings

class ProductionSettings(Settings):
    # API Configuration
    api_rate_limit: int = 100
    api_timeout: int = 300
    
    # Model Configuration
    ensemble_weights: dict = {
        "xgboost": 0.7,
        "pat": 0.3
    }
    
    # Clinical Thresholds
    risk_thresholds: dict = {
        "depression": {
            "low": 0.3,
            "moderate": 0.5,
            "high": 0.7,
            "critical": 0.9
        }
    }
```

## 🚀 Performance Optimization

### Parallel Processing

```bash
# Use multiple workers for batch processing
python src/big_mood_detector/main.py process data/large_dataset/ \
    --workers 8 \
    --chunk-size 1000 \
    -o batch_results.json
```

### Caching Strategy

```python
# Enable caching for repeated analyses
from big_mood_detector.infrastructure.caching import FeatureCache

cache = FeatureCache(
    cache_dir="/path/to/cache",
    ttl_days=7  # Cache for 7 days
)

# Processing will use cache when available
with cache:
    features = process_health_data(data_path)
```

## 🔍 Debugging and Troubleshooting

### Debug Mode

```bash
# Enable debug logging
export BIG_MOOD_LOG_LEVEL=DEBUG

# Run with verbose output
python src/big_mood_detector/main.py predict data/health_auto_export/ \
    -vvv \
    --debug \
    --profile
```

### Performance Profiling

```bash
# Profile memory usage
mprof run python src/big_mood_detector/main.py process large_export.xml
mprof plot

# Profile execution time
python -m cProfile -o profile.stats src/big_mood_detector/main.py process data/
```

## 📊 Clinical Integration

### HL7 FHIR Export

```python
# Export predictions as FHIR observations
from big_mood_detector.interfaces.fhir import FHIRExporter

exporter = FHIRExporter()
observations = exporter.export_predictions(
    predictions,
    patient_id="patient_123",
    practitioner_id="dr_smith"
)

# Send to EHR
exporter.send_to_ehr(observations, ehr_endpoint="https://hospital.ehr.com/fhir")
```

### Clinical Decision Support

```python
# Generate CDS alerts
from big_mood_detector.application.services.clinical_decision_support import CDSEngine

cds = CDSEngine()
alerts = cds.evaluate(
    predictions,
    patient_history,
    current_medications
)

for alert in alerts:
    if alert.severity == "high":
        # Send immediate notification
        notify_clinician(alert)
```

## 🎯 Best Practices

1. **Data Quality**
   - Ensure at least 30 days of continuous data
   - Check for data gaps before processing
   - Validate sensor accuracy

2. **Model Selection**
   - Use ensemble for general screening
   - Use personalized models after 90+ days of data
   - Retrain monthly for optimal performance

3. **Clinical Integration**
   - Always include confidence intervals
   - Document model version in reports
   - Maintain audit trail of predictions

4. **Privacy & Security**
   - Encrypt data at rest
   - Use secure API endpoints
   - Implement access controls

---

For more information, see the [Developer Documentation](../developer/) or [Clinical Guidelines](../clinical/).