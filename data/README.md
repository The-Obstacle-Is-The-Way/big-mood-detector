# Data Directory Structure

This directory contains all user-specific data for the Big Mood Detector. **All contents are gitignored** to protect personal health information.

## Directory Layout

```
data/
├── input/                      # Place your health data here
│   ├── apple_export/          # Apple Health XML exports
│   │   ├── export.xml         # Main health data export
│   │   ├── export_cda.xml     # Clinical document export
│   │   ├── electrocardiograms/
│   │   └── workout-routes/
│   │
│   └── health_auto_export/    # Health Auto Export JSON files
│       ├── Sleep Analysis.json
│       ├── Heart Rate.json
│       ├── Step Count.json
│       └── ... (other metrics)
│
├── output/                    # Processing results go here
│   ├── features_*.csv         # Extracted features
│   ├── predictions_*.json     # Model predictions
│   └── clinical_report.txt    # Clinical summary reports
│
├── uploads/                   # API file uploads (temporary)
│
└── temp/                      # Temporary processing files
```

## How to Use

### 1. Apple Health Export
Place your Apple Health export files in `data/input/apple_export/`:

```bash
# Process Apple Health export
mood-detector process data/input/apple_export/export.xml
mood-detector predict data/input/apple_export/export.xml
```

### 2. Health Auto Export
Place your JSON files from Health Auto Export app in `data/input/health_auto_export/`:

```bash
# Process Health Auto Export data
mood-detector process data/input/health_auto_export/
mood-detector predict data/input/health_auto_export/
```

### 3. API Upload
The API will automatically save uploaded files to `data/uploads/` for processing.

## Important Notes

- **Privacy**: This entire directory is gitignored. Never commit personal health data.
- **Size**: Apple Health exports can be very large (500MB+). Ensure you have enough disk space.
- **Cleanup**: Periodically clean the `output/` and `temp/` directories to save space.

## Model Weights Location

Model weights are stored separately in `model_weights/` at the repository root. These are pre-trained models included with the software.