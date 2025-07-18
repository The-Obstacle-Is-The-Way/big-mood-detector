# 🧠 Big Mood Detector

> **Clinical-grade bipolar mood prediction from Apple Health data using validated ML models**

[![Tests](https://img.shields.io/badge/tests-695%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)](htmlcov/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Models](https://img.shields.io/badge/models-XGBoost%20%2B%20PAT-purple)](model_weights/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-ready system for detecting mood episodes in bipolar disorder using wearable sensor data. Based on peer-reviewed research from Nature Digital Medicine, Harvard Medical School, and Dartmouth.

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/yourusername/big-mood-detector.git
cd big-mood-detector
pip install -e ".[dev,ml,monitoring]"

# Process health data
python src/big_mood_detector/main.py process data/health_auto_export/

# Generate predictions
python src/big_mood_detector/main.py predict data/health_auto_export/ --report

# Start API server
python src/big_mood_detector/main.py serve
```

## 📊 What It Does

Analyzes Apple Health data to predict risk of mood episodes with clinical-grade accuracy:

- **Depression Detection**: AUC 0.80 (Seoul National study)
- **Mania Detection**: AUC 0.98 (validated on 168 patients)
- **Hypomania Detection**: AUC 0.95 (44,787 observation days)

### Sample Output
```
Big Mood Detector - Clinical Report
Generated: 2025-07-18 10:30:00

RISK SUMMARY:
- Depression Risk: MODERATE (0.65)
- Mania Risk: LOW (0.12)
- Hypomania Risk: LOW (0.18)

KEY FINDINGS:
✓ Sleep duration: 6.2 hours average (below optimal)
✓ Circadian phase delay: -1.2 hours detected
✓ Activity fragmentation: Increased

CLINICAL FLAGS:
⚠️ Decreased sleep duration trend
⚠️ Irregular sleep schedule
```

## 🏗️ Architecture

```
├── src/big_mood_detector/
│   ├── domain/              # Core business logic (Clean Architecture)
│   ├── application/         # Use cases and orchestration
│   ├── infrastructure/      # ML models, parsers, repositories
│   └── interfaces/          # CLI and API endpoints
├── model_weights/           # Pre-trained XGBoost + PAT models
├── reference_repos/         # Academic implementations
├── docs/literature/         # Research papers and clinical studies
└── tests/                   # 695 comprehensive tests
```

## 🧬 Key Features

### 1. **Dual ML Pipeline**
- **XGBoost Models**: 36 sleep/circadian features from Seoul National study
- **PAT Transformer**: Activity sequence analysis (29,307 participants)
- **Ensemble Predictions**: Combined accuracy exceeding individual models

### 2. **Clinical Features**
- Sleep window analysis (3.75-hour merging algorithm)
- Circadian rhythm calculation (phase, amplitude, stability)
- Activity pattern extraction (1440-minute sequences)
- Heart rate variability analysis

### 3. **Production Ready**
- FastAPI server with async processing
- Streaming parser for 500MB+ files
- Background task queue (Celery + Redis)
- Docker deployment ready

### 4. **Personal Calibration**
- Fine-tune models on individual data
- Label historical episodes for training
- Adaptive thresholds based on personal baselines

## 📋 Commands

| Command | Description | Example |
|---------|-------------|---------|
| `process` | Extract features from health data | `process data/ -o features.json` |
| `predict` | Generate mood predictions | `predict data/ --ensemble --report` |
| `label` | Create ground truth annotations | `label episode --date-range 2024-01-01:2024-01-14` |
| `train` | Fine-tune personalized model | `train --user-id patient_123 --data features.csv` |
| `serve` | Start API server | `serve --port 8000 --reload` |
| `watch` | Monitor directory for new files | `watch data/health_auto_export/` |

## 🔬 Research Foundation

Based on 6 peer-reviewed studies:

1. **[Nature Digital Medicine 2024](docs/literature/converted_markdown/xgboost-mood/)** - Seoul National University
   - 168 patients, 44,787 observation days
   - Circadian phase as top predictor

2. **[Bipolar Disorders 2024](docs/literature/converted_markdown/fitbit-bipolar-mood/)** - Harvard Medical School
   - Consumer device validation
   - BiMM Forest algorithm

3. **[Pretrained Actigraphy Transformer](docs/literature/converted_markdown/pretrained-actigraphy-transformer/)** - Dartmouth
   - Foundation model for movement data
   - 29,307 participant training set

## 📊 Data Requirements

### Supported Formats
- **Apple Health XML Export** - Complete sensor data
- **Health Auto Export JSON** - Daily aggregated summaries

### Minimum Data
- 30 days for initial predictions
- 60+ days for optimal accuracy
- Consistent device usage required

## 🚀 API Integration

```python
import requests

# Upload health data
response = requests.post(
    "http://localhost:8000/api/v1/upload/file",
    files={"file": open("health_data.json", "rb")}
)

# Start processing
job = requests.post(
    "http://localhost:8000/api/v1/process/start",
    json={"upload_id": response.json()["upload_id"]}
)

# Get results
results = requests.get(
    f"http://localhost:8000/api/v1/results/{job.json()['job_id']}"
)
```

## 🏥 Clinical Integration

- **DSM-5 Aligned**: Uses clinical thresholds from psychiatric guidelines
- **FHIR Compatible**: Export predictions for EHR integration
- **Multi-rater Support**: Built-in inter-rater reliability
- **Audit Trail**: Complete prediction history and confidence scores

## 📚 Documentation

- **[Quick Start Guide](docs/user/QUICK_START_GUIDE.md)** - Get running in 5 minutes
- **[Architecture Overview](docs/developer/ARCHITECTURE_OVERVIEW.md)** - System design
- **[API Reference](docs/developer/API_REFERENCE.md)** - REST endpoints
- **[Clinical Documentation](docs/clinical/CLINICAL_DOSSIER.md)** - Thresholds and validation
- **[Deployment Guide](docs/developer/DEPLOYMENT_GUIDE.md)** - Production setup

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test categories
make test-unit      # Domain logic tests
make test-integration  # Parser and DB tests
make test-ml        # Model validation tests

# Coverage report
make coverage
```

## 🔒 Security & Privacy

- All processing happens locally by default
- No data sent to external servers
- Configurable data retention policies
- Audit logging for compliance

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

Key areas for contribution:
- Additional wearable device support
- Clinical validation studies
- Performance optimizations
- Documentation improvements

## ⚠️ Disclaimers

1. **Clinical Tool**: Provides risk assessments, not diagnoses
2. **Professional Consultation**: Always consult healthcare providers
3. **Research Use**: Validated in research settings
4. **Individual Variability**: Requires personal calibration

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Seoul National University Bundang Hospital
- Harvard Medical School 
- Dartmouth Center for Technology and Behavioral Health
- UC Berkeley Center for Human Sleep Science

---

**Built with ❤️ for the mental health community**

*For AI agents: See [CLAUDE.md](CLAUDE.md) for codebase orientation*