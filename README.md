# 🧠 Big Mood Detector

## ⚠️ CRITICAL MEDICAL DISCLAIMERS

**This application is for RESEARCH and PERSONAL USE ONLY. It is NOT FDA-approved, NOT a medical device, and CANNOT diagnose mental health conditions. ALWAYS consult qualified healthcare professionals. If experiencing a mental health crisis, seek immediate help: Call 988 (US) or emergency services.**

**[📋 PLEASE READ IMPORTANT INFORMATION FIRST](docs/IMPORTANT_PLEASE_READ.md)**

---

> **Clinical-grade bipolar mood prediction from Apple Health data using validated ML models**

[![Tests](https://img.shields.io/badge/tests-695%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)](htmlcov/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Models](https://img.shields.io/badge/models-XGBoost%20%2B%20PAT-purple)](model_weights/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

A production-ready system for detecting mood episodes in bipolar disorder using wearable sensor data. Based on peer-reviewed research from Nature Digital Medicine, Harvard Medical School, and Dartmouth.

## ⚠️ Important v0.2.0 Limitations

**What Works Today:**
- ✅ **XGBoost predictions** - Fully validated mood risk scores (0.80-0.98 AUC)
- ✅ **Feature extraction** - Robust processing of Apple Health data
- ✅ **PAT embeddings** - Adds signal to XGBoost features

**Current Limitations:**
- ❌ **Not a true ensemble** - Only XGBoost makes predictions
- ❌ **PAT can't predict mood** - Outputs embeddings only (no classification heads)
- ❌ **Single model dependency** - No redundancy from dual predictions

**Coming in v0.3.0:** True dual-model ensemble with independent predictions. [See roadmap →](docs/ROADMAP_V0.3.0.md)

**Note:** This implementation has not been clinically validated. For research and personal use only.

## 🆕 What's New (v0.2.1)

- ✅ **Date Range Filtering**: Process large XML files with `--days-back` or `--date-range` options
- ✅ **Personal Baselines**: Adaptive predictions based on YOUR normal patterns
- ✅ **Enhanced Features**: XGBoost predictions with PAT embeddings
- ✅ **Clinical Reports**: DSM-5 aligned risk assessments with explanations
- ✅ **Python 3.12**: Full support with performance improvements
- ✅ **Enhanced Documentation**: Complete feature reference and math details

## 📋 Requirements

- Python 3.12 or higher
- 8GB RAM minimum (16GB recommended for large datasets)
- macOS, Linux, or Windows with WSL2

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector
pip install -e ".[dev,ml,monitoring]"

# Process health data
python src/big_mood_detector/main.py process data/health_auto_export/

# Generate predictions
python src/big_mood_detector/main.py predict data/health_auto_export/ --report

# Start API server
python src/big_mood_detector/main.py serve
```

## 📊 Clinical Performance

XGBoost model validated on 168 patients over 44,787 observation days:

| Condition | Accuracy (AUC) | Prediction Window | Clinical Use |
|-----------|----------------|-------------------|--------------|
| **Mania** | **0.98** ⭐ | Next 24 hours | Rule out bipolar episodes |
| **Hypomania** | **0.95** | Next 24 hours | Early warning system |
| **Depression** | **0.80** | Next 24 hours | Preventive interventions |

*Source: Nature Digital Medicine 2024 (Seoul National University)*

**Important Notes:**
- All predictions are **24-hour forecasts** (tomorrow's risk, not today's)
- v0.2.0 uses XGBoost only (PAT provides embeddings, not predictions)
- True ensemble with dual predictions coming in v0.3.0

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

Clean Architecture with Domain-Driven Design:

```
big-mood-detector/
├── src/big_mood_detector/
│   ├── domain/              # Core business logic (no dependencies)
│   ├── application/         # Use cases and orchestration
│   ├── infrastructure/      # External integrations (ML, DB, parsers)
│   └── interfaces/          # User interfaces (CLI, API)
├── model_weights/           # Pre-trained models (XGBoost + PAT)
├── docs/                    # Comprehensive documentation
│   ├── user/               # User guides and tutorials
│   ├── clinical/           # Clinical validation and research
│   ├── developer/          # Technical documentation
│   └── models/             # ML model details and math
└── tests/                   # 695 tests (91% coverage)
```

## 🧬 Key Features

### 1. **ML Models (v0.2.0 Status)**

- **XGBoost** ✅: Fully functional with 36 engineered features, validated predictions (0.80-0.98 AUC)
- **PAT Transformer** ⚠️: Provides 96-dim embeddings to enhance XGBoost features (no independent predictions)
- **Current "Ensemble"** 🔄: XGBoost with PAT-enhanced features (true ensemble coming v0.3.0)

### 2. **Personal Baseline System**
- Learns YOUR normal patterns (not population average)
- Adapts to chronotypes (night owls vs early birds)
- Reduces false positives for athletes, shift workers
- Updates continuously with new data

### 3. **Clinical-Grade Analysis**
- **Sleep Architecture**: Advanced 3.75-hour window merging
- **Circadian Phase**: Gold-standard DLMO estimation
- **Activity Patterns**: 1440-minute daily sequences
- **Feature Engineering**: 36 validated biomarkers

### 4. **Production Architecture**
- **Performance**: <100ms predictions, handles 500MB+ files
- **Scalable**: Async FastAPI, Redis queuing, Docker ready
- **Privacy-First**: Local processing, no cloud dependencies
- **Extensible**: Clean architecture for new models/features

## 📋 Commands

| Command | Description | Example |
|---------|-------------|---------|
| `process` | Extract features from health data | `process data/ -o features.json` |
| `predict` | Generate mood predictions | `predict data/ --report` |
| `label` | Create ground truth annotations | `label episode --date-range 2024-01-01:2024-01-14` |
| `train` | Fine-tune personalized model | `train --user-id patient_123 --data features.csv` |
| `serve` | Start API server | `serve --port 8000 --reload` |
| `watch` | Monitor directory for new files | `watch data/health_auto_export/` |

### Date Range Filtering (New in v0.2.1)

For large XML files (500MB+), you can now filter data by date to reduce processing time:

```bash
# Process only the last 90 days
python src/big_mood_detector/main.py process export.xml --days-back 90

# Process a specific date range
python src/big_mood_detector/main.py process export.xml --date-range 2024-01-01:2024-03-31

# Same options work for predict command
python src/big_mood_detector/main.py predict export.xml --days-back 30 --report
```

This is especially useful for:
- Large Apple Health exports (years of data)
- Quick analysis of recent periods
- Memory-constrained environments
- Faster iteration during development

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

### For Users
- **[Quick Start Guide](docs/user/QUICK_START_GUIDE.md)** ⭐ - Get running in 5 minutes
- **[Application Workflow](docs/user-guide/APPLICATION_WORKFLOW.md)** - How it works
- **[Apple Health Export](docs/user/APPLE_HEALTH_EXPORT.md)** - Data export guide

### For Developers  
- **[Architecture Overview](docs/developer/ARCHITECTURE_OVERVIEW.md)** - System design
- **[API Reference](docs/developer/API_REFERENCE.md)** - REST endpoints
- **[Model Mathematics](docs/models/)** - Feature formulas and model details

### For Clinicians
- **[Clinical Validation](docs/clinical/CLINICAL_DOSSIER.md)** - Research foundation
- **[Feature Reference](docs/models/xgboost-features/FEATURE_REFERENCE.md)** - All 36 biomarkers
- **[Literature Review](docs/literature/)** - Peer-reviewed papers

### 📖 [Full Documentation](docs/README.md)

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

## ⚠️ Medical & Safety Disclaimers

1. **NOT A DIAGNOSTIC TOOL**: Provides risk assessments only, cannot diagnose conditions
2. **NOT FDA APPROVED**: This is research software, not a medical device
3. **REQUIRES PROFESSIONAL CONSULTATION**: Always work with qualified healthcare providers
4. **RESEARCH STATUS**: Based on validated papers but this implementation is not clinically tested
5. **INDIVIDUAL VARIABILITY**: Accuracy varies by person and improves with calibration
6. **EMERGENCY**: If in crisis, call 988 (US) or seek immediate professional help

**[Full disclaimers and safety information](docs/IMPORTANT_PLEASE_READ.md)**

## 📄 License

Apache 2.0 License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Seoul National University Bundang Hospital
- Harvard Medical School 
- Dartmouth Center for Technology and Behavioral Health
- UC Berkeley Center for Human Sleep Science

---

**Built with ❤️ for the mental health community**

*For AI agents: See [CLAUDE.md](CLAUDE.md) for codebase orientation*