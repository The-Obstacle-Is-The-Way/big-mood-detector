# Big Mood Detector

Temporal mood prediction from wearable data. Two models, two windows: PAT analyzes your current state, XGBoost predicts tomorrow's risk.

[![Tests](https://img.shields.io/badge/tests-976%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](htmlcov/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## What This Actually Does

Processes your Apple Health export and generates mood risk predictions using two distinct approaches:

1. **PAT (Pretrained Actigraphy Transformer)** - Assesses current depression state from past 7 days of activity
2. **XGBoost** - Predicts tomorrow's risk for depression/mania/hypomania based on circadian patterns

This temporal separation is novel - most systems blend past and future. We don't.

## Performance Claims vs Reality

### Published Research Claims
- **Mania detection**: 0.98 AUC (Seoul National University)
- **Hypomania**: 0.95 AUC 
- **Depression**: 0.80 AUC

### Our Implementation
- **PAT-S Depression**: 0.56 AUC (matches paper's 0.560)
- **PAT-M Depression**: 0.54 AUC (paper: 0.559)
- **PAT-L**: Training in progress
- **XGBoost**: Using pre-trained weights from paper (no independent validation)

**Reality check**: 0.56 AUC is barely better than random (0.5). The XGBoost claims are from the paper - we haven't validated them independently.

## Quick Start

```bash
# Install
pip install -e ".[dev,ml,monitoring]"

# Process your Apple Health export
python src/big_mood_detector/main.py process ~/Downloads/export.xml

# Get predictions
python src/big_mood_detector/main.py predict ~/Downloads/export.xml --report
```

## What You Get

```
Temporal Mood Assessment Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current State Assessment (PAT - Past 7 days):
  Depression Risk: 23% (LOW)
  Model Confidence: 0.56 AUC

Tomorrow's Predictions (XGBoost - Circadian):
  Depression: 15% (LOW)
  Mania: 3% (LOW)
  Hypomania: 28% (LOW)
  
Key Patterns:
  • Sleep duration: -1.2 hrs from baseline
  • Circadian phase: +0.8 hrs (slight delay)
  • Activity fragmentation: Normal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## CLI Commands

```bash
# Process health data
python src/big_mood_detector/main.py process <export_file>

# Generate predictions
python src/big_mood_detector/main.py predict <export_file> [--report]

# Start API server
python src/big_mood_detector/main.py serve

# Watch directory for new exports
python src/big_mood_detector/main.py watch <directory>

# Label episodes (for research)
python src/big_mood_detector/main.py label episode \
  --episode-type depressive \
  --severity moderate \
  --start-date 2024-01-15 \
  --end-date 2024-01-22

# Train personal model (experimental)
python src/big_mood_detector/main.py train \
  --model-type xgboost \
  --user-id user123 \
  --data features.csv \
  --labels labels.csv
```

## Architecture

Clean Architecture with dependency injection:

```
├── domain/           # Business logic, no external dependencies
├── application/      # Use cases, orchestration
├── infrastructure/   # ML models, data access, external services
└── interfaces/       # CLI, API, future web UI
```

Key components:
- **Temporal Ensemble Orchestrator** - Manages PAT + XGBoost predictions
- **Streaming XML Parser** - Handles 500MB+ files with <100MB RAM
- **Personal Baseline Tracker** - Adapts to individual patterns
- **Clinical Report Generator** - Produces interpretable outputs

## The Science

### XGBoost (Seoul National University, 2024)
- 168 patients with mood disorders
- 44,787 days of data
- 36 engineered features (sleep, circadian, activity)
- Key insight: Circadian phase shift is the strongest predictor

### PAT (Dartmouth, 2024)
- Transformer architecture for wearable time series
- Pretrained on 29,307 NHANES participants
- Processes 10,080 minutes (7 days) of activity data
- Limited by general population training (not clinical cohort)

## Performance

- **XML Processing**: 33MB/s (>40k records/second)
- **Full Pipeline**: 17.4s for 365 days of data
- **Memory**: <100MB for any file size
- **API Response**: <200ms average

## Limitations

1. **PAT performance is mediocre** - 0.56 AUC barely beats random
2. **No clinical validation** - Models from papers, not validated on real patients
3. **Population mismatch** - PAT trained on general population, not bipolar cohort
4. **XGBoost black box** - Can't inspect or validate the pre-trained models
5. **Not FDA approved** - Research prototype, not medical device

## Dependencies

Core:
- Python 3.12+
- XGBoost for predictions
- PyTorch for PAT implementation
- FastAPI for API server
- Click for CLI

## Development

```bash
# Setup
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml,monitoring]"

# Run tests (976 passing)
make test

# Type checking
make type-check

# Linting
make lint

# Full quality check
make quality
```

## Medical Disclaimer

**This is research software. Not FDA-approved. Cannot diagnose conditions. Not a substitute for professional care. If in crisis, call 988 (US) or emergency services.**

## Contributing

Real needs:
- Clinical validation with actual patient data
- Improve PAT depression detection beyond 0.56 AUC
- Add more wearable device support
- Create interpretability tools for XGBoost predictions

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citations

```bibtex
@article{lee2024predicting,
  title={Accurately predicting mood episodes in mood disorder patients using wearable sleep and circadian rhythm features},
  author={Lee, Dongju and others},
  journal={npj Digital Medicine},
  volume={7},
  number={1},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group}
}

@article{ruan2024pat,
  title={AI Foundation Models for Wearable Movement Data},
  author={Ruan, Franklin and others},
  journal={Dartmouth College},
  year={2024}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE)

---

For AI agents: See [CLAUDE.md](CLAUDE.md) for codebase orientation.