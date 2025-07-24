# 🧠 Big Mood Detector

**Analyze your Apple Health data to understand your mood risk - both current state and future predictions.**

[![Tests](https://img.shields.io/badge/tests-976%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](htmlcov/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## ⚠️ Medical Disclaimer

**This is research software for personal exploration only. It is NOT FDA-approved, NOT a medical device, and CANNOT diagnose mental health conditions. Always consult qualified healthcare professionals. If experiencing a mental health crisis, call 988 (US) or emergency services.**

## What This Does

Big Mood Detector analyzes your Apple Health data to provide two key insights:

1. **Current Mood State** - Using the Pretrained Actigraphy Transformer (PAT), it analyzes your past 7 days of activity patterns to assess your current depression risk
2. **Tomorrow's Risk** - Using XGBoost with 36 engineered features, it predicts your next-day risk for mood episodes (depression, mania, hypomania)

Think of it as having two different lenses to understand your mood patterns - one looking at where you are now, another looking at where you might be heading.

## How It Works

```bash
# 1. Export your Apple Health data (Settings → Health → Export)

# 2. Process your data
python src/big_mood_detector/main.py process export.xml

# 3. Get predictions
python src/big_mood_detector/main.py predict export.xml --report
```

You'll receive a report showing:
- Your current mood state assessment (based on recent activity patterns)
- Tomorrow's predicted risk levels (depression, mania, hypomania)
- Key behavioral patterns detected (sleep disruption, activity changes, circadian shifts)

## The Science Behind It

This application implements two peer-reviewed research models:

### XGBoost Model (Seoul National University, 2024)
- Predicts **next-day** mood episode risk
- Performance on 168 patients over 44,787 days:
  - Depression: 0.80 AUC
  - Mania: 0.98 AUC
  - Hypomania: 0.95 AUC
- Key insight: Circadian phase shifts are the strongest predictor

### PAT Model (Dartmouth, 2024)
- Assesses **current** mood state from activity patterns
- Pretrained on 29,307 US participants
- Our implementation achieves paper-matching performance:
  - Depression detection: 0.56 AUC (PAT-S)
  - Currently training larger models for improved accuracy

## Important Limitations

1. **Model Training Cohorts**
   - XGBoost: Trained on Korean patients at Seoul National University
   - PAT: Trained on US NHANES 2013-2014 data
   - Generalization to other populations needs validation

2. **Predictive Strength**
   - 0.56-0.80 AUC indicates moderate predictive ability
   - Not a replacement for clinical assessment
   - Best used as one data point among many

3. **Current Implementation Status**
   - XGBoost predictions: ✅ Fully operational
   - PAT current state assessment: 🔄 Models trained, integration in progress
   - Personal calibration: ✅ Adapts to your individual patterns

## Quick Start

### Requirements
- Python 3.12+
- 8GB RAM (16GB recommended)
- macOS, Linux, or Windows with WSL2

### Installation

```bash
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector
pip install -e ".[dev,ml,monitoring]"
```

### Basic Usage

```bash
# Process last 90 days (recommended for large files)
python src/big_mood_detector/main.py process export.xml --days-back 90

# Get predictions with clinical report
python src/big_mood_detector/main.py predict export.xml --report

# Start API server for continuous monitoring
python src/big_mood_detector/main.py serve
```

## Advanced Features

### Personal Baseline Calibration
The system learns YOUR normal patterns, not population averages. This reduces false positives for:
- Athletes (naturally lower heart rate)
- Night owls (different circadian patterns)
- Shift workers (irregular schedules)

### Performance Optimization
- Handles 365 days of data in 17 seconds
- Processes 500MB+ XML files without memory issues
- Smart date filtering for large exports

### Privacy First
- All processing happens locally on your device
- No data sent to external servers
- You control your health information

## Coming Soon

- **All-cause mortality risk assessment** - PAT can be fine-tuned to predict general health outcomes
- **Additional mood disorder support** - Expanding beyond bipolar to other conditions
- **More wearable devices** - Currently Apple Watch focused, expanding to Fitbit, Garmin

## Documentation

- **[Quick Start Guide](docs/user/QUICK_START_GUIDE.md)** - Get running in 5 minutes
- **[Understanding Your Report](docs/user-guide/REPORT_INTERPRETATION.md)** - What the numbers mean
- **[API Documentation](docs/developer/API_REFERENCE.md)** - For developers
- **[Research Papers](docs/literature/)** - The science behind the models

## Contributing

We welcome contributions! Key areas:
- Completing PAT integration for current state assessment
- Improving model generalization across populations
- Adding support for more wearable devices
- Clinical validation studies

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

Built on research from:
- Seoul National University Bundang Hospital
- Dartmouth Center for Technology and Behavioral Health
- Harvard Medical School

---

**For AI agents working on this codebase:** See [CLAUDE.md](CLAUDE.md) for technical orientation.