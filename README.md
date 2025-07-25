# 🧠 Big Mood Detector

**Wearable-data mood insights for personal research**

> **For Researchers**: See [PAT Depression Training](docs/training/PAT_DEPRESSION_TRAINING.md) for PAT model training details

[![Tests](https://img.shields.io/badge/tests-976%20passing-brightgreen)](tests/) [![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](htmlcov/) [![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml) [![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Analyze your Apple Health data to understand mood patterns. Two models, two windows: PAT assesses your current state, XGBoost predicts tomorrow's risk.

## Why This Matters

Early warning signals for mood episodes help people seek care sooner. This project turns raw health data into two complementary risk scores:
- **Current state** from your past week of activity
- **Tomorrow's risk** from circadian rhythm patterns

Based on peer-reviewed research. Runs 100% locally. Your data never leaves your device.

## ⚠️ Medical Disclaimer

**Research software only. Not FDA-approved. Cannot diagnose conditions. Always consult healthcare professionals. If in crisis, call 988 (US) or emergency services.**

[Full disclaimer →](docs/clinical/README.md#medical-disclaimer)

## 🚀 Quick Start

```bash
# 1. Install
pip install big-mood-detector

# 2. Export Apple Health data (Settings → Health → Export)
#    Then unzip to get export.xml

# 3. Analyze the last 90 days
big-mood process export.xml --days-back 90
big-mood predict export.xml --report
```

**Need help?** See the [User Quick Start →](docs/user/QUICK_START_GUIDE.md)

## How It Works

```
Your Health Data
      ↓
┌─────────────────────┐
│   Past 7 Days       │ ← PAT analyzes patterns
├─────────────────────┤
│   Past 30 Days      │ ← XGBoost finds rhythms  
└─────────────────────┘
      ↓
Current State + Future Risk
```

[Architecture details →](docs/developer/ARCHITECTURE_OVERVIEW.md)

## Key Features

| Feature | Status | Details |
|---------|--------|---------|
| Process Apple Health exports | ✅ | XML and JSON formats |
| Current mood assessment (PAT) | ✅ | 0.56 AUC (matches paper) |
| Next-day predictions (XGBoost) | ✅ | 0.80-0.98 AUC (paper claims) |
| Personal baseline calibration | ✅ | Adapts to your patterns |
| Privacy-first architecture | ✅ | 100% on-device processing |
| Clinical-grade algorithms | ✅ | From Nature Digital Medicine |
| Real-time API | ✅ | REST endpoints for integration |
| Continuous monitoring | 🔜 | Coming soon |

## 📊 Performance

### Model Accuracy
- **Mania detection**: 0.98 AUC (exceptional)*
- **Hypomania**: 0.95 AUC (excellent)*
- **Depression**: 0.80 AUC (good)*
- **Current depression**: 0.56 AUC (limited)

*From published research, not independently validated

### Processing Speed
- **365 days in 17 seconds**
- **<100MB RAM** for any file size
- **33MB/s** parsing throughput

[Performance details →](docs/performance/OPTIMIZATION_TRACKING.md)

## 🛑 Limitations

- **Population mismatch** - Models trained on specific cohorts
- **No clinical validation** - Research prototype only
- **Moderate accuracy** - Especially for depression (0.56 AUC)
- **Not diagnostic** - Screening tool at best

[Full limitations →](docs/clinical/README.md#critical-limitations)

## Installation

### Requirements
- Python 3.12+
- 8GB RAM (16GB recommended)
- macOS, Linux, or Windows WSL2

### From PyPI
```bash
pip install big-mood-detector
```

### From Source
```bash
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector
pip install -e ".[dev,ml,monitoring]"
```

## CLI Reference

```bash
# Core commands
big-mood process <export.xml>          # Process health data
big-mood predict <export.xml> --report # Generate predictions
big-mood serve                         # Start API server

# Advanced
big-mood watch <directory>             # Monitor for new exports
big-mood label episode --type <type>   # Label past episodes
big-mood train --model <model>         # Train personal model
```

[Full CLI documentation →](docs/user/README.md#cli-command-reference)

## 📚 Documentation

| Audience | Start Here |
|----------|------------|
| **Users** | [Quick Start Guide](docs/user/QUICK_START_GUIDE.md) |
| **Developers** | [Architecture Overview](docs/developer/ARCHITECTURE_OVERVIEW.md) |
| **Researchers** | [Clinical Validation](docs/clinical/README.md) |

## Research Foundation

```bibtex
@article{lee2024predicting,
  title={Accurately predicting mood episodes in mood disorder patients 
         using wearable sleep and circadian rhythm features},
  author={Lee, Dongju and others},
  journal={npj Digital Medicine},
  volume={7},
  pages={154},
  year={2024},
  publisher={Nature Publishing Group}
}

@article{ruan2024pat,
  title={Pretrained Actigraphy Transformer (PAT): Plug-and-play 
         foundation model for wearable sensor data},
  author={Ruan, Franklin Y and others},
  journal={PLOS Digital Health},
  year={2024}
}
```

## Contributing

We need help with:
- 🏥 Clinical validation studies
- 🌍 Diverse population testing
- 📱 More device support (Fitbit, Garmin)
- 🧪 Improving PAT accuracy beyond 0.56

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Acknowledgments

Built on research from Seoul National University, Dartmouth College, and Harvard Medical School.

---

**For AI agents:** See [CLAUDE.md](CLAUDE.md) for codebase orientation.