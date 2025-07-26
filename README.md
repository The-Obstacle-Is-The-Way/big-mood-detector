# 🧠 Big Mood Detector

**Predict mood episodes from your wearable data — clinically informed, privacy-first, open-source.**

> **For Researchers**: See [PAT Depression Training](docs/training/PAT_DEPRESSION_TRAINING.md) for PAT model training details

[![Tests](https://img.shields.io/badge/tests-1000%2B%20passing-brightgreen)](tests/) [![Coverage](https://img.shields.io/badge/coverage-72%25-yellow)](htmlcov/) [![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml) [![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Big Mood Detector analyzes your Apple Health data to predict mood episode risk using AI. Two complementary models: PAT transformer assesses current state, XGBoost predicts tomorrow's risk. Built by a clinical psychiatrist, based on Nature research, and runs 100% locally.

**Current status**: Research prototype — the first of its kind, but not yet clinically validated.

## Why Use Big Mood Detector?

**The clinical problem**: No objective tools exist for predicting mood episodes or distinguishing bipolar from unipolar depression or borderline personality disorder. Clinicians rely on subjective recall; patients often seek help after crises begin.

**Our breakthrough**:
- **Early detection**: Spot mood episode risk before symptoms spiral
- **Two timescales**: Current state (PAT, 7-day patterns) + tomorrow's risk (XGBoost, circadian rhythms)  
- **Objective data**: Complement clinical assessment with continuous behavioral biomarkers
- **Research foundation**: First implementation of peer-reviewed algorithms from Nature Digital Medicine
- **Privacy-first**: Runs entirely on your device — your data never leaves your machine

**For researchers**: Validate these approaches across populations, build the evidence base for digital mental health.

## ⚠️ Research Limitations

**Population specificity**:
- XGBoost: Trained only on Korean bipolar patients (ages 18-35)
- PAT: US NHANES participants who completed surveys (selection bias likely)

**Performance constraints**:
- Depression detection: Moderate accuracy (0.56-0.80 AUC)
- No validation across ethnicities, age groups, or comorbid conditions
- Research tool only — not FDA approved or clinically validated

## 🚀 Quick Start

*Takes 2 minutes on any Mac/PC*

```bash
# 1. Install
pip install big-mood-detector

# 2. Export Apple Health data (Settings → Health → Export)
#    Unzip to get export.xml

# 3. Analyze your data (research purposes)
big-mood process export.xml --days-back 90
big-mood predict export.xml --report
```

**Need help?** See the [User Quick Start →](docs/user/QUICK_START_GUIDE.md)

## How It Works

```
Your Apple Health Data
      ↓
┌─────────────────────┐
│   Past 7 Days       │ ← PAT (transformer) analyzes activity patterns
├─────────────────────┤
│   Past 30 Days      │ ← XGBoost models circadian rhythms  
└─────────────────────┘
      ↓
Research Risk Scores (Not Diagnostic)
```

**PAT** = transformer AI, **XGBoost** = gradient boosting, **ensemble** = enhanced reliability.

[Architecture details →](docs/developer/ARCHITECTURE_OVERVIEW.md)

## Technical Features

| Component | Status | Performance |
|-----------|--------|-------------|
| Apple Health XML/JSON parsing | ✅ | 33MB/s, <100MB RAM |
| PAT transformer model | ✅ | 0.56-0.70 AUC (research) |
| XGBoost circadian model | ✅ | 0.80-0.98 AUC (Korean cohort) |
| Privacy-first processing | ✅ | 100% local, no data sharing |
| Clinical feature extraction | ✅ | DSM-5 aligned thresholds |
| REST API | ✅ | Real-time predictions |
| Population adaptation | 🔬 | Research needed |

## Performance Benchmarks

**Processing Speed**:
- 365 days of data: 17 seconds
- Memory usage: <100MB for any file size
- Parsing throughput: 33MB/s

**Model Accuracy** (from original research):
- Mania prediction: 0.98 AUC (Korean bipolar cohort)*
- Hypomania: 0.95 AUC (Korean bipolar cohort)*  
- Depression (bipolar): 0.80 AUC (Korean cohort)*
- Depression (general): 0.56 AUC (US NHANES)*

*Research results, not independently validated

[Performance details →](docs/performance/OPTIMIZATION_TRACKING.md)

## What Makes This Revolutionary

**Clinical innovation**: First tool to predict mood episodes from everyday wearable data

**Scientific rigor**: Implements published algorithms from Nature Digital Medicine, transparent methodology

**Privacy breakthrough**: No cloud dependency, no data collection — your mental health data stays private

**Open research**: Complete transparency enables validation, improvement, and trust

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
# Core research commands
big-mood process <export.xml>          # Process health data
big-mood predict <export.xml> --report # Generate research scores
big-mood serve                         # Start API server

# Advanced research tools
big-mood label episode --type <type>   # Annotate historical episodes
big-mood train --model <model>         # Experiment with personal models
```

[Full CLI documentation →](docs/user/README.md#cli-command-reference)

## Research Applications

**For researchers**:
- Validate algorithms across diverse populations
- Study wearable data patterns in mental health
- Develop population-specific models

**For developers**:
- Build mental health applications
- Integrate mood prediction into health platforms
- Explore transformer architectures for time-series health data

**For individuals**:
- Understand your own activity patterns
- Contribute to research (with appropriate IRB protocols)
- Explore personal digital biomarkers

## 📚 Documentation

| Audience | Start Here |
|----------|------------|
| **Users** | [Quick Start Guide](docs/user/QUICK_START_GUIDE.md) |
| **Developers** | [Architecture Overview](docs/developer/ARCHITECTURE_OVERVIEW.md) |
| **Researchers** | [Clinical Requirements](docs/clinical/README.md) |

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
  title={Pretrained Actigraphy Transformer (PAT): Foundation model 
         for wearable sensor data in mental health research},
  author={Ruan, Franklin Y and others},
  journal={PLOS Digital Health},
  year={2024}
}
```

## Contributing to Research

**Critical research needs**:
- 🏥 Clinical validation across diverse populations
- 🌍 Multi-ethnic, multi-age cohort studies  
- 📱 Integration with additional wearable devices
- 🧪 Improving transformer model accuracy
- 🔬 Longitudinal outcome studies

*For clinical validation collaborations or enterprise applications, open an issue or contact the maintainers.*

See [CONTRIBUTING.md](CONTRIBUTING.md) for research collaboration guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Acknowledgments

Built on pioneering research from Seoul National University, Dartmouth College, and Harvard Medical School. This implementation makes their breakthrough algorithms accessible for the first time.

---

**Have feedback? Want to join the next phase of wearable mental health? Open an issue or contact us.**

**For AI agents:** See [CLAUDE.md](CLAUDE.md) for codebase orientation.