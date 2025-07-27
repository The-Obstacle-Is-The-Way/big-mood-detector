# 🧠 Big Mood Detector

**Predict mood episodes from your wearable data — clinically informed, privacy-first, open-source.**

> **For Researchers**: See [PAT Depression Training](docs/training/PAT_DEPRESSION_TRAINING.md) for PAT model training details

[![Tests](https://img.shields.io/badge/tests-1000%2B%20passing-brightgreen)](tests/) [![Coverage](https://img.shields.io/badge/coverage-72%25-yellow)](htmlcov/) [![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml) [![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Big Mood Detector analyzes your Apple Health data to predict mood episode risk using AI. 

Two complementary models: 
- PAT transformer screens for current depression
- XGBoost predicts tomorrow's depression/mania/hypomania risk.

Built by a clinical psychiatrist, implementing published research, and runs 100% locally.

**Current status**: Research prototype — the first of its kind, but not yet clinically validated.

## Why Use Big Mood Detector?

**The clinical problem**: No objective tool exist for predicting mood episodes or distinguishing unipolar from bipolar depression or borderline personality disorder. Clinicians rely on subjective recall; patients often seek help after crises begin.

**Our breakthrough**:
- **Early detection**: Spot mood episode risk before symptoms spiral
- **Two applications**: Current depression screening (PAT, general population) + next-day episode prediction (XGBoost, mood disorder patients)
- **Objective data**: Complement clinical assessment with continuous behavioral biomarkers
- **Research foundation**: First implementation combining two breakthrough papers:
  - XGBoost: [Nature Digital Medicine 2024](https://www.nature.com/articles/s41746-024-01333-z) ([GitHub](https://github.com/mcqeen1207/mood_ml))
  - PAT: [Dartmouth Foundation Model](https://arxiv.org/abs/2411.15240) ([GitHub](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer/))
- **Privacy-first**: Runs entirely on your device — your data never leaves your machine

**For researchers**: Validate these approaches across populations, build the evidence base for digital mental health.

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
| PAT transformer model | ✅ | 0.610 AUC depression (NHANES) |
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

*Current depression screening (PAT, general population)*:
- Depression detection: 0.610 AUC (US NHANES 2013-14)¹

*Next-day episode prediction (XGBoost, mood disorder patients)*:
- Depression episodes: 0.80 AUC (Korean cohort, MDD+BD patients)²
- Manic episodes: 0.98 AUC (Korean cohort, BD patients)²
- Hypomanic episodes: 0.95 AUC (Korean cohort, BD patients)²

¹[Ruan et al., 2024](https://arxiv.org/abs/2411.15240) | ²[Lim et al., 2024](https://www.nature.com/articles/s41746-024-01333-z)

[Performance details →](docs/performance/OPTIMIZATION_TRACKING.md)

## What Makes This Revolutionary

**Clinical innovation**: First open-source tool combining two state-of-the-art approaches to predict mood episodes from wearable data

**Scientific rigor**: Faithful implementation of published algorithms:
- XGBoost circadian model from Seoul National University (Nature Digital Medicine 2024)
- PAT transformer from Dartmouth (first foundation model for actigraphy)

**Privacy breakthrough**: No cloud dependency, no data collection — your mental health data stays private

**Open research**: Complete transparency enables validation, improvement, and trust

## ⚠️ Research Limitations

**Population specificity**:
- **XGBoost**: Trained on 168 Korean adults (18-35y) with mood disorders:
  - 57 (34%) with Major Depressive Disorder (MDD)
  - 42 (25%) with Bipolar I Disorder (BD1)
  - 69 (41%) with Bipolar II Disorder (BD2)
- **PAT depression model**:
  - **Pre-training**: 21,538 US adults from NHANES (2003-04, 2005-06, 2011-12)
  - **Fine-tuning (depression task)**: 2,800 participants from NHANES 2013-14 with PHQ-9 scores

**Performance constraints**:
- Current depression screening: Moderate accuracy (0.610 AUC)
- Next-day episode prediction: Limited to Korean mood disorder cohort (0.80-0.98 AUC)
- No validation across ethnicities, age groups, or comorbid conditions
- Research tool only — not FDA approved or clinically validated

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
@article{lim2024accurately,
  title={Accurately predicting mood episodes in mood disorder patients 
         using wearable sleep and circadian rhythm features},
  author={Lim, Dongju and Jeong, Jaegwon and Song, Yun Min and 
         Cho, Chul-Hyun and Yeom, Ji Won and Lee, Taek and 
         Lee, Jung-Been and Lee, Heon-Jeong and Kim, Jae Kyoung},
  journal={npj Digital Medicine},
  volume={7},
  pages={324},
  year={2024},
  doi={10.1038/s41746-024-01333-z}
}

@article{ruan2024foundation,
  title={AI Foundation Models for Wearable Movement Data in 
         Mental Health Research},
  author={Ruan, Franklin Y. and Zhang, Aiwei and Oh, Jenny Y. and 
         Jin, SouYoung and Jacobson, Nicholas C.},
  journal={arXiv preprint arXiv:2411.15240},
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

Built on pioneering research:
- **XGBoost models**: Seoul National University, Korea University, KAIST, and collaborators ([Lim et al., 2024](https://www.nature.com/articles/s41746-024-01333-z))
- **PAT foundation model**: Dartmouth College Center for Technology and Behavioral Health ([Ruan et al., 2024](https://arxiv.org/abs/2411.15240))

This implementation makes their breakthrough algorithms accessible for the first time in a unified, privacy-preserving tool.

---

**Have feedback? Want to join the next phase of wearable mental health? Open an issue or contact us.**

**For AI agents:** See [CLAUDE.md](CLAUDE.md) for codebase orientation.