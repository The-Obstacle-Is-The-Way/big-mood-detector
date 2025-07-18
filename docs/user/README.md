# 📱 Big Mood Detector - User Documentation

Welcome to Big Mood Detector! This clinical-grade tool analyzes your Apple Health data to detect patterns associated with mood episodes using validated ML models.

## 🎯 What Can This Tool Do?

- **Predict Mood Episodes**: Detect risk of depression, mania, and hypomania
- **Analyze Sleep Patterns**: Identify circadian rhythm disruptions
- **Track Activity Levels**: Monitor physical activity and its relationship to mood
- **Personal Calibration**: Fine-tune predictions based on your individual patterns
- **Clinical Reports**: Generate detailed reports for healthcare providers

## 📚 Documentation Overview

### Getting Started
- **[Quick Start Guide](./QUICK_START_GUIDE.md)** - Get up and running in 5 minutes
- **[Apple Health Export Guide](./APPLE_HEALTH_EXPORT.md)** - How to export your health data
- **[Advanced Usage](./ADVANCED_USAGE.md)** - Power user features and workflows

### Understanding Your Results
- **[Interpreting Predictions](./INTERPRETING_PREDICTIONS.md)** - What the risk scores mean
- **[Clinical Thresholds](./CLINICAL_THRESHOLDS.md)** - Evidence-based cutoffs used

### Data and Privacy
- **[Data Requirements](./DATA_REQUIREMENTS.md)** - What data is needed and why
- **[Privacy Guide](./PRIVACY_GUIDE.md)** - How your data is protected

## 🚀 Quick Example

```bash
# 1. Process your health data
python src/big_mood_detector/main.py process data/health_auto_export/

# 2. Get predictions with a report
python src/big_mood_detector/main.py predict data/health_auto_export/ --report

# 3. Start monitoring for new data
python src/big_mood_detector/main.py watch data/health_auto_export/
```

## 🏥 Clinical Foundation

This tool is based on peer-reviewed research from:
- Seoul National University (Nature Digital Medicine, 2024)
- Harvard Medical School (Bipolar Disorders, 2024)
- Dartmouth College (PAT Transformer)

It uses:
- **36 sleep and circadian features** validated in clinical studies
- **XGBoost models** with AUC 0.80-0.98 for mood prediction
- **Transformer models** for activity pattern analysis
- **DSM-5 aligned** clinical thresholds

## ⚠️ Important Disclaimers

1. **Not a Diagnostic Tool**: This provides risk assessments, not diagnoses
2. **Consult Healthcare Providers**: Always discuss results with qualified professionals
3. **Individual Variability**: Predictions improve with personal calibration over time
4. **Data Quality Matters**: Accuracy depends on consistent device usage

## 🆘 Getting Help

- **Installation Issues**: See [Troubleshooting](./TROUBLESHOOTING.md)
- **Understanding Features**: Check [Technical Glossary](./GLOSSARY.md)
- **Clinical Questions**: Review [Clinical FAQ](./CLINICAL_FAQ.md)
- **Bug Reports**: File issues on GitHub

## 📊 Typical Use Cases

### Personal Health Monitoring
Track your mood patterns over time to identify triggers and early warning signs.

### Clinical Support
Share reports with your healthcare provider for data-driven treatment decisions.

### Research Participation
Contribute anonymized data to advance mental health research (with consent).

### Family Care
Monitor loved ones' patterns (with their permission) to provide timely support.

## 🔒 Your Data, Your Control

- All processing happens locally on your device
- No data is sent to external servers by default
- You control what data is analyzed and shared
- Encryption available for stored results

## 📈 What's Next?

1. Start with the [Quick Start Guide](./QUICK_START_GUIDE.md)
2. Export your Apple Health data
3. Run your first prediction
4. Review the results with your healthcare provider

---

*Big Mood Detector - Advancing mental health through responsible AI*