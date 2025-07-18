# Big Mood Detector

<div align="center">
  <img src="assets/apple-health-icon.jpeg" alt="Big Mood Detector Logo" width="200">
  
  **Clinical-grade mood episode detection from wearable data**
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
  [![Tests](https://img.shields.io/badge/tests-695%20passing-brightgreen.svg)](tests/)
  [![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)](tests/)
</div>

## 🎯 Overview

Big Mood Detector is a production-ready system that analyzes Apple Health and wearable data to detect mood episodes (depression, hypomania, mania) in individuals with bipolar disorder. It combines state-of-the-art machine learning models with clinical guidelines from DSM-5.

## ✨ Key Features

- **🏥 Clinical-Grade Detection**: Based on peer-reviewed research from Seoul National University Hospital and Dartmouth College
- **🤖 Dual ML Models**: XGBoost (primary) + PAT Transformer (ensemble) for robust predictions
- **📱 Apple Health Integration**: Seamless processing of Health Export data (sleep, activity, heart rate)
- **⚡ Production Ready**: Fast streaming parser handles 500MB+ files in seconds
- **🔒 Privacy First**: All processing happens locally - your health data never leaves your device
- **🏷️ Labeling System**: Built-in tools for clinicians to validate and improve predictions

## 🚀 Quick Start

```bash
# Install
pip install big-mood-detector

# Process health data
mood process /path/to/apple_health_export.zip

# Make predictions
mood predict /path/to/processed_data.json

# Start API server
mood serve
```

## 📊 Performance

- **Accuracy**: AUC 0.80-0.98 across episode types
- **Speed**: Processes 1 year of data in < 1 second
- **Memory**: Streaming parser uses < 100MB RAM for any file size
- **Models**: Pretrained on NHANES data, fine-tunable with personal data

## 🏗️ Architecture

The system follows Clean Architecture principles with clear separation of concerns:

```
Domain Layer (Pure Business Logic)
    ↑
Application Layer (Use Cases)
    ↑
Infrastructure Layer (Data, ML, APIs)
    ↑
Interface Layer (CLI, API, Web)
```

## 📚 Documentation

- [Quick Start Guide](user/QUICK_START_GUIDE.md) - Get up and running in 5 minutes
- [Architecture Overview](developer/ARCHITECTURE_OVERVIEW.md) - Understand the system design
- [API Reference](developer/API_REFERENCE.md) - Complete API documentation
- [Clinical Documentation](clinical/CLINICAL_REQUIREMENTS_DOCUMENT.md) - Medical context and validation

## 🔬 Research Foundation

This project implements findings from:

1. **Seoul National University Hospital Study** (2024)
   - "Accurately predicting mood episodes using wearable sleep and circadian rhythm features"
   - npj Digital Medicine
   - Provides XGBoost model architecture and 36 key features

2. **Dartmouth PAT Model** (2024)
   - "AI Foundation Models for Wearable Movement Data"
   - Pretrained transformer for activity sequences
   - NHANES population-level training

## 🛡️ Privacy & Security

- **Local Processing**: All analysis happens on your device
- **No Cloud Dependencies**: Works completely offline
- **Data Control**: You own and control all your data
- **Open Source**: Fully auditable codebase

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](license.md) file for details.

## 🙏 Acknowledgments

- Seoul National University Hospital for the XGBoost research
- Dartmouth College for the PAT model
- The open source community for amazing tools

## ⚠️ Medical Disclaimer

This software is for research and educational purposes only. It is not intended to diagnose, treat, cure, or prevent any disease. Always consult with qualified healthcare professionals for medical advice.

---

<div align="center">
  Made with ❤️ for the mental health community
</div>