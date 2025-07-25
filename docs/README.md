# Big Mood Detector Documentation

Documentation for the temporal mood prediction system.

## Quick Links

### For Users
- [Quick Start Guide](user/QUICK_START_GUIDE.md) - Get running in 5 minutes
- [Apple Health Export](user/APPLE_HEALTH_EXPORT.md) - How to export your data
- [Understanding Reports](user-guide/APPLICATION_WORKFLOW.md) - What the predictions mean

### For Developers  
- [Architecture Overview](developer/ARCHITECTURE_OVERVIEW.md) - System design
- [API Reference](developer/API_REFERENCE.md) - REST endpoints
- [Deployment Guide](developer/DEPLOYMENT_GUIDE.md) - Production setup

### For Researchers
- [Clinical Requirements](clinical/CLINICAL_REQUIREMENTS_DOCUMENT.md) - Research foundation
- [Feature Reference](models/xgboost-features/FEATURE_REFERENCE.md) - All 36 features explained
- [Literature](literature/) - Research papers

## Documentation Structure

```
docs/
├── user/              # End-user guides
├── developer/         # Technical documentation
├── clinical/          # Clinical validation
├── models/            # ML model details
├── api/               # API specifications
└── literature/        # Research papers
```

## Key Documents

### Understanding the System
- [Application Workflow](user-guide/APPLICATION_WORKFLOW.md) - How everything fits together
- [Ensemble Mathematics](models/ensemble/ENSEMBLE_MATHEMATICS.md) - The temporal prediction approach

### Technical Deep Dives
- [Dual Pipeline Architecture](developer/DUAL_PIPELINE_ARCHITECTURE.md) - JSON/XML processing
- [Model Weight Architecture](developer/MODEL_WEIGHT_ARCHITECTURE.md) - ML model management
- [Performance Optimization](performance/OPTIMIZATION_TRACKING.md) - Speed improvements

### Clinical Context
- [Clinical Dossier](clinical/CLINICAL_DOSSIER.md) - DSM-5 criteria and thresholds
- [XGBoost Paper](literature/converted_markdown/xgboost-mood/xgboost-mood.md) - Seoul study
- [PAT Paper](literature/converted_markdown/pretrained-actigraphy-transformer/pretrained-actigraphy-transformer.md) - Dartmouth transformer

## Contributing to Docs

When adding documentation:
1. Keep it factual and accurate
2. Include code examples where relevant
3. Update this index
4. Test all commands/examples

## Version

Documentation for Big Mood Detector v0.4.1

## Recent Updates

### Training Documentation
- [PAT-Conv-L Achievement](training/PAT_CONV_L_ACHIEVEMENT.md) - Best model (0.5929 AUC)
- [Training Summary](training/TRAINING_SUMMARY.md) - Current model status
- [Training Output Structure](training/TRAINING_OUTPUT_STRUCTURE.md) - Where files go

### Setup & Deployment
- [Setup Guide](setup/SETUP_GUIDE.md) - Installation instructions
- [Docker Setup](setup/DOCKER_SETUP_GUIDE.md) - Container deployment
- [Deployment Readiness](deployment/DEPLOYMENT_READINESS.md) - Production checklist