# 📚 Big Mood Detector Documentation

Welcome to the comprehensive documentation for Big Mood Detector - a clinical-grade bipolar mood prediction system that analyzes Apple Health data using validated machine learning models to predict risk of mood episodes.

## 🎯 Clinical Accuracy

Based on peer-reviewed research with 168 patients and 44,787 observation days:

| Condition | Model | Accuracy (AUC) | Key Reference |
|-----------|-------|----------------|---------------|
| **Mania** | XGBoost | **0.98** ⭐ | Nature Digital Medicine 2024 |
| **Hypomania** | XGBoost | **0.95** | Nature Digital Medicine 2024 |
| **Depression** | XGBoost | **0.80** | Nature Digital Medicine 2024 |
| **Depression** | PAT | **0.77** | Dartmouth Study 2024 |

The system combines both models for optimal performance across all mood states.

## 🆕 New Documentation Highlights

### Essential Reading
- **[Application Workflow](user-guide/APPLICATION_WORKFLOW.md)** ⭐ - Understand how the system works from baseline to predictions
- **[XGBoost Feature Reference](models/xgboost-features/FEATURE_REFERENCE.md)** ⭐ - All 36 features explained with formulas
- **[Ensemble Mathematics](models/ensemble/ENSEMBLE_MATHEMATICS.md)** - How predictions combine mathematically
- **[Documentation Plan](DOCUMENTATION_PLAN.md)** - Our comprehensive documentation strategy

## 🗺️ Documentation Structure

### 👥 [User Documentation](./user/)
**For users who want to analyze their health data**

- **[Quick Start Guide](./user/QUICK_START_GUIDE.md)** ⭐ - Get running in 5 minutes
- **[Application Workflow](user-guide/APPLICATION_WORKFLOW.md)** ⭐ - How it works end-to-end
- **[Advanced Usage](./user/ADVANCED_USAGE.md)** - Power user features
- **[Apple Health Export Guide](./user/APPLE_HEALTH_EXPORT.md)** - How to export your data
- **[User Guide Overview](./user/README.md)** - Complete user documentation index

### 🧮 [Model Documentation](./models/)
**Deep dive into the ML models**

- **[XGBoost Features](./models/xgboost-features/FEATURE_REFERENCE.md)** ⭐ - All 36 features with formulas
- **[Ensemble Mathematics](./models/ensemble/ENSEMBLE_MATHEMATICS.md)** - Mathematical foundations
- **[PAT Architecture](./models/pat-transformer/)** - Transformer model details (coming soon)

### 🏥 [Clinical Documentation](./clinical/)
**Clinical knowledge and validation**

- **[Clinical Dossier](./clinical/CLINICAL_DOSSIER.md)** - DSM-5 criteria and thresholds
- **[Clinical Requirements](./clinical/CLINICAL_REQUIREMENTS_DOCUMENT.md)** - Research foundation

### 👩‍💻 [Developer Documentation](./developer/)
**Technical documentation for developers**

- **[Architecture Overview](./developer/ARCHITECTURE_OVERVIEW.md)** ⭐ - System design
- **[API Reference](./developer/API_REFERENCE.md)** - REST API endpoints
- **[Deployment Guide](./developer/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[Model Weight Architecture](./developer/MODEL_WEIGHT_ARCHITECTURE.md)** - ML model management
- **[Dual Pipeline Architecture](./developer/DUAL_PIPELINE_ARCHITECTURE.md)** - JSON/XML processing
- **[Data Dossier](./developer/DATA_DOSSIER.md)** - Data structures
- **[Model Integration Guide](./developer/model_integration_guide.md)** - Adding new models
- **[Git Workflow](./developer/GIT_WORKFLOW.md)** - Development process
- **[Security](./developer/SECURITY.md)** - Security considerations

### ✅ [Completed Work](./completed/)
**Implementation records and sprint documentation**

Records of completed features, validations, and implementation decisions.

### 📚 [Literature](./literature/)
**Research papers and clinical references**

- **[Clinical References to Read](./literature/clinical_references_to_read.md)** - Curated bibliography
- **[Converted Papers](./literature/converted_markdown/)** - Key papers in markdown format
- **[Original PDFs](./literature/pdf/)** - Original research papers

### 🗄️ [Archive](./archive/)
**Historical documentation**

Outdated or superseded documentation kept for reference.

## 🚀 Quick Navigation

### For New Users
1. Start with [Quick Start Guide](./user/QUICK_START_GUIDE.md)
2. Export your data using [Apple Health Export Guide](./user/APPLE_HEALTH_EXPORT.md)
3. Run your first prediction

### For Developers
1. Read [Architecture Overview](./developer/ARCHITECTURE_OVERVIEW.md)
2. Set up development environment
3. Check [API Reference](./developer/API_REFERENCE.md) for integration

### For Clinicians
1. Review [Clinical Dossier](./clinical/CLINICAL_DOSSIER.md)
2. Understand validation in [Clinical Requirements](./clinical/CLINICAL_REQUIREMENTS_DOCUMENT.md)
3. Browse [Research Literature](./literature/)

## 📖 Documentation Standards

### User Documentation
- Step-by-step instructions with examples
- Screenshots where helpful
- Common troubleshooting tips
- No technical jargon

### Clinical Documentation
- DSM-5 references and citations
- Peer-reviewed research backing
- Clinical threshold explanations
- Risk stratification guidelines

### Developer Documentation
- Code examples and API samples
- Architecture diagrams
- Performance benchmarks
- Security considerations

## 🔍 Finding Information

### By Topic
- **Installation**: [Quick Start Guide](./user/QUICK_START_GUIDE.md)
- **Commands**: [Quick Start Guide](./user/QUICK_START_GUIDE.md#-commands)
- **API Integration**: [API Reference](./developer/API_REFERENCE.md)
- **ML Models**: [Model Weight Architecture](./developer/MODEL_WEIGHT_ARCHITECTURE.md)
- **Clinical Thresholds**: [Clinical Dossier](./clinical/CLINICAL_DOSSIER.md)
- **Deployment**: [Deployment Guide](./developer/DEPLOYMENT_GUIDE.md)

### By Role
- **End User**: Start in [user/](./user/)
- **Developer**: Focus on [developer/](./developer/)
- **Researcher**: Check [clinical/](./clinical/) and research papers
- **DevOps**: See [Deployment Guide](./developer/DEPLOYMENT_GUIDE.md)

## 🤝 Contributing to Documentation

When adding or updating documentation:

1. **Keep it current** - Update docs with code changes
2. **Use examples** - Show, don't just tell
3. **Be concise** - Get to the point quickly
4. **Cross-reference** - Link related documents
5. **Test commands** - Ensure all examples work

## 📊 Documentation Coverage

- ✅ User guides with examples
- ✅ Complete API documentation
- ✅ Architecture overview
- ✅ Clinical validation
- ✅ Deployment instructions
- 🚧 Video tutorials (coming soon)
- 🚧 Interactive demos (planned)

## 🔗 External Resources

- **Research Papers**: See [literature/](./literature/) directory
- **Reference Implementations**: Check [reference_repos/](../reference_repos/)
- **GitHub Issues**: Report problems or suggest improvements
- **Community**: Join discussions on GitHub

---

*Documentation last updated: 2025-07-20*

*For AI agents working with this codebase, see [CLAUDE.md](../CLAUDE.md) and [repo_map.json](../repo_map.json)*