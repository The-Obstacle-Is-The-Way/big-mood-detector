# Big Mood Detector Documentation Overhaul Plan

## Executive Summary

This plan outlines a comprehensive documentation strategy to transform the Big Mood Detector repository into a professional, clinically-grounded resource that explains both the scientific foundations and practical implementation of our bipolar mood detection system.

## Current State Analysis

### Strengths
- Extensive clinical literature collection (14 research papers converted to markdown)
- Comprehensive mkdocs.yml structure already in place
- Good architectural documentation in `docs/architecture/`
- Detailed clinical requirements and dossiers

### Gaps
1. **Mathematical Foundation**: No clear documentation of the underlying math for:
   - XGBoost's 36 sleep/circadian features
   - PAT's transformer architecture and patch embeddings
   - Ensemble weight calculation and calibration
   - Personal baseline calculation algorithms

2. **Clinical Integration**: Missing practical guides on:
   - How baseline data establishes personal norms
   - How subsequent uploads trigger retroactive analysis
   - Clinical interpretation of risk scores
   - DSM-5/CANMAT threshold mappings

3. **User Journey**: No clear documentation of:
   - Initial setup and baseline establishment
   - Data upload workflow
   - Understanding prediction outputs
   - Privacy and data retention

4. **Technical Deep Dives**: Limited documentation on:
   - Feature engineering pipeline
   - Model integration architecture
   - Performance optimization strategies

## Proposed Documentation Structure

### 1. Getting Started (User-Focused)
```
docs/
├── getting-started/
│   ├── index.md                    # Overview and prerequisites
│   ├── installation.md             # Step-by-step setup
│   ├── quick-start.md              # 5-minute tutorial
│   ├── understanding-predictions.md # How to interpret results
│   └── privacy-security.md         # Data handling and privacy
```

### 2. Clinical Science Foundation
```
docs/
├── clinical-science/
│   ├── index.md                    # Overview of mood disorders
│   ├── bipolar-depression-continuum.md  # Clinical background
│   ├── digital-biomarkers.md      # What we measure and why
│   ├── clinical-thresholds.md     # DSM-5, CANMAT mappings
│   ├── baseline-importance.md     # Why personal baselines matter
│   └── literature-review.md       # Summary of key papers
```

### 3. Mathematical Models
```
docs/
├── models/
│   ├── index.md                    # Model overview and ensemble
│   ├── xgboost-features/
│   │   ├── sleep-features.md      # Sleep window calculations
│   │   ├── circadian-features.md  # DLMO, phase calculations
│   │   ├── activity-features.md   # Step counts, variance
│   │   └── feature-engineering.md # Complete feature list
│   ├── pat-transformer/
│   │   ├── architecture.md        # Transformer design
│   │   ├── patch-embeddings.md    # How we handle 10k+ minutes
│   │   ├── pretraining.md         # NHANES dataset details
│   │   └── fine-tuning.md         # Adaptation strategy
│   └── ensemble/
│       ├── orchestration.md       # How models combine
│       ├── confidence-scoring.md  # Calibration methods
│       └── clinical-routing.md    # Decision logic
```

### 4. Technical Implementation
```
docs/
├── technical/
│   ├── architecture/
│   │   ├── clean-architecture.md  # DDD principles
│   │   ├── data-flow.md          # End-to-end pipeline
│   │   └── api-design.md         # REST endpoints
│   ├── algorithms/
│   │   ├── sleep-merging.md      # 3.75h window algorithm
│   │   ├── baseline-updates.md   # Rolling statistics
│   │   ├── feature-extraction.md # Detailed algorithms
│   │   └── performance-tips.md   # Optimization guide
│   └── deployment/
│       ├── docker-setup.md       # Container deployment
│       ├── scaling.md            # Performance at scale
│       └── monitoring.md         # Observability
```

### 5. User Guides
```
docs/
├── user-guide/
│   ├── data-preparation/
│   │   ├── apple-health-export.md # Export instructions
│   │   ├── data-formats.md       # Supported formats
│   │   └── troubleshooting.md    # Common issues
│   ├── using-the-cli/
│   │   ├── process-command.md    # Feature extraction
│   │   ├── predict-command.md    # Generate predictions
│   │   ├── label-command.md      # Ground truth labeling
│   │   └── train-command.md      # Personal calibration
│   ├── using-the-api/
│   │   ├── authentication.md     # API access
│   │   ├── endpoints.md          # Complete reference
│   │   └── examples.md           # Code examples
│   └── interpreting-results/
│       ├── risk-scores.md        # Understanding outputs
│       ├── clinical-reports.md   # Report sections
│       └── feature-importance.md # What drives predictions
```

### 6. Developer Guide
```
docs/
├── developer/
│   ├── contributing.md            # How to contribute
│   ├── development-setup.md       # Dev environment
│   ├── testing-strategy.md        # Test organization
│   ├── code-standards.md          # Style guide
│   └── extending/
│       ├── adding-features.md    # New feature guide
│       ├── custom-models.md      # Model integration
│       └── plugin-system.md      # Future extensibility
```

### 7. Clinical Integration
```
docs/
├── clinical-integration/
│   ├── for-clinicians.md          # Overview for providers
│   ├── for-researchers.md        # Research applications
│   ├── validation-studies.md     # Clinical validation
│   ├── case-studies.md           # Example use cases
│   └── ethical-considerations.md # Responsible use
```

## Implementation Priority

### Phase 1: Core Documentation (Week 1-2)
1. **Mathematical Models** section - Critical for understanding
2. **Clinical Science Foundation** - Establishes credibility
3. Update existing **Getting Started** guides

### Phase 2: User Journey (Week 3)
1. **User Guides** - Complete workflow documentation
2. **Interpreting Results** - Critical for user understanding
3. Privacy and security documentation

### Phase 3: Technical Deep Dives (Week 4)
1. **Technical Implementation** details
2. **Developer Guide** for contributors
3. API reference updates

### Phase 4: Clinical Integration (Week 5)
1. **For Clinicians** guide
2. Validation studies summary
3. Ethical considerations

## Key Documentation Artifacts to Create

### 1. Mathematical Feature Reference Card
A single-page PDF/MD that lists all 36 XGBoost features with their formulas, clinical significance, and normal ranges.

### 2. Model Comparison Table
| Aspect | XGBoost | PAT | Ensemble |
|--------|---------|-----|----------|
| Window | 30 days | 7 days | Both |
| Features | 36 engineered | Raw activity | Combined |
| Best for | Mania (AUC 0.98) | Depression | Both |
| Latency | <100ms | ~1s | ~1s |

### 3. Clinical Workflow Diagram
Visual flow from data upload → processing → prediction → clinical action

### 4. Personal Baseline Explainer
Interactive diagram showing how baselines adapt over time and affect predictions

## Documentation Standards

### Code Examples
- Every feature should have Python code examples
- Use type hints and docstrings
- Show both CLI and API usage

### Clinical Accuracy
- All clinical claims must cite papers
- Use exact AUC/sensitivity/specificity from studies
- Include confidence intervals where available

### Visual Design
- Use mermaid diagrams for flows
- Include matplotlib visualizations for features
- Create example prediction reports

## Success Metrics

1. **Completeness**: 100% of features documented
2. **Accuracy**: All math/clinical claims verified against papers
3. **Usability**: New users can run predictions within 30 minutes
4. **Clarity**: Clinical users understand outputs without ML background

## Next Steps

1. Review and approve this plan
2. Create documentation sprint tickets
3. Assign writers to each section
4. Set up review process with clinical advisors
5. Plan user testing of documentation

## Appendix: Key Papers to Reference

1. **XGBoost-Mood** (Nature Digital Medicine 2024)
   - 36 features, AUC values, clinical cohort

2. **Pre-trained Actigraphy Transformer** (Dartmouth)
   - Architecture, pretraining, depression detection

3. **CANMAT Guidelines** (2018, 2021)
   - Clinical thresholds, treatment algorithms

4. **DSM-5-TR** (2022)
   - Diagnostic criteria, episode definitions

5. **Fitbit Bipolar Study** (Harvard 2024)
   - Real-world validation, digital biomarkers