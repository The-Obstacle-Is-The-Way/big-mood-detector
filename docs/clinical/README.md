# Clinical Documentation

> [← Back to main README](../../README.md)

Clinical validation, research foundation, and medical context for Big Mood Detector.

## 🏥 Quick Links

- **[Clinical Requirements](CLINICAL_REQUIREMENTS_DOCUMENT.md)** - Research foundation
- **[Clinical Dossier](CLINICAL_DOSSIER.md)** - DSM-5 criteria & thresholds
- **[Research Papers](../literature/)** - Original studies

## Performance Metrics

### Published Research Claims
| Condition | Model | AUC | Study |
|-----------|-------|-----|--------|
| **Mania** | XGBoost | 0.98 | Seoul National University |
| **Hypomania** | XGBoost | 0.95 | Seoul National University |
| **Depression** | XGBoost | 0.80 | Seoul National University |
| **Depression** | PAT | 0.589 | Dartmouth (paper) |

### Our Implementation
| Model | Task | AUC | Status |
|-------|------|-----|---------|
| PAT-S | Depression | 0.56 | ✅ Matches paper |
| PAT-M | Depression | 0.54 | ✅ Complete |
| PAT-L | Depression | TBD | 🔄 Training |
| XGBoost | All conditions | Paper claims | ⚠️ No independent validation |

## Clinical Context

### What This Means
- **0.56 AUC** = Barely better than random (0.5)
- **0.80 AUC** = Moderate predictive ability
- **0.98 AUC** = Excellent (almost too good?)

### Temporal Windows Matter
- **PAT**: Assesses current state from past 7 days
- **XGBoost**: Predicts next 24 hours
- **Why separate**: Treatment decisions differ for "now" vs "tomorrow"

### Key Biomarkers

**Strongest Predictors (from Seoul study):**
1. **Circadian phase shift** (Z-score)
2. **Sleep duration deviation**
3. **Activity fragmentation**
4. **Heart rate variability**

**DSM-5 Thresholds:**
- Depression: PHQ-9 ≥ 10
- Mania: YMRS ≥ 12
- Hypomania: YMRS 8-11

## Critical Limitations

### Population Mismatch
- **XGBoost**: Korean patients (n=168)
- **PAT**: US general population (n=29,307)
- **Your data**: Individual variation

### Not Validated
- Models from papers, not independently validated
- No prospective clinical trials
- Not FDA approved

### Clinical Reality
- Digital biomarkers supplement, don't replace clinical assessment
- Many factors affect mood beyond activity/sleep
- Individual calibration essential

## Research Foundation

### Primary Papers
1. **Seoul XGBoost Study** (2024)
   - Nature Digital Medicine
   - 168 patients, 44,787 days
   - [Full paper](../literature/converted_markdown/xgboost-mood/xgboost-mood.md)

2. **Dartmouth PAT Study** (2024)
   - Pretrained Actigraphy Transformer
   - NHANES 2013-2014 cohort
   - [Full paper](../literature/converted_markdown/pretrained-actigraphy-transformer/pretrained-actigraphy-transformer.md)

## For Researchers

### What We Need
1. **Independent validation** on diverse populations
2. **Prospective trials** (not just retrospective)
3. **Clinical integration** studies
4. **Adverse event** monitoring

### Contributing
- Share anonymized validation results
- Collaborate on clinical trials
- Improve model generalization
- Add new biomarkers

## Medical Disclaimer

**This is research software only.**
- NOT a medical device
- NOT FDA approved
- CANNOT diagnose conditions
- CANNOT replace clinical care

**If experiencing mental health crisis:**
- US: Call 988
- UK: Call 116 123
- Emergency: 911/999/112

---

*For technical details, see [Developer Documentation](../developer/README.md)*  
*For usage guides, see [User Documentation](../user/README.md)*