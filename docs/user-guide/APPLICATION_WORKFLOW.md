# Big Mood Detector Application Workflow

## Overview

The Big Mood Detector analyzes your Apple Health data to predict risk of mood episodes (depression, mania, hypomania) using validated machine learning models. This guide explains how the application works from data upload through prediction generation.

## How It Works

### 1. Baseline Establishment (First Use)

When you first use the application, it needs to establish your personal baseline:

```bash
# Process your initial health data export
big-mood process ~/Desktop/apple_health_export/

# Generate baseline predictions
big-mood predict ~/Desktop/apple_health_export/ --report
```

**What happens:**
1. The system analyzes your historical data (ideally 30+ days)
2. Calculates your personal norms for:
   - Sleep duration and timing
   - Activity levels and patterns
   - Circadian rhythm markers
3. Stores these baselines for future comparison

**Important:** The first predictions may be less accurate as the system is still learning your patterns.

### 2. Ongoing Monitoring

After baseline establishment, you periodically upload new data:

```bash
# Process updated health export (e.g., weekly or monthly)
big-mood process ~/Desktop/apple_health_export_latest/

# Generate updated predictions with personal calibration
big-mood predict ~/Desktop/apple_health_export_latest/ --report
```

**What happens:**
1. New data is compared against your established baseline
2. Deviations from your normal patterns are calculated
3. Both models analyze the data:
   - **PAT Model**: Last 7 days of minute-by-minute activity
   - **XGBoost Model**: Last 30 days of sleep and circadian features
4. Predictions are calibrated to your personal patterns

### 3. Understanding the Predictions

The system provides three types of risk scores:

#### Depression Risk
- Based on patterns similar to PHQ-8 ≥ 10 (moderate depression)
- Key indicators: Delayed sleep phase, reduced activity, irregular patterns

#### Mania Risk
- Based on patterns similar to ASRM ≥ 6 (manic episode)
- Key indicators: Advanced sleep phase, very short sleep, erratic activity

#### Hypomania Risk
- Based on elevated mood patterns below full mania threshold
- Key indicators: Moderately reduced sleep, increased activity

### 4. Clinical Report Interpretation

Each prediction generates a clinical report:

```
CLINICAL DECISION SUPPORT (CDS) REPORT
==================================================

PATIENT DATA SUMMARY
Analysis Period: 30 days
Data Quality Score: 85.3%

CLINICAL RISK ASSESSMENT
------------------------------
Depression Risk: 72.5% [HIGH]
Hypomanic Risk: 15.2% [LOW]
Manic Risk: 8.3% [LOW]

KEY FINDINGS
------------------------------
• Circadian phase delayed by 2.3 hours from baseline
• Sleep duration reduced: 5.2h average (your normal: 7.5h)
• Activity levels 40% below your typical range
• Sleep efficiency decreased to 76%

CLINICAL RECOMMENDATIONS
------------------------------
⚠️ HIGH DEPRESSION RISK
• Schedule follow-up within 1 week
• Consider sleep hygiene intervention
• Monitor for worsening symptoms
• Review current treatment plan
```

## The Science Behind It

### Dual Model Approach

1. **XGBoost Model** (Seoul National University)
   - Analyzes 36 engineered features
   - 30-day analysis window
   - Best for detecting mania (98% accuracy)
   - Good for depression (80% accuracy)

2. **PAT Model** (Dartmouth)
   - Transformer neural network
   - Analyzes raw activity patterns
   - 7-day analysis window
   - Specialized for depression detection

3. **Ensemble Combination**
   - Weighs both models based on data quality
   - XGBoost: 60% weight (default)
   - PAT: 40% weight (default)
   - Adjusts based on available data

### Personal Calibration

The system adapts to YOUR normal:

```python
# Example: Sleep duration deviation
your_normal_sleep = 7.5  # hours (from your baseline)
current_sleep = 5.0      # hours (last night)
deviation = (current_sleep - your_normal_sleep) / your_std_dev
# Result: -2.5 standard deviations (significant change)
```

This personalization means:
- Athletes with naturally low heart rates won't trigger false alarms
- Night owls are compared to their own late schedule, not population average
- The system learns what's normal for YOU

## Data Privacy

- **Local Processing**: All analysis happens on your device
- **No Cloud Upload**: Your health data never leaves your computer
- **You Control Storage**: Baselines stored where you specify
- **Delete Anytime**: Remove stored baselines to reset

## Limitations

1. **Not a Diagnosis**: Provides risk assessment, not clinical diagnosis
2. **Requires Consistent Data**: Best with daily device wear
3. **30-Day Learning**: Full accuracy after establishing baseline
4. **Clinical Context**: Should be interpreted with professional guidance

## Optimizing Accuracy

For best results:

1. **Wear your device consistently** (especially during sleep)
2. **Export data regularly** (weekly or bi-weekly)
3. **Maintain device charge** (avoid data gaps)
4. **Update after major life changes** (travel, illness, med changes)

## Next Steps

- See [Interpreting Results](interpreting-results.md) for detailed score explanations
- Read [Clinical Integration](../clinical-integration/for-clinicians.md) for provider guidance
- Check [Privacy & Security](privacy-security.md) for data handling details