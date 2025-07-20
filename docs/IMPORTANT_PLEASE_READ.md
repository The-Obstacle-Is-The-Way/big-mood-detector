# 🚨 IMPORTANT: Understanding Big Mood Detector - Please Read First

## ⚠️ Medical Disclaimers

**CRITICAL SAFETY INFORMATION:**

1. **This application is for RESEARCH and PERSONAL USE ONLY**
2. **This is NOT a medical device and has NOT been approved by the FDA**
3. **This application CANNOT diagnose mental health conditions**
4. **ALWAYS consult with qualified healthcare professionals for medical advice**
5. **If you are experiencing a mental health crisis, IMMEDIATELY seek professional help**

**Emergency Resources:**
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 911

**Research Status:**
- This application is based on peer-reviewed machine learning papers from Nature Digital Medicine, Harvard, and Dartmouth
- However, **this specific application has NOT been clinically validated**
- It has NOT been tested on real patients in a controlled clinical setting
- Exercise extreme caution and use only as a supplementary tool

## 🎯 Why This Application Matters

### The Clinical Problem

As a clinical psychiatrist knows, differentiating between these conditions is one of the hardest challenges in mental health:

1. **Unipolar Depression** (Major Depressive Disorder)
2. **Bipolar Depression** (depressive phase of bipolar disorder)
3. **Borderline Personality Disorder** (with mood symptoms)

**Why is this differentiation critical?**
- Wrong diagnosis → Wrong treatment → Potential harm
- Antidepressants alone in bipolar disorder can trigger mania
- Each condition requires different treatment approaches

**This is the FIRST application in the world** that theoretically could help differentiate these conditions using passive wearable data.

## 📋 What You Actually Need to Use This Application

### Good News: NO Labeling Required!

**Both models work out-of-the-box:**

1. **XGBoost Model**
   - Pre-trained on 168 clinical patients
   - 235 psychiatrist-labeled episodes
   - NO user labels needed
   - Just needs 30-60 days of your sleep/activity data

2. **PAT Transformer**
   - Pre-trained on 29,307 participants
   - Foundation model approach
   - NO user labels needed
   - Works with as little as 7 days of data

### What You DO Need:

1. **Apple Health Data Export** (or compatible format)
   - Minimum 30 days for initial predictions
   - 60+ days for optimal accuracy
   - Consistent device wear (especially during sleep)

2. **Time to Establish Baseline**
   - First 30 days: System learns YOUR normal patterns
   - After 30 days: Predictions become personalized
   - After 60 days: Maximum accuracy achieved

3. **Regular Data Updates**
   - Export and process new data weekly or bi-weekly
   - System improves with more data

### Optional: Labeling for Enhanced Accuracy

While NOT required, you CAN label past mood episodes to:
- Improve personal calibration
- Validate model accuracy
- Contribute to research

## 🔮 Forecasting Capabilities

Based on the research literature:

### XGBoost Model (Nature Digital Medicine 2024)
- **Prediction Window**: Next-day (24 hours ahead)
- **What it predicts**: Tomorrow's mood state based on past 30 days
- **Best for**: Mania detection (98% accuracy)

### PAT Model (Dartmouth 2024)
- **Prediction Window**: Current state analysis
- **What it predicts**: Depression risk based on last 7 days
- **Best for**: Depression detection

### Combined System
- **Continuous Monitoring**: Daily risk assessments
- **Early Warning**: 1-2 days before clinical symptoms
- **Trend Analysis**: Identifies deteriorating patterns

## 🚀 How to Use This Application

### Step 1: Initial Setup (One Time)
```bash
# Install the application
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector
pip install -e ".[dev,ml,monitoring]"
```

### Step 2: Establish Your Baseline (First Month)
```bash
# Process your historical Apple Health data
python src/big_mood_detector/main.py process ~/Desktop/apple_health_export/

# Generate initial predictions (may be less accurate initially)
python src/big_mood_detector/main.py predict ~/Desktop/apple_health_export/ --report
```

### Step 3: Ongoing Monitoring (Weekly/Bi-weekly)
```bash
# Export new Apple Health data
# Process the update
python src/big_mood_detector/main.py process ~/Desktop/apple_health_export_new/

# Get updated predictions (now personalized to you)
python src/big_mood_detector/main.py predict ~/Desktop/apple_health_export_new/ --report
```

### Step 4: Optional - Add Labels for Better Accuracy
```bash
# If you know when you had mood episodes
python src/big_mood_detector/main.py label episode \
    --episode-type depressive \
    --severity moderate \
    --start-date 2024-01-15 \
    --end-date 2024-01-22
```

## 📊 Understanding Your Results

### Risk Levels
- **LOW** (<30%): Continue monitoring
- **MODERATE** (30-50%): Increased vigilance, consider checking in with provider
- **HIGH** (50-70%): Schedule appointment with provider soon
- **VERY HIGH** (>70%): Seek clinical evaluation promptly

### What the Predictions Mean
- These are **RISK ASSESSMENTS**, not diagnoses
- Based on patterns similar to clinically validated episodes
- Should be discussed with your healthcare provider

## 🔬 Current Limitations

1. **Not Validated on This Specific Implementation**
   - Papers are peer-reviewed, but this code is new
   - Needs real-world validation studies

2. **Requires Consistent Data**
   - Gaps in data reduce accuracy
   - Device must be worn during sleep

3. **Individual Variability**
   - Works better for some people than others
   - Accuracy improves over time with your data

4. **Cannot Replace Clinical Judgment**
   - Always secondary to professional assessment
   - Not for emergency situations

## 🤝 Call for Contributors

This is an **active research project** and we need help!

### How You Can Contribute:
1. **Developers**: Improve code, add features, fix bugs
2. **Researchers**: Validate models, publish studies
3. **Clinicians**: Provide feedback, suggest improvements
4. **Users**: Report issues, share experiences (anonymously)

### Contact
- GitHub Issues: Report bugs or suggest features
- Research Collaborations: See CONTRIBUTING.md

## 🔑 Key Takeaways

1. **NO LABELS REQUIRED** - Works out-of-the-box
2. **Predictions start after 30 days** of baseline data
3. **Best accuracy after 60 days** of consistent use
4. **Forecasts 1-2 days ahead** for mood episodes
5. **ALWAYS consult healthcare providers** for medical decisions

## 📚 Further Reading

- [Clinical Science Behind the Models](clinical/CLINICAL_DOSSIER.md)
- [Technical Architecture](developer/ARCHITECTURE_OVERVIEW.md)
- [Research Papers](literature/)

---

**Remember**: This tool is meant to SUPPLEMENT, not REPLACE, professional mental healthcare. Your safety and wellbeing come first.

*Last Updated: 2025-07-20*