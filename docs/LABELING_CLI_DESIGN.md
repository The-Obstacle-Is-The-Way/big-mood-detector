# Big Mood Detector - Labeling CLI Design Document

## Executive Summary

The labeling CLI is a critical component that enables continuous learning and personalization of mood prediction models. This document outlines a user-friendly, clinically-informed design that leverages existing predictions to guide efficient ground-truth collection.

## Design Philosophy

### Core Principles
1. **Prediction-Guided**: Use existing model predictions to prioritize high-value labeling opportunities
2. **Clinical Alignment**: Validate all labels against DSM-5 criteria
3. **User-Friendly**: Minimize cognitive load through intelligent defaults and clear guidance
4. **Incremental**: Support resumable, partial labeling sessions
5. **Privacy-First**: All data remains local, no PHI transmission

### Key Innovation: Prediction-Assisted Labeling
Rather than asking users to label every day, the system will:
- Show model predictions with confidence scores
- Prioritize days with high-risk predictions or model uncertainty
- Provide clinical context (sleep patterns, activity levels) to aid recall
- Validate episode duration against DSM-5 requirements

## User Interface Design

### Command Structure
```bash
# Basic usage - interactive mode
bmd label --predictions output/predictions.json

# Advanced usage with filters
bmd label --predictions output/predictions.json \
         --start 2024-01-01 \
         --end 2024-06-30 \
         --threshold 0.4 \
         --output labels/ground_truth.csv
```

### Interactive Flow

#### 1. Session Initialization
```
╔══════════════════════════════════════════════════════════════╗
║                  Big Mood Detector Labeling Tool              ║
║                           Version 1.0.0                        ║
╚══════════════════════════════════════════════════════════════╝

Loading predictions from: output/predictions.json
Found 180 days of predictions (2024-01-01 to 2024-06-30)

Previous labels found: labels/ground_truth.csv (45 labeled)
Resuming from last session...

Filtering strategy: High-risk days (≥40% any risk)
Days to label: 32 (18 depression, 8 hypomanic, 6 mixed signals)
```

#### 2. Daily Labeling Interface
```
┌─────────────────────────────────────────────────────────────┐
│ Date: 2024-03-15 (Friday)                    [Day 12 of 32] │
├─────────────────────────────────────────────────────────────┤
│ Model Predictions:                                          │
│   • Depression Risk: 72% [HIGH] ⚠️                          │
│   • Hypomanic Risk: 12% [LOW]                              │
│   • Manic Risk: 8% [LOW]                                   │
│                                                             │
│ Digital Biomarkers:                                         │
│   • Sleep: 3.2 hrs (↓ 4.8 hrs from baseline)              │
│   • Activity: 2,341 steps (↓ 68% from baseline)           │
│   • Sleep Efficiency: 42% (poor)                           │
│   • Circadian Phase: +2.3 hrs delayed                      │
├─────────────────────────────────────────────────────────────┤
│ What was your mood state on this day?                      │
│                                                             │
│   [1] Depressed                                             │
│   [2] Hypomanic                                             │
│   [3] Manic                                                 │
│   [4] Mixed (both depression + mania symptoms)             │
│   [5] Stable/Normal                                         │
│   [6] Don't remember / Skip                                 │
│   [7] View more details                                     │
│                                                             │
│ Choice (1-7): _                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3. Severity Assessment (if episode selected)
```
┌─────────────────────────────────────────────────────────────┐
│ Depression Episode Severity                                 │
├─────────────────────────────────────────────────────────────┤
│ How severe was your depression on 2024-03-15?              │
│                                                             │
│   [1] Mild - Some symptoms, minimal impairment             │
│   [2] Moderate - Clear symptoms, moderate impairment       │
│   [3] Severe - Many symptoms, significant impairment       │
│   [4] Very Severe - Most symptoms, major impairment        │
│                                                             │
│ Clinical Context:                                           │
│   • PHQ-9 equivalent: Moderate = 10-14, Severe = 15-19    │
│   • Consider: work/social function, self-care, energy      │
│                                                             │
│ Severity (1-4): _                                           │
└─────────────────────────────────────────────────────────────┘
```

#### 4. Episode Span Detection
```
┌─────────────────────────────────────────────────────────────┐
│ Episode Duration Check                                      │
├─────────────────────────────────────────────────────────────┤
│ You marked 2024-03-15 as "Depressed (Moderate)"           │
│                                                             │
│ Was this part of a longer episode?                         │
│                                                             │
│   [1] Single day only                                      │
│   [2] Part of ongoing episode                              │
│   [3] Don't know                                           │
│                                                             │
│ Choice (1-3): 2                                             │
│                                                             │
│ Episode Start Date [2024-03-10]: _                         │
│ Episode End Date [2024-03-20]: _                           │
│                                                             │
│ DSM-5 Note: Depression episodes require ≥14 days          │
└─────────────────────────────────────────────────────────────┘
```

#### 5. Optional Clinical Notes
```
┌─────────────────────────────────────────────────────────────┐
│ Additional Notes (Optional)                                 │
├─────────────────────────────────────────────────────────────┤
│ Any additional context about this period?                   │
│ Examples: medication changes, life events, symptoms         │
│                                                             │
│ Notes: Started new job, high stress, stopped medication    │
│ _                                                           │
│                                                             │
│ Press Enter to continue or type notes...                   │
└─────────────────────────────────────────────────────────────┘
```

### Progress Tracking
```
┌─────────────────────────────────────────────────────────────┐
│ Labeling Progress                                           │
├─────────────────────────────────────────────────────────────┤
│ Current Session:                                            │
│   ■■■■■■■■□□□□□□□□□□□□ 40% (12/30 days)                   │
│                                                             │
│ Overall Progress:                                           │
│   ■■■■■■■■■■■■■□□□□□□□ 65% (117/180 days)                 │
│                                                             │
│ Episodes Identified:                                        │
│   • Depression: 3 episodes (42 days total)                 │
│   • Hypomanic: 2 episodes (8 days total)                   │
│   • Stable: 67 days                                        │
│                                                             │
│ [C]ontinue  [B]reak  [S]ave & Exit                        │
└─────────────────────────────────────────────────────────────┘
```

## Data Model

### Label Schema (CSV Format)
```csv
date,label,severity,confidence,start_date,end_date,duration_days,notes,labeled_at,model_agreed
2024-03-15,depression,moderate,high,2024-03-10,2024-03-20,11,Work stress,2024-12-15T10:30:00,false
2024-03-16,depression,moderate,high,2024-03-10,2024-03-20,11,Work stress,2024-12-15T10:31:00,false
2024-04-01,stable,none,high,2024-04-01,2024-04-01,1,,2024-12-15T10:32:00,true
```

### Integration with EpisodeLabeler
```python
# Under the hood, the CLI uses the existing EpisodeLabeler
labeler = EpisodeLabeler()

# For single day
labeler.add_episode(date="2024-03-15", episode_type="depressive", severity=2)

# For episode spans
labeler.add_episode(
    start_date="2024-03-10",
    end_date="2024-03-20", 
    episode_type="depressive",
    severity=2,
    notes="Work stress"
)

# For baseline periods
labeler.add_baseline(
    start_date="2024-04-01",
    end_date="2024-04-30",
    notes="Stable on medication"
)
```

## Intelligent Features

### 1. Active Learning Integration
- Prioritize days where model confidence is low (0.4 < risk < 0.6)
- Focus on boundary cases that would most improve the model
- Track model agreement to identify systematic errors

### 2. Context-Aware Prompting
- Show sleep/activity patterns around the target date
- Highlight significant deviations from personal baseline
- Display medication adherence if available

### 3. Episode Continuity Detection
- Automatically suggest episode spans based on consecutive high-risk days
- Validate against DSM-5 duration requirements
- Merge adjacent episodes of the same type

### 4. Smart Defaults
- Pre-select likely label based on model prediction
- Suggest episode boundaries based on risk patterns
- Auto-save every 10 labels

## Implementation Architecture

### Module Structure
```
src/big_mood_detector/cli/
├── label.py              # Main CLI command
├── labeling/
│   ├── __init__.py
│   ├── session.py        # Labeling session management
│   ├── ui.py             # Terminal UI components
│   ├── validator.py      # DSM-5 validation logic
│   └── persistence.py    # Save/load progress
```

### Key Dependencies
- `click`: CLI framework (already used)
- `rich`: Enhanced terminal UI with colors and tables
- `prompt_toolkit`: Advanced input handling
- Built-in `csv` module for data persistence

### Testing Strategy
```python
# Mock stdin for automated testing
def test_label_single_day(mock_stdin):
    mock_stdin.return_value = ["1", "2", "N", ""]  # Depression, Moderate, No span
    result = run_labeling_session(date="2024-03-15")
    assert result.label == "depression"
    assert result.severity == "moderate"
```

## Privacy & Security

### Data Handling
- All labels stored locally in user-specified directory
- No network calls or external data transmission
- Sensitive notes encrypted at rest (optional)
- CSV files use standard permissions

### PHI Considerations
- No patient identifiers in label files
- Dates are the only temporal reference
- Notes field sanitized of names/locations
- Export includes privacy disclaimer

## Future Enhancements

### Phase 2: Enhanced Context
- Integrate medication tracking
- Show therapy session dates
- Display life event markers
- Include social rhythm data

### Phase 3: Clinical Integration
- Export to EHR-compatible formats
- Generate clinical reports
- Support multi-clinician review
- Implement inter-rater reliability

### Phase 4: Advanced ML
- Real-time model retraining
- Personalized threshold adjustment
- Uncertainty quantification
- Active learning optimization

## Success Metrics

### Usability
- Average time per label: < 15 seconds
- Session completion rate: > 80%
- User-reported satisfaction: > 4/5

### Data Quality
- Label-model agreement: Track over time
- Episode duration validity: > 95% DSM-5 compliant
- Missing data rate: < 10%

### Clinical Impact
- Model performance improvement: +10% AUC
- Personalization effectiveness: +15% accuracy
- Early warning detection: 3-5 days earlier

## Conclusion

This labeling CLI design balances clinical rigor with user experience, leveraging model predictions to create an efficient ground-truth collection system. By showing users what the model "thinks" and asking for confirmation or correction, we reduce cognitive load while gathering high-quality training data for personalization.

The modular architecture allows for incremental development, starting with the MVP command-line interface and expanding to more sophisticated features based on user feedback and clinical validation.