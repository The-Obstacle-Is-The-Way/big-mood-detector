# NHANES Data Directory

This directory contains National Health and Nutrition Examination Survey (NHANES) data files used for fine-tuning the PAT (Pretrained Actigraphy Transformer) model.

## Data Structure

```
nhanes/
├── 2013-2014/          # NHANES 2013-2014 cycle
│   ├── PAXHD_H.xpt     # Physical Activity Monitor - Header
│   ├── PAXMIN_H.xpt    # Physical Activity Monitor - Minute data
│   ├── DPQ_H.xpt       # Depression Questionnaire (PHQ-9)
│   ├── RXQ_DRUG.xpt    # Prescription Drug Information
│   └── RXQ_RX_H.xpt    # Prescription Medications
└── processed/          # Processed datasets (created by nhanes_processor.py)
```

## File Descriptions

### Activity Monitor Files
- **PAXHD_H.xpt**: Header file with participant info and device metadata
- **PAXMIN_H.xpt**: Minute-by-minute accelerometer data (x, y, z axes)

### Clinical Data Files
- **DPQ_H.xpt**: PHQ-9 depression screening scores
- **RXQ_DRUG.xpt**: Detailed drug information
- **RXQ_RX_H.xpt**: Prescription medication usage

## Why 2013-2014?

The PAT paper used NHANES cycles 2003-2004, 2005-2006, and 2011-2012 for pre-training, explicitly excluding 2013-2014 for testing. This makes 2013-2014 perfect for:
1. Testing our fine-tuned models
2. Avoiding data leakage
3. Fair comparison with published results

## Usage

```python
from big_mood_detector.infrastructure.fine_tuning import NHANESProcessor

processor = NHANESProcessor(
    data_dir=Path("data/nhanes/2013-2014"),
    output_dir=Path("data/nhanes/processed")
)

# Load and process data
actigraphy = processor.load_actigraphy("PAXMIN_H.xpt")
depression = processor.load_depression_scores("DPQ_H.xpt")
medications = processor.load_medications("RXQ_RX_H.xpt")
```

## Download Instructions

If files are missing, download from:
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013

1. Select "2013-2014" cycle
2. Navigate to:
   - Examination Data → Physical Activity Monitor
   - Questionnaire Data → Mental Health - Depression Screener
   - Questionnaire Data → Prescription Medications

## Privacy Note

NHANES data is publicly available and de-identified. No special privacy considerations needed beyond standard research ethics.

## Citation

When using NHANES data, cite:
```
Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS). 
National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. Department of 
Health and Human Services, Centers for Disease Control and Prevention, 2013-2014.
```