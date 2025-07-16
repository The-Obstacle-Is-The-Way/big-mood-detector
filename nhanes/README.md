# NHANES Data for Future Fine-Tuning

## Overview

This directory contains National Health and Nutrition Examination Survey (NHANES) data files that will be used for future fine-tuning of our Pretrained Actigraphy Transformer (PAT) models.

## Data Files

| File | Size | Description |
|------|------|-------------|
| `DPQ_H.xpt` | 511K | Depression Screener (DPQ) - Mental health questionnaire data |
| `PAXHD_H.xpt` | 429K | Physical Activity Monitor - Header Data |
| `PAXMIN_H.xpt` | **8.7G** | Physical Activity Monitor - Minute Level Data |
| `RXQ_DRUG.xpt` | 3.2M | Prescription Medications - Drug Information |
| `RXQ_RX_H.xpt` | 9.4M | Prescription Medications - Participant Data |

## Why NHANES?

NHANES provides:
1. **Large-scale population data** with both actigraphy and mental health assessments
2. **High-quality annotations** from clinical interviews and validated questionnaires
3. **Diverse demographics** representing the US population
4. **Medication data** to understand treatment effects on activity patterns

## Future Plans

### PAT Fine-Tuning
- Use PAXMIN_H minute-level actigraphy data to fine-tune PAT for bipolar disorder detection
- Leverage DPQ depression scores as labels for supervised learning
- Incorporate medication data to account for treatment effects

### Data Processing Pipeline
1. Extract actigraphy sequences from PAXMIN_H
2. Align with mental health assessments from DPQ_H
3. Filter for relevant medications (mood stabilizers, antidepressants)
4. Create training/validation splits maintaining temporal integrity
5. Fine-tune PAT with bipolar-specific objectives

## Important Notes

⚠️ **These files are NOT committed to git** due to their large size (especially PAXMIN_H.xpt at 8.7GB).

To obtain NHANES data:
1. Visit https://wwwn.cdc.gov/nchs/nhanes/
2. Download the specific survey years needed
3. Place .xpt files in this directory

## Data Format

NHANES uses SAS Transport (.xpt) format. To read in Python:
```python
import pandas as pd
df = pd.read_sas('DPQ_H.xpt', format='xport')
```

## Privacy & Ethics

- NHANES data is de-identified and publicly available
- Follow all NHANES data use guidelines
- Cite NHANES appropriately in any publications

## Citation

When using NHANES data, cite:
```
Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS). 
National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. Department of 
Health and Human Services, Centers for Disease Control and Prevention, [appropriate year(s)].
```