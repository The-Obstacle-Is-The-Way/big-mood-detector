
# PAT-L AUC Targets from Paper

## Target Performance for Depression Detection (PHQ-9 > 9)

| MODEL              | Avg Score* | n=500 | n=1000 | n=2500 | n=2800 | Params | Our Progress |
|--------------------|------------|-------|--------|--------|--------|--------|--------------|
| PAT-L (LP)         | 0.582      | 0.495 | 0.595  | 0.618  | 0.620  | 1.99 M | -            |
| PAT-L (FT)         | 0.589      | 0.541 | 0.577  | 0.618  | **0.620**  | 1.99 M | 0.5888 (3 epochs) |
| PAT Conv-M (LP)    | 0.589      | 0.556 | 0.584  | 0.611  | 0.605  | 1.00 M | -            |
| PAT Conv-M (FT)    | 0.594      | 0.576 | 0.585  | 0.609  | 0.606  | 1.00 M | -            |
| PAT Conv-L (FT)    | 0.610      | 0.594 | 0.606  | 0.617  | 0.624  | 1.99 M | -            |
| PAT Conv-L (LP)    | 0.611      | 0.594 | 0.606  | 0.618  | 0.625  | 1.99 M | -            |

#### **Supplemental Table 5. Evaluating models predicting depression from actigraphy.**

## Our Training Progress

### Fixed Issues
1. **Normalization Bug**: Was using fixed values (mean=2.5, std=2.0) instead of StandardScaler
2. **Paper's Method**: "standardized separately using Sklearn's StandardScaler"
3. **Impact**: AUC stuck at 0.4756 → Now 0.5888 and climbing

### Current Training (July 24, 2025)
- **Model**: PAT-L with Full Fine-Tuning (FT)
- **Dataset**: NHANES 2013-2014 (n=3077, paper mentions n=2800)
- **Target**: 0.620 AUC
- **Progress**:
  - Epoch 1: 0.5693
  - Epoch 2: 0.5759
  - Epoch 3: 0.5888
  - Continuing...

*Supplemental Table 5 Evaluating models predicting depression from actigraphy. In this dataset, the input is actigraphy, and the label indicates whether that participant has depression (PHQ-9 scores > 9). Each model is trained on dataset sizes "500", "1,000", "2,500", and "2,800", (seen in the columns) and evaluated using AUC on a held-out test set of 2,000 participants. The "Avg AUC" represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that it was trained on smoothed data. LP stands for linear probing, and FT stands for end-to-end finetuning. An underline indicates the best baseline model. A bolded PAT model suggests that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. PATs outperform the baseline models in every dataset size in this task.*