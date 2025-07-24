#!/bin/bash
# Clean remaining documentation references

echo "🧹 Cleaning remaining documentation..."

# 1. Update predictions.md - remove ensemble references
echo "Updating docs/api/predictions.md..."
sed -i '' 's/ensemble/temporal/g' docs/api/predictions.md
sed -i '' 's/TensorFlow/PyTorch/g' docs/api/predictions.md

# 2. Update API_REFERENCE.md
echo "Updating docs/developer/API_REFERENCE.md..."
sed -i '' 's/ensemble predictions/temporal predictions/g' docs/developer/API_REFERENCE.md
sed -i '' 's/ensemble model/temporal separation/g' docs/developer/API_REFERENCE.md
sed -i '' 's/TensorFlow/PyTorch/g' docs/developer/API_REFERENCE.md

# 3. Update CLINICAL_DOSSIER.md
echo "Updating docs/clinical/CLINICAL_DOSSIER.md..."
sed -i '' 's/ensemble approach/temporal approach/g' docs/clinical/CLINICAL_DOSSIER.md
sed -i '' 's/ensemble model/temporal separation/g' docs/clinical/CLINICAL_DOSSIER.md

# 4. Update ADVANCED_USAGE.md
echo "Updating docs/user/ADVANCED_USAGE.md..."
sed -i '' 's/--ensemble/--temporal/g' docs/user/ADVANCED_USAGE.md
sed -i '' 's/ensemble mode/temporal mode/g' docs/user/ADVANCED_USAGE.md
sed -i '' 's/Ensemble predictions/Temporal predictions/g' docs/user/ADVANCED_USAGE.md

# 5. Update ENSEMBLE_MATHEMATICS.md to reflect reality
echo "Updating docs/models/ensemble/ENSEMBLE_MATHEMATICS.md..."
cat > docs/models/ensemble/TEMPORAL_SEPARATION.md << 'EOF'
# Temporal Separation Mathematics

## Overview

Big Mood Detector uses temporal separation rather than traditional ensemble methods. This document explains the mathematical foundation.

## Temporal Windows

### PAT Model (Current State)
- **Input**: Past 7 days of minute-level activity (10,080 data points)
- **Output**: Current depression risk assessment
- **Window**: T-7 days to T (now)

### XGBoost Model (Future Risk)
- **Input**: Past 30 days of aggregated features (36 dimensions)
- **Output**: Next-day mood episode risk
- **Window**: T+1 day prediction

## Mathematical Formulation

### Current State (PAT)
```
S_current = PAT(A_{t-10080:t})
```
Where A is activity sequence.

### Future Risk (XGBoost)
```
R_future = XGB(F_{t-30d:t})
```
Where F is feature vector.

### Combined Assessment
```
Assessment = {
    "current": S_current,
    "future": R_future,
    "confidence": min(conf_PAT, conf_XGB)
}
```

## Why Not Traditional Ensemble?

Traditional ensemble: `(0.6 × XGB + 0.4 × PAT)`

Our approach: Keep predictions separate because:
1. Different temporal windows
2. Different clinical meanings
3. Different intervention strategies

## Clinical Interpretation

- **High current, Low future**: Acute episode, may resolve
- **Low current, High future**: Warning sign, preventive action
- **High both**: Sustained episode, immediate intervention
- **Low both**: Stable state

EOF
rm docs/models/ensemble/ENSEMBLE_MATHEMATICS.md

echo "✅ Documentation cleanup complete!"
echo ""
echo "Summary of changes:"
echo "- Replaced 'ensemble' with 'temporal' throughout"
echo "- Updated TensorFlow references to PyTorch"
echo "- Created new TEMPORAL_SEPARATION.md"
echo ""
echo "Run validation:"
echo "grep -r 'ensemble\|TensorFlow' docs/ --include='*.md' | grep -v archive | grep -v literature | wc -l"