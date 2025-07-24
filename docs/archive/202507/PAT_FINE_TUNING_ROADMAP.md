# PAT Fine-Tuning Roadmap: From Embeddings to Predictions

**Date:** 2025-07-20  
**Status:** Critical Path Forward  
**Priority:** HIGH

## Current Reality Check

### What We Have ✅
1. **PAT Encoder Weights**: We have the pre-trained encoder that outputs 96-dim embeddings
2. **XGBoost Models**: Fully functional mood predictors with .pkl weights
3. **Clever Workaround**: Ensemble concatenates PAT embeddings with XGBoost features
4. **NHANES Processor**: Code to process NHANES data exists

### What We DON'T Have ❌
1. **PAT Classification Heads**: No fine-tuned heads for mood prediction
2. **NHANES Data**: The actual data files are missing from the repo
3. **True Ensemble**: PAT isn't making mood predictions, just providing features

### What's Actually Happening 🤔
```python
# Current "ensemble" flow:
1. PAT encoder → 96-dim embeddings
2. Take first 16 PAT embeddings + 20 XGBoost features = 36 features
3. Feed enhanced features → XGBoost
4. Result: "PAT-enhanced XGBoost" not true ensemble
```

## The Path Forward

### Option 1: Get PAT Classification Heads (Fastest)
**Timeline: 1-2 days**

1. Contact PAT authors:
   ```
   Email: franklin.y.ruan.24@dartmouth.edu
   Subject: Request for fine-tuned classification heads
   ```
2. Request their depression/medication prediction heads
3. Integrate heads into our PAT model
4. Enable true ensemble predictions

### Option 2: Fine-tune PAT Ourselves (Most Control)
**Timeline: 1-2 weeks**

#### Step 1: Get NHANES Data ✅ COMPLETE!
```bash
# You already have the data! Move it to the correct location:
mv /Users/ray/Downloads/*.xpt /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector/data/nhanes/2013-2014/

# Files you have:
# - PAXHD_H.xpt     ✅ Physical Activity Monitor - Header
# - PAXMIN_H.xpt    ✅ Physical Activity Monitor - Minute data  
# - DPQ_H.xpt       ✅ Depression Questionnaire (PHQ-9)
# - RXQ_DRUG.xpt    ✅ Prescription Drug Information
# - RXQ_RX_H.xpt    ✅ Prescription Medications
```

#### Step 2: Process NHANES Data
```python
# Use existing nhanes_processor.py
processor = NHANESProcessor(
    data_dir=Path("data/nhanes/2013-2014"),
    output_dir=Path("data/processed/nhanes")
)

# Extract features and labels
actigraphy_df = processor.load_actigraphy("PAXHD_H.XPT")
depression_df = processor.load_depression_scores("DPQ_H.XPT")
medication_df = processor.load_medications("RXQ_RX_H.XPT")

# Create labeled dataset
labeled_data = processor.create_labeled_dataset(
    task="depression",  # or "benzodiazepine", "ssri"
    min_wear_days=4
)
```

#### Step 3: Fine-tune PAT Head
```python
# Create classification head
from tensorflow import keras

def create_classification_head(input_dim=96, num_classes=2):
    return keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

# Fine-tune on NHANES
head = create_classification_head()
pat_with_head = combine_encoder_and_head(pat_encoder, head)

# Train for depression detection
history = pat_with_head.fit(
    X=nhanes_sequences,
    y=depression_labels,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)
```

#### Step 4: Update Ensemble
```python
# True ensemble with both models predicting
class TrueEnsembleOrchestrator:
    def predict(self, features, activity):
        # XGBoost prediction
        xgb_pred = self.xgboost.predict(features)
        
        # PAT prediction (now possible!)
        pat_sequence = self.build_sequence(activity)
        pat_pred = self.pat_with_head.predict(pat_sequence)
        
        # Weighted average
        ensemble = weighted_average(xgb_pred, pat_pred)
        return ensemble
```

### Option 3: Document Current Limitations (Immediate)
**Timeline: Today**

Update documentation to reflect reality:
- PAT provides embeddings only
- "Ensemble" is actually enhanced XGBoost
- True ensemble requires classification heads

## Implementation Priority

### Immediate (Today)
1. ✅ Update documentation to reflect actual capabilities
2. ✅ Add warnings about PAT limitations
3. ✅ Document fine-tuning as next contribution

### Short Term (This Week)
1. [ ] Download NHANES 2013-2014 data
2. [ ] Test nhanes_processor.py functionality
3. [ ] Contact PAT authors for heads
4. [ ] Create fine-tuning notebook

### Medium Term (Next 2 Weeks)
1. [ ] Implement PAT fine-tuning pipeline
2. [ ] Train depression classification head
3. [ ] Validate on held-out NHANES data
4. [ ] Update ensemble to use real predictions

### Long Term (Month+)
1. [ ] Train heads for all conditions (mania, hypomania)
2. [ ] Implement personal fine-tuning
3. [ ] Create pre-trained head zoo
4. [ ] Publish our fine-tuned weights

## Code Changes Needed

### 1. Update PATModel to Support Heads
```python
class PATModel:
    def load_classification_head(self, head_path: Path):
        """Load a fine-tuned classification head."""
        self.classification_head = keras.models.load_model(head_path)
        self.can_predict = True
    
    def predict_mood(self, sequence: PATSequence) -> MoodPrediction:
        """Make mood prediction (requires classification head)."""
        if not self.can_predict:
            raise RuntimeError("No classification head loaded")
        
        embeddings = self.extract_features(sequence)
        probs = self.classification_head.predict(embeddings)
        
        return MoodPrediction(
            depression_risk=probs[0],
            confidence=max(probs)
        )
```

### 2. Create Fine-tuning Script
```python
# scripts/fine_tune_pat.py
def main():
    # Load PAT encoder
    pat = PATModel(model_size="medium")
    pat.load_pretrained_weights(PAT_WEIGHTS)
    
    # Load NHANES data
    X_train, y_train = load_nhanes_depression_data()
    
    # Create and train head
    head = create_classification_head()
    model = combine_encoder_head(pat.encoder, head)
    
    # Fine-tune
    model.fit(X_train, y_train, epochs=20)
    
    # Save
    head.save("model_weights/pat/heads/depression_head.h5")
```

## Success Metrics

1. **PAT makes actual predictions**: Not just embeddings
2. **True ensemble**: Both models contribute predictions
3. **Improved accuracy**: PAT+XGBoost > XGBoost alone
4. **Documentation accuracy**: No more misleading claims

## The Big Picture

The previous developers did excellent work but misunderstood PAT's capabilities. They created a clever workaround (using embeddings as features) but marketed it as a true ensemble. Now we can:

1. **Be transparent** about current limitations
2. **Implement real PAT predictions** via fine-tuning
3. **Deliver on the ensemble promise** with both models predicting

This positions the project for genuine clinical utility rather than just technical demonstrations.

## Next Actions

1. **Today**: Update docs, add warnings
2. **Tomorrow**: Download NHANES, contact authors
3. **This Week**: Start fine-tuning experiments
4. **Next Week**: Deploy true ensemble

---

*Note: This roadmap turns a documentation crisis into an opportunity for significant contribution to the project.*