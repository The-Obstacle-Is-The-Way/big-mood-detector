# Big Mood Detector Research

A comprehensive research repository for mood detection using wearable data, containing converted academic papers and reference implementations.

## 📁 Project Structure

```
big-mood-detector/
├── 📖 pdf/                          # Original research papers (5 PDFs)
├── 📝 converted_markdown/           # High-quality markdown conversions + extracted images
├── 💻 reference_repos/              # Standalone implementation code
│   ├── mood_ml/                     # XGBoost sleep-based mood prediction
│   └── Pretrained-Actigraphy-Transformer/  # State-of-the-art transformer models
└── 📋 README.md                     # This file
```

## 🎯 Research Papers Included

1. **Sleep & Circadian Rhythm Prediction** - XGBoost models using sleep-wake patterns
2. **Digital Biomarkers (TIMEBASE)** - Empatica E4 wearable analysis  
3. **Fitbit Consumer Devices** - Personalized ML with consumer wearables
4. **Pretrained Actigraphy Transformer** - Foundation model for movement data

## 🏋️ Model Weights Setup

The Pretrained Actigraphy Transformer (PAT) models require downloading weights separately:

### Download Location
Get the PAT model weights from these Dropbox links (29k NHANES dataset - recommended):

- **PAT-L** (Large, 8MB): https://www.dropbox.com/scl/fi/exk40hu1nxc1zr1prqrtp/PAT-L_29k_weights.h5?rlkey=t1e5h54oob0e1k4frqzjt1kmz&st=7a20pcox&dl=1
- **PAT-M** (Medium, 4MB): https://www.dropbox.com/scl/fi/hlfbni5bzsfq0pynarjcn/PAT-M_29k_weights.h5?rlkey=frbkjtbgliy9vq2kvzkquruvg&st=mxc4uet9&dl=1  
- **PAT-S** (Small, 1MB): https://www.dropbox.com/scl/fi/12ip8owx1psc4o7b2uqff/PAT-S_29k_weights.h5?rlkey=ffaf1z45a74cbxrl7c9i2b32h&st=mfk6f0y5&dl=1

### Where to Place Them
Save downloaded weights to:
```
reference_repos/Pretrained-Actigraphy-Transformer/model_weights/
```

### XGBoost Models
The XGBoost models (`mood_ml/`) come with pre-trained `.pkl` files already included - no additional downloads needed.

## 🚀 Quick Start

1. **For sleep-based prediction**: Open `reference_repos/mood_ml/mood_ml.ipynb`
2. **For transformer models**: 
   - Download PAT weights (see above)
   - Open `reference_repos/Pretrained-Actigraphy-Transformer/Fine-tuning/PAT_finetuning.ipynb`

## 📊 What's Included

- ✅ **Complete research papers** converted to searchable markdown
- ✅ **All figures and tables** extracted as high-quality images  
- ✅ **Working code implementations** for both approaches
- ✅ **Pre-trained XGBoost models** ready to use
- ✅ **PAT transformer architecture** with fine-tuning tutorials
- ✅ **Baseline comparison models** (LSTM, CNN, ConvLSTM)

---

*Research repository for understanding and implementing state-of-the-art mood detection from wearable data* 