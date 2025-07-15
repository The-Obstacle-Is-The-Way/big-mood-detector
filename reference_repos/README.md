# Reference Repositories - Big Mood Detector Research

This folder contains standalone reference implementations for the research papers in our `converted_markdown/` directory.

## 📚 Repository-Paper Mappings

### 1. **XGBoost Sleep & Circadian Rhythm Models** 
**Repo:** `mood_ml/`  
**Paper:** `../converted_markdown/bipolar-depression-activity/`  
**Original:** [mcqeen1207/mood_ml](https://github.com/mcqeen1207/mood_ml)

**Key Features:**
- ✅ **Pre-trained XGBoost models** for depression, mania, and hypomania prediction
- ✅ **MATLAB sleep index calculator** (36 sleep & circadian features)
- ✅ **Ready-to-use pipeline** with example data
- ✅ **High accuracy**: 80% depression, 98% mania, 95% hypomania detection

**Key Files:**
- `mood_ml.ipynb` - Main prediction pipeline
- `XGBoost_DE.pkl` - Depression episode model
- `XGBoost_ME.pkl` - Manic episode model  
- `XGBoost_HME.pkl` - Hypomanic episode model
- `Index_calculation.m` - MATLAB sleep feature extraction
- `example.csv` - Sample sleep data format

### 2. **Pretrained Actigraphy Transformer (PAT)**
**Repo:** `Pretrained-Actigraphy-Transformer/`  
**Paper:** `../converted_markdown/pretrained-actigraphy-transformer/`  
**Original:** [njacobsonlab/Pretrained-Actigraphy-Transformer](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer)

**Key Features:**
- 🤖 **Foundation model** trained on 29,307 participants (NHANES data)
- 🚀 **State-of-the-art** wearable movement analysis 
- 🔧 **Fine-tuning capabilities** for custom mental health tasks
- 📊 **Built-in model explainability** and interpretability
- ⚡ **Lightweight & efficient** transformer architecture

**Key Files:**
- `Fine-tuning/PAT_finetuning.ipynb` - Main fine-tuning tutorial
- `Fine-tuning/PAT_Conv_finetuning.ipynb` - Convolutional variant
- `Model Explainability/PAT_Explainability.ipynb` - Interpretability tools
- `Pretraining/PAT_Pretraining.ipynb` - Self-supervised pretraining
- `Baseline Models/` - Comparison models (LSTM, CNN, ConvLSTM)

## 🎯 Usage Recommendations

### For Sleep-Based Mood Detection:
1. **Start with `mood_ml/`** - Ready-to-use XGBoost models
2. Use if you have: Sleep start/end times, sleep duration, wake time
3. **Pros:** Immediately usable, proven high accuracy
4. **Cons:** Limited to sleep-only features

### For Advanced Wearable Analytics:
1. **Start with `Pretrained-Actigraphy-Transformer/`** - Modern AI approach
2. Use if you have: Raw accelerometer/actigraphy time series data
3. **Pros:** State-of-the-art, flexible, explainable
4. **Cons:** Requires more setup and fine-tuning

## 🔧 Setup Notes

- **Git linkages removed** - These are standalone reference copies
- **Dependencies preserved** - Original requirements intact
- **Documentation included** - Full READMEs from original repos
- **Model weights available** - PAT weights downloadable from original repo

## 📖 Integration with Converted Papers

Each implementation directly corresponds to research detailed in our converted markdown papers:

```
reference_repos/mood_ml/ ←→ converted_markdown/bipolar-depression-activity/
reference_repos/Pretrained-Actigraphy-Transformer/ ←→ converted_markdown/pretrained-actigraphy-transformer/
```

Use the markdown papers for **understanding the research** and these repos for **implementing the solutions**.

---
*Reference repos cloned and decoupled for research purposes* 