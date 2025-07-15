# PAT Model Weights - 29k NHANES Dataset

Downloaded pre-trained Pretrained Actigraphy Transformer (PAT) model weights.

## 🏋️ Available Model Sizes

All models trained on **2003-2004, 2005-2006, 2011-2012, 2013-2014 NHANES Actigraphy data (N=29,307)**

| Model | File | Size | Description |
|-------|------|------|-------------|
| **PAT-L** | `PAT-L_29k_weights.h5` | 8.0MB | **Large** - Best performance, higher computational cost |
| **PAT-M** | `PAT-M_29k_weights.h5` | 4.0MB | **Medium** - Balanced performance/efficiency |
| **PAT-S** | `PAT-S_29k_weights.h5` | 1.1MB | **Small** - Fastest, lower resource requirements |

## 🚀 Usage

Load these weights in the fine-tuning notebooks:

```python
# In PAT_finetuning.ipynb
model_weights_path = "model_weights/PAT-M_29k_weights.h5"  # Choose your size
```

## 📊 Model Architecture Details

| Size | Patch Size | Embed Dim | Heads | FF Dim | Layers | Dropout |
|------|------------|-----------|-------|--------|--------|---------|
| **Small** | 18 | 96 | 6 | 256 | 1 | 0.1 |
| **Medium** | 18 | 192 | 12 | 512 | 3 | 0.1 |
| **Large** | 18 | 384 | 24 | 1024 | 6 | 0.1 |

## 🎯 Recommendation

- **Start with PAT-M** for most applications (good balance)
- **Use PAT-L** for maximum accuracy on important tasks
- **Use PAT-S** for rapid prototyping or resource-constrained environments

## 📖 Related Files

- **Fine-tuning tutorial:** `../Fine-tuning/PAT_finetuning.ipynb`
- **Explainability:** `../Model Explainability/PAT_Explainability.ipynb`
- **Paper reference:** `../../../converted_markdown/pretrained-actigraphy-transformer/`

---
*Downloaded from official Dropbox links on the PAT GitHub repository* 