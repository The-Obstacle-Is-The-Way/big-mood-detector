# AI Foundation Models for Wearable Movement Data in Mental Health Research 🏃‍♀️🏃

This is the GitHub associated with the paper "AI Foundation Models for Wearable Movement Data in Mental Health Research" <br>

Pretrained foundation models and transformer architectures have driven the success of large language models (LLMs) and other modern AI breakthroughs. However, similar advancements in health data modeling remain limited due to the need for innovative adaptations. Wearable movement data offers a valuable avenue for exploration, as it’s a core feature in nearly all commercial smartwatches, well established in clinical and mental health research, and the sequential nature of the data shares similarities to language. We introduce the Pretrained Actigraphy Transformer (PAT), the first open source foundation model designed for time-series wearable movement data. Leveraging transformer-based architectures and novel techniques, such as patch embeddings, and pretraining on data from 29,307 participants in a national U.S. sample, PAT achieves state-of-the-art performance in several mental health prediction tasks. PAT is also lightweight and easily interpretable, making it a robust tool for mental health research. 
 <br>

paper: https://doi.org/10.48550/arXiv.2411.15240

## 📔 Tutorial/Demo Notebooks (via Google Colab) 
These notebooks should be self-sufficient so you should be able to run them all the way through. Enjoy exploring!

### ⭐ How to: Fine-tune PAT + Built-in Model Explainability
[Launch Notebook](https://colab.research.google.com/drive/1HemPmkADQYRW214ft8ep8ARkfxPkwEij#scrollTo=eAlhD3TN148g)

This notebook will guide you through:
1. Setting up (importing/connecting to GPU)
2. Loading Demo Data
3. Loading Model
4. Finetuning Model
5. Evaluating Model
6. Model Explainability

> **Note:** You need to connect to GPU, or this notebook will be **very** slow.  
> In Google Colab, go to `runtime -> change runtime type`, then select a GPU. A GPU should already be the default when you open the link.

### 🎓 How to: Self-Supervised Pretraining
[Launch Notebook](https://colab.research.google.com/drive/1I_q3rRkGSYLZH-joYPobmOYobZPsAxag#scrollTo=FLBWYOLfN7Vt)

This notebook will guide you through:
1. Setting Up (importing/connecting to TPU)
2. Choosing Hyperparameters & Settings
3. Loading Demo Data
4. Loading Masked Autoencoder
5. Self-Supervised Training
6. Saving the Model Encoder
7. Inspecting the Autoencoder

> **Note:** You need to connect to TPU, or this notebook will **not** work.  
> In Google Colab, go to `runtime -> change runtime type`, then select TPUv2. This should already be the default when you open the link.<br>
> It is possible to easily adapt this notebook so that you can run with GPUs instead if you choose

---
## 💾 PAT Encoders Weights Download
These are the H5 files that store the pre-trained transformer encoder weights. Don't worry, they are small, just a few megabytes. The demo notebook above, `How to: Fine-tune PAT + Built-in Model Explainability`, shows you where to load encoder weights. 

### Pretrained on 2003-2004, 2005-2006, 2011-2012, 2013-2014 NHANES Actigraphy (N=29,307)
1. [PAT-L](https://www.dropbox.com/scl/fi/exk40hu1nxc1zr1prqrtp/PAT-L_29k_weights.h5?rlkey=t1e5h54oob0e1k4frqzjt1kmz&st=7a20pcox&dl=1)
2. [PAT-M](https://www.dropbox.com/scl/fi/hlfbni5bzsfq0pynarjcn/PAT-M_29k_weights.h5?rlkey=frbkjtbgliy9vq2kvzkquruvg&st=mxc4uet9&dl=1)
3. [PAT-S](https://www.dropbox.com/scl/fi/12ip8owx1psc4o7b2uqff/PAT-S_29k_weights.h5?rlkey=ffaf1z45a74cbxrl7c9i2b32h&st=mfk6f0y5&dl=1)

### Pretrained on 2003-2004, 2005-2006, 2011-2012 NHANES Actigraphy (N=21,538)
> ❗ Good if you want to conduct a study with 2013-2014 NHANES actigraphy data
1. [PAT-L](https://www.dropbox.com/scl/fi/dglz917p3hqw5mwbovsv2/PAT-L_21k_weights.h5?rlkey=ppzxvp9i7t9k8j3w9x77fjfil&st=3g3mm845&dl=1)
2. [PAT-M](https://www.dropbox.com/scl/fi/dsd6px97gcipqm80iie17/PAT-M_21k_weights.h5?rlkey=q480rjj5g2id88xt9feie70tj&st=ou924quo&dl=1)
3. [PAT-S](https://www.dropbox.com/scl/fi/ik45lrtqgenm61cgkkgkz/PAT-S_21k_weights.h5?rlkey=n2zv3jhdnvp7w8inir96y1ime&st=xch3lnra&dl=1)

---

## FAQs ❓

 
<details>
<summary><strong>Can PAT handle any input length?</strong></summary>
Yes, it can!  
Our model can handle inputs of any length (both longer and shorter than 1 week).  
Check out the `How to: Fine-tune PAT + Built-in Model Explainability` notebook above for a demo.
</details>

 
<details>
<summary><strong>Can I use GPUs / can I locally fine-tune PAT?</strong></summary>
Absolutely!  

 ---
#### ⚠️ **ANNOUNCEMENT (as of 13 January 2025 Colab Update): GPUs are recommended over TPUs for fine-tuning PAT** 
- **GPUs:**  
  - Fully compatible with PAT.  
  - On Colab, GPUs may be slower than TPUs but are more reliable.
- **TPUs:**  
  - Fine-tuning and evaluation may be unstable.  
  - ⚠️ Model explainability is **not supported** on TPUs.
---

</details>


<details>
<summary><strong>What are the hyperparameters for the model sizes?</strong></summary>

```python
# Model Size
if size == "small":
    patch_size = 18
    embed_dim = 96
    # encoder
    encoder_num_heads = 6
    encoder_ff_dim = 256
    encoder_num_layers = 1
    encoder_rate = 0.1

if size == "medium":
    patch_size = 18
    embed_dim = 96
    # encoder
    encoder_num_heads = 12
    encoder_ff_dim = 256
    encoder_num_layers = 2
    encoder_rate = 0.1

if size == "large":
    patch_size = 9
    embed_dim = 96
    # encoder
    encoder_num_heads = 12
    encoder_ff_dim = 256
    encoder_num_layers = 4
    encoder_rate = 0.1
```
</details>

---

## 📜 Attribution
Please cite our work if you use it 
```
@misc{
     ruanzhang2024PAT,
     title={AI Foundation Models for Wearable Movement Data in Mental Health Research},
     author={Ruan, Franklin Y. and Zhang, Aiwei and Oh, Jenny and Jin, SouYoung and Jacobson, Nicholas C.},
     publisher={arXiv:2411.15240},
     year={2024}
}
```

## ☎️ Contacts
**Corresponding Author:** Franklin Ruan | franklin.y.ruan.24@dartmouth.edu <br>


