# **AI Foundation Models for Wearable Movement Data in Mental Health Research**

Franklin Y. Ruan1,2†\* , Aiwei Zhang1,2† , Jenny Y. Oh<sup>1</sup> , SouYoung Jin<sup>2</sup> , & Nicholas C. Jacobson1,2,3,4

<sup>1</sup>Center for Technology and Behavioral Health, Geisel School of Medicine, Dartmouth College,

Lebanon, NH, United States

<sup>2</sup>Department of Computer Science, Dartmouth College, Hanover, NH, United States

<sup>3</sup>Department of Biomedical Data Science, Geisel School of Medicine, Dartmouth College,

Lebanon, NH, United States

<sup>4</sup>Department of Psychiatry, Geisel School of Medicine, Dartmouth College, Lebanon, NH,

United States

†Authors contributing equally to this work

#### **Author Statement:**

Franklin Y. Ruan https://orcid.org/0000-0002-6360-2899 Aiwei Zhang https://orcid.org/0009-0006-9343-7544 Jenny Y. Oh https://orcid.org/0000-0002-2504-4398 SouYoung Jin https://orcid.org/0000-0003-1889-7808 Nicholas C. Jacobson <https://orcid.org/0000-0002-8832-4741>

\*Correspondence concerning this article should be addressed to Franklin Y. Ruan, Center for

Technology and Behavioral Health, Dartmouth College, 46 Centerra Parkway Suite 300,

Lebanon, NH 03766. Email: franklin.y.ruan.24@dartmouth.edu

### **Acknowledgements**

#### **Funding**

This research was supported by the National Institute of Mental Health (NIMH) and the National Institute of General Medical Sciences (NIGMS) grant R01MH123482.

### **Competing Interests**

NCJ has received a grant from Boehringer-Ingelheim. NCJ has edited a book through Academic Press and receives book royalties, and NCJ also receives speaking fees related to his research.

### **Author Contributions**

FYR, AZ, SJ, and NCJ conceptualized the project and its design. FYR and AZ acquired the data and conducted preliminary processing. FYR and AZ performed model design and model evaluation. FYR, AZ, JYO, SJ, and NCJ performed interpretation and analysis of results. All authors were major contributors in drafting, revising, editing, and reviewing the manuscript. NCJ obtained funding, and NCJ and SJ supervised the project. All authors have read and approved the final manuscript.

### **Abstract**

Pretrained foundation models and transformer architectures have driven the success of large language models (LLMs) and other modern AI breakthroughs. However, similar advancements in health data modeling remain limited due to the need for innovative adaptations. Wearable movement data offers a valuable avenue for exploration, as it's a core feature in nearly all commercial smartwatches, well established in clinical and mental health research, and the sequential nature of the data shares similarities to language. We introduce the Pretrained Actigraphy Transformer (PAT), the first open source foundation model designed for time-series wearable movement data. Leveraging transformer-based architectures and novel techniques, such as patch embeddings, and pretraining on data from 29,307 participants in a national U.S. sample, PAT achieves state-of-the-art performance in several mental health prediction tasks. PAT is also lightweight and easily interpretable, making it a robust tool for mental health research. GitHub: <https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer/>

### **1. Introduction**

Movement-based actigraphy, or wearable accelerometer data, was utilized as early as the 1970s to non-intrusively and effectively measure sleep and circadian patterns, particularly in "free-living" (non-laboratory) environments1–3 . As wearables became more ubiquitous, researchers began to see that actigraphy contained data beyond sleep cycle information; movement intensities throughout time could be correlated to a participant's behavioral patterns4–7 . Indeed, developing a fully proportional accelerometer<sup>8</sup> significantly enhanced the precision of activity monitoring across various settings, facilitating broader applications in psychopathology9,10 , chronobiology11,12 , and health risk management5,13–17 . Presently, wearable sensor data is now a core feature of nearly every commercially available mobile consumer device (e.g. watches, smartphones)16,18,19 , and actigraphy data has become a rising modality and an important source of health information for clinical researchers20,21 .

Despite their promise, current actigraphy studies face notable limitations. Historically, actigraphy studies have predominantly relied on simple processing pipelines or traditional statistical feature-based models<sup>20</sup> , and although CNNs and RNN-based LSTMs22–26 have been applied to actigraphy, these architectures have inherent limitations. While they excel at capturing short-term patterns, CNNs struggle to identify long-range dependencies critical for time-series data27–29 . RNN-based LSTMs often face challenges in retaining information over extended time periods due to issues like vanishing gradients, which limit their effectiveness for datasets spanning days or even weeks30–32 . A significant gap remains between these approaches and the state-of-the-art architectures in fields like natural language processing and computer vision. In these fields,

transformers excel at capturing complex, long-range dependencies—features crucial for time-series applications like actigraphy that require modeling interactions across hours or even days33–35 . To our knowledge, there is no existing foundation model that specifically adapts transformer architectures for wearable accelerometer data. These challenges highlight the need for novel approaches that leverage the strength of transformers while addressing the demands of wearable sensors.

However, adapting traditional transformers to actigraphy is challenging due to the lengthy and dense nature of the data. Transformer models often limit input tokens to the hundreds, as each additional token quadratically increases memory and computation costs36,37 . For example, one week of actigraphy data produces over 10,000 tokens with minute-level sampling, quickly overwhelming standard transformers. Moreover, many current studies using wearables or actigraphy data tend to feature a limited number of participants38,39 , as it is often difficult to amass large participant pools for clinical research purposes. Paired with a limited participant size, the complexity of longitudinal actigraphy data poses a challenge for machine learning models, which face a difficult time abstracting from such data. Therefore, foundation models pretrained on extensive datasets offer a promising alternative to training models from scratch. These pretrained models can be fine-tuned on smaller, task-specific actigraphy datasets, leveraging knowledge from broader data sources to achieve robust performance even with limited participant samples.

To address these challenges, we propose designing a simple-to-use yet robust actigraphy foundation model that can be easily deployed for health related studies. First, we construct an attention model architecture for actigraphy. Moving beyond CNNs and LSTM models, we present an Actigraphy Transformer (AT) architecture that utilizes attention mechanisms and patching techniques<sup>40</sup> to efficiently embed lengthy actigraphy data. Second, we design a mechanism for model explainability. Using attention weights, our model can output visualizations that indicate which minutes of physical activity are most impactful in the model decision. This allows researchers to make reliable inferences about how actigraphy may relate to a health outcome within a study. Third, we pretrain the model in a BERT <sup>41</sup> or Masked Autoencoder<sup>42</sup> -like manner on actigraphy data from almost 30,000 participants from a US cohort to create the Pretrained Actigraphy Transformer (PAT).

Using datasets where actigraphy is labeled with a participant's psychotropic medication usage, sleep disorders and abnormalities, or PHQ-9 score (for depression), we aim to evaluate PAT's efficacy when fine-tuned for mental health studies. We hypothesize that PAT, our explainable foundation model, can deliver state-of-the-art results when fine-tuned for various actigraphy studies, even in datasets containing small amounts of labeled actigraphy data.

### **2. Results**

For evaluative purposes, PAT was pretrained on week-long actigraphy data from three nationally representative cohorts of participants in the United States: the NHANES 2003-2004, 2005-2006, and 2011-2012 datasets<sup>43</sup> , totaling 21,538 participants. The fourth cohort, NHANES 2013-2014, consisting of 7,769 participants, was entirely held out in order to create various supervised datasets for testing. Models pretrained on all 29,307 participants, including the 2013-2014 cohort, are also available on our GitHub.

### **2.1 PAT Transfer Learning Results**

We held out all NHANES 2013-2014 data to create supervised datasets. Using these datasets, we will evaluate PAT's ability to predict the following labels: Benzodiazepine medication usage, SSRI medication usage, sleep disorders, sleep abnormalities, and depression. Each label has its own dataset, such that the input is the participant's actigraphy, and the output is the label (e.g., depression). For each dataset, a test set of 2,000 participants is held out, while the remaining participants are used for training. We compare against several baseline models. These include a logistic regression model trained on wavelet transform features, an LSTM, a 1D CNN, a 3D CNN, and a ConvLSTM. The 3D CNN and ConvLSTM modeling strategies for actigraphy are adapted from Heinz et al.<sup>22</sup> and Rahman and Adjeroh<sup>25</sup> and, to our knowledge, are currently the best models used for actigraphy understanding tasks.

**Predicting benzodiazepine usage.** To examine model performance across varying dataset sizes, we subset participants into groups of "500", "1,000", "2,500", and "5,769" to simulate different levels of data availability. This approach allows us to assess the model's applicability across scenarios with limited and abundant data, reflecting both personalized and large-scale settings. As seen in Table 1, the best baseline model was the 3D CNN with smoothed actigraphy data, achieving an AUC of 0.677 with 500 participants, 0.695 with 1,000 participants, 0.695 with 2,500 participants, and 0.719 with all 5,769 participants, with an average AUC of 0.697 across all sizes.

PAT models consistently outperform the best baseline. PAT-L scored the best, with an AUC of 0.771 when fine-tuned with only 500 participants, 0.765 with 1,000 participants, 0.760 with

2,500 participants, and 0.771 with all 5,769 participants. The average AUC across all four datasets for PAT-L is 0.767, achieving a 7.0 percentage-point absolute improvement over the best baseline.

| MODEL                 | Avg AUC* | n=500 | n=1000 | n=2500 | n=5769 | Params |
|-----------------------|----------|-------|--------|--------|--------|--------|
| LSTM                  | 0.493    | 0.501 | 0.487  | 0.474  | 0.512  | 15 K   |
| LSTM (smoothing)      | 0.499    | 0.506 | 0.508  | 0.482  | 0.499  | 15 K   |
| Wavelet Transform     | 0.620    | 0.674 | 0.625  | 0.598  | 0.583  | 10 K   |
| CNN-1D                | 0.632    | 0.621 | 0.630  | 0.640  | 0.637  | 10 K   |
| CNN-1D (smoothing)    | 0.639    | 0.633 | 0.634  | 0.644  | 0.646  | 10 K   |
| Conv LSTM (smoothing) | 0.667    | 0.666 | 0.680  | 0.653  | 0.671  | 1.75 M |
| Conv LSTM             | 0.668    | 0.663 | 0.681  | 0.650  | 0.677  | 1.75 M |
| CNN-3D                | 0.693    | 0.683 | 0.693  | 0.693  | 0.703  | 790 K  |
| CNN-3D (smoothing)    | 0.697    | 0.677 | 0.695  | 0.696  | 0.719  | 790 K  |
| PAT-S                 | 0.701    | 0.706 | 0.718  | 0.677  | 0.703  | 285 K  |
| PAT Conv-S            | 0.726    | 0.737 | 0.711  | 0.722  | 0.735  | 285 K  |
| PAT-M                 | 0.744    | 0.743 | 0.745  | 0.742  | 0.745  | 1.00 M |
| PAT Conv-M            | 0.761    | 0.753 | 0.756  | 0.760  | 0.773  | 1.00 M |
| PAT Conv-L            | 0.762    | 0.763 | 0.756  | 0.754  | 0.773  | 1.99 M |
| PAT-L                 | 0.767    | 0.771 | 0.765  | 0.760  | 0.771  | 1.99 M |

**Table 1: Evaluating models on the ability to predict benzodiazepine usage from actigraphy**

*Table 1 Evaluating models on the ability to predict benzodiazepine usage from actigraphy. In this dataset, the input is actigraphy and the label indicates whether a participant is taking benzodiazepines. Each model is trained on dataset sizes "500", "1,000", "2,500", and "5,769" (seen in the columns) and evaluated using AUC on a held-out test set of 2,000 participants. The "Avg AUC" represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that the model was trained on smoothed data. Underlined text indicates the best baseline model. PAT-S/M/L denotes Small, Medium, Large. A bolded PAT model indicates that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. PATs significantly outperform the baseline models in every dataset size in this task.*

**Predicting SSRI usage, sleep disorder, sleep abnormalities, and depression.** We evaluate our

models on predicting SSRI usage, sleep disorders, sleep abnormalities, and depression (see Table

2) in the same way we evaluate our model on predicting benzodiazepine usage, as seen in Table

| MODEL                 | SSRI usage | Sleep Disorder | Sleep<br>Abnormalities | Depression | Params |
|-----------------------|------------|----------------|------------------------|------------|--------|
| LSTM                  | 0.527      | 0.494          | 0.513                  | 0.489      | 15 K   |
| LSTM (smoothing)      | 0.523      | 0.506          | 0.515                  | 0.506      | 15 K   |
| Wavelet Transform     | 0.674      | 0.529          | 0.525                  | 0.523      | 10 K   |
| CNN-1D                | 0.616      | 0.563          | 0.534                  | 0.522      | 10 K   |
| CNN-1D (smoothing)    | 0.611      | 0.558          | 0.519                  | 0.517      | 10 K   |
| Conv LSTM (smoothing) | 0.655      | 0.609          | 0.579                  | 0.547      | 1.75 M |
| Conv LSTM             | 0.606      | 0.606          | 0.585                  | 0.550      | 1.75 M |
| CNN-3D                | 0.677      | 0.608          | 0.606                  | 0.583      | 790 K  |
| CNN-3D (smoothing)    | 0.680      | 0.605          | 0.615                  | 0.586      | 790 K  |
| PAT-S                 | 0.641      | 0.587          | 0.555                  | 0.560      | 285 K  |
| PAT Conv-S            | 0.656      | 0.616          | 0.573                  | 0.587      | 285 K  |
| PAT-M                 | 0.690      | 0.641          | 0.641                  | 0.559      | 1.00 M |
| PAT Conv-M            | 0.668      | 0.616          | 0.627                  | 0.594      | 1.00 M |
| PAT Conv-L            | 0.695      | 0.631          | 0.659                  | 0.610      | 1.99 M |
| PAT-L                 | 0.700      | 0.632          | 0.665                  | 0.589      | 1.99 M |
|                       |            |                |                        |            |        |

**Table 2: Model performance across different actigraphy tasks (predicting SSRI usage, Sleep Disorder, Sleep Abnormalities, and Depression)**

*Table 2 Model performance across dif erent actigraphy tasks (predicting SSRI usage, Sleep Disorder, Sleep Abnormalities, and Depression)***.** *The table summarizes the performance of our PAT models versus various baseline models (including LSTM, CNN, ConvLSTM, and 3D CNN) across four tasks: predicting SSRI usage, history of any sleep disorder, abnormal sleep patterns, and depression. Each model is trained on dataset sizes "500", "1,000", "2,500", and all available data (5,769 for SSRI usage, 3,429 for Sleep Disorder and Sleep abnormalities, and 2,800 for Depression) and evaluated using AUC on a held-out test set of 2,000 participants. The score for each model here represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that the model was trained on smoothed data. An underline indicates the best baseline model. PAT-S/M/L denotes Small, Medium, Large. A bolded PAT model indicates that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. The results suggest that PATs outperform baseline models in various actigraphy understanding tasks and at various dataset sizes.*

Most of our PAT models above the small size performed better than the best baseline model found in each dataset. Using the average AUC metric seen in Table 1 and comparing the best PAT model with the best baseline model, we find a 2.0 percentage-point absolute improvement when predicting SSRIs, a 3.2 percentage-point absolute improvement when predicting disorders, a 5.0 percentage-point absolute improvement when predicting sleep abnormalities, and a 2.4 percentage-point absolute improvement when predicting depression. A further breakdown of the model scores can be found in Table 2 and our supplemental information. More information about the datasets can be found in the methods.

### **2.2 PAT Experiments**

We conducted experiments on PAT-M to find the optimal pretraining conditions. We derive all our experiment results from a validation set of 1,000 participants in the benzodiazepine dataset, separate and with no overlap from the held-out test set of the 2,000 participants used for Figure 1.

Table 3 shows that (1) Higher mask ratios during pretraining improve performance; the optimal mask ratio found was 0.90. (2) Smoothing the data does not improve performance. (3) In the reconstruction step of our pretraining, we found that calculating the mean squared error (MSE) for every time step produced a significantly better pretrained model than calculating the MSE only for the masked portions. As such, all PAT models for Figures 1 and 2 are pretrained with these hyperparameters optimized.

#### **Table 3. PAT-M Experiments**

|                             | (a)<br>MODEL<br>(Mask<br>Ratio) | Avg<br>Score*                      |               |
|-----------------------------|---------------------------------|------------------------------------|---------------|
|                             | Medium<br>0.25                  | 0.737                              |               |
|                             | Medium<br>0.50                  | 0.707                              |               |
|                             | Medium<br>0.75                  | 0.743                              |               |
|                             | Medium<br>0.90                  | 0.773                              |               |
| (b)<br>MODEL<br>(Smoothing) | Avg<br>Score*                   | (c)<br>MODEL<br>(Loss<br>Function) | Avg<br>Score* |
| Medium<br>(Smoothed)        | 0.741                           | Medium<br>(MSE<br>Only<br>Masked)  | 0.541         |
| Medium<br>(Not<br>Smooth)   | 0.773                           | Medium<br>(MSE<br>All)             | 0.773         |

*Table 3 PAT-M Experiments. All models are pretrained and then end-to-end fine-tuned on the benzodiazepine training data. For the experiments, 1,000 participants were removed from the training data to create an evaluation set, and these 1,000 participants are separate from the held-out test set seen in Table 1. The "Avg Score" metric is the average AUC score on the evaluation set after the medium model was trained on dataset sizes "500", "1,000", "2,500", and "4,769". (a) We tested a PAT-M pretrained using MSE loss on every data point. We found that a higher mask ratio during pretraining leads to better results. (b) We tested PAT-M pretrained on 90% masking and MSE loss on all data and found that smoothing does not improve performance. (c) We tested a PAT-M with 90% masking and found that MSE on only the masked patches decreased performance drastically.*

### **2.3 Model Explainability**

Health-related studies are often interested in the mechanisms behind a phenomenon, but the black-box nature of deep learning models makes it difficult for researchers to characterize associations even if they are found44,45 . As such, we use the attention weights inherently built into our model to infer which patterns within the accelerometry data are most paid attention to when making a prediction. This allows for easy model explainability, enabling researchers to make characterizations about actigraphy and a given label, as seen in Figure 1.

![](_page_11_Figure_0.jpeg)

**Figure 1 Example of PAT Model Explainability.**

*Figure 1. Example of PAT Model Explainability. Plots (a) and (b) are of one participant labeled as taking benzodiazepines. (a) represents the participant's actigraphy data across a week, and (b) represents that week's actigraphy data averaged into 1 day. The levels of importance shown in the heatmap are derived from the model's attention scores during the classification task. Attention scores, which indicate the relative focus of the model on specific time points, were normalized and mapped to a color scale, with red representing regions of higher attention (interpreted as higher importance) and blue indicating regions of lower attention. In this example, the model pays high attention to the behaviors in the early mornings and picks up on the participant's late (noon) wake-up time when predicting benzodiazepine usage.*

### **3. Discussion**

Accelerometer data from wearables contain a rich source of information for healthcare and research5,46,47 . Moreover, actigraphy is becoming increasingly relevant as wearable technology becomes more ubiquitous. The easily accessible wearable data, which can be obtained retrospectively and in real-time, shows promise for representing longitudinal and naturalistic behavioral patterns21,48 .

We note, however, that major improvements can be made to the model architectures currently used for actigraphy studies. Wearable accelerometer data is unique in its ability to provide continuous, minute-level information over long periods, making it invaluable for studying longitudinal behaviors like sleep, activity levels, and mental health outcomes1,2 . However, the complexity and longitudinal structure of actigraphy data limit feature engineering and traditional machine learning models<sup>23</sup> , as patterns can span across hours or even days, requiring models that capture both short and long term changes such as daily movement and weekly activity cycles, respectively.

The preferred model used for many actigraphy studies, ConvLSTM24,25,49,50 , is limited by local context, as recurrent and convolutional-based models rely on the inductive bias that data points close together are more related than those further apart. Actigraphy data is often extensive and relies on capturing long-range dependencies and this presents a challenge for the ConvLSTM architecture. To circumvent this, the ConvLSTM model expands the context window for temporal and spatial models by reshaping data into a 3D matrix of 7 days × 24 hours × 60 minutes so that the start of each day is close time-wise, and the start of each hour is close

space-wise. Even so, the ConvLSTM may miss certain associations, as researchers introduce strong inductive biases during modeling<sup>51</sup> . For example, the model would struggle to relate events from a Tuesday night to a Thursday morning, as they would continue to be spatially distant. These limitations, driven by inductive biases inherent to recurrent and convolutional architectures, can lead to missed associations when modeling actigraphy data.

We address these issues through our PAT model. Indeed, PAT consistently outperforms the most popular models used for actigraphy in various tasks and at various dataset sizes (Table 1 and Table 2). We develop a transformer-based model because it offers the advantages of global attention<sup>33</sup> , allowing us to extrapolate beyond the local context. For instance, in the context of predicting sleep disorders, our transformers can relate movement patterns at night to a change in behavior in the evening of the next day, or even the day after—despite the fact that these events could be separated by thousands of time steps. Additionally, attention-based models require few inductive priors in contrast to recurrent or convolutional based models; attention-based models extrapolate which minutes of actigraphy data are most related through associations found in the data alone. As such, transformers are more flexible and general, allowing for better embeddings of temporally and spatially complex actigraphy data33,52 . With PAT, context is elevated to a global scale such that any set of minutes can be associated with any other set of minutes along the week, regardless of spatial or temporal distance.

Another major problem in machine learning and actigraphy-based research is the limited number of participants in many clinical studies38,39 ; it can be difficult for models to abstract information from a limited pool of participants. As such, we pretrained our PAT models on over 20k

participants' unlabeled actigraphy data so that PAT can still produce robust results even when fine-tuned on small amounts of labeled actigraphy data. Table 1 suggests that PAT works well even in limited data conditions. PAT-L shows an 8.8 percentage-point absolute improvement compared to the best baseline model when both were trained on a labeled dataset with only 500 participants.

By leveraging pretrained representations, researchers can reduce the reliance on large labeled datasets, facilitating effective model training across a broader range of clinical scenarios. This robustness makes PATs suitable for studying populations with low base rate diseases or rare conditions, where recruiting sufficient participants can be challenging. In tandem, PATs show promise in personalized or individualized health modeling. Pretrained models like PAT could help address the "cold-start" problem in sensing applications by providing a strong initial model before additional data is collected, or the system becomes actionable.

On the contrary, PATs also show robustness in scaling, which can be important as wearable data becomes more accessible. Natural language processing and computer vision studies have suggested that attention-based models scale better than their convolutional or RNN counterparts42,52 . Likewise, PAT-L has a 5.2 percentage-point absolute improvement compared to the best baseline model when both models were trained on all available labeled data in Figure 1—and we expect PAT performance to increase with more pretraining data.

Having established the strengths of transformer-based models for actigraphy data, it is equally important to consider how they may be implemented. To increase the model's accessibility for

research, all the code, alongside tutorial notebooks for model explainability, fine-tuning PAT, and pretraining PAT, can be found in our GitHub repository. We also note that our largest model, PAT-L, is just under 2 million parameters. All models can take in variable input lengths, can be developed and trained using accelerators on Google Colab with a Pro account, which costs \$10 (USD) per month—a relatively modest expense compared to traditional high-performance computing setups. Each model iteration was able to train in under six hours on Colab Pro, making it feasible for researchers with limited resources to experiment and adapt these models to their specific datasets. Our models are robust yet lightweight and easy to deploy without heavy computational resources in research settings. As wearable technology and longitudinal data become widely available, models like PAT may prove instrumental in harnessing this data to its full clinical and academic capability.

### **4. Methods**

### **4.1 Actigraphy Transformer (AT) Architecture**

We first create a transformer architecture tailored to embed lengthy actigraphy data (see Figure 2). Using a technique called patching40,53 , we design our model to efficiently embed long signals or sequences.

**Model Design.** In the architecture design, we follow the original Transformer<sup>33</sup> and the Vision Transformer<sup>52</sup> as closely as possible to benefit from the known scalable nature of computer vision and NLP transformer architectures. A transformer generally receives a 1D sequence of token embeddings as its input, which translates well to the sequential nature of actigraphy data. However, due to the quadratic complexity of the attention mechanism, it is inefficient to input a week's worth (10,080 minutes) of actigraphy data as individual tokens. Following the Vision Transformer's and PatchTST's solution to circumvent the attention's quadratic complexity in images and lengthy time series, Actigraphy Transformer (AT) breaks the sequence of actigraphy into fixed-sized non-overlapping patches. N=T/S is the number of patches, where T is the total number of time steps, and S is the number of time steps in a patch. The resultant N represents the number of tokens input into AT. Like the Vision Transformer model, the AT attention layer uses a constant vector size D throughout, and we create patch embeddings by mapping each patch to D dimensions using a trainable linear projection. Each AT Transformer block consists of a multiheaded self-attention layer, a layer normalization layer, a feed-forward network, and another layer normalization layer.

**Figure 2. Actigraphy Transformer**

![](_page_17_Figure_1.jpeg)

*Figure 2 Actigraphy Transformer (Not Pretrained). The figure presents an example case of using an Actigraphy Transformer (AT) for classification. Actigraphy data is fed into the AT and transformed into patches. Assuming a patch size of 18, the number of time steps fed into our transformer is reduced from 10,080 to 560. We add positional embeddings to our patch embeddings before feeding them to the transformer layers. An output layer head is attached to our Actigraphy Transformer and is shown to perform a model classification task.*

**Positional Embeddings.** Since transformer-based models are inherently permutationally invariant, we add positional embeddings for each patch. We use fixed sine and cosine positional embeddings to retain the location of each patch, which has been shown to work well for transformers in NLP and computer vision42,52 .

**Convolutional Patch Embeddings.** As an alternative to creating patch embeddings through linear projections, we also designed an architecture that uses 1D convolutional layers to embed each patch. Models with this embedding have "conv" added to the end of their name.

### **4.2 Pretrained Actigraphy Transformer (PAT)**

Enabled by our Actigraphy Transformer architecture, we utilize a simple masked autoencoder pretraining approach to create deep bidirectional representations of our unlabeled longitudinal actigraphy data (see Figure 3). This way, our pretrained PAT model can be fine-tuned through any labeled actigraphy dataset, including small ones. PAT can then be harnessed to create state-of-the-art models for various applications, such as predicting and characterizing depression, medication use, or sleep disorders through actigraphy data.

**Masked Autoencoder Design.** Our pretraining task is simple but effective; an encoder creates a latent representation of our input, and a decoder reconstructs the original signal from the encoder representation. We utilize an asymmetric autoencoder similar to the masked autoencoder (MAE) pretraining method used for images<sup>42</sup> . In MAE, random portions of the input data are masked, and the model is trained to reconstruct the missing sections. This process encourages the model to learn meaningful patterns by focusing on context and relationships in the data rather than relying on adjacent information alone. Our encoder encodes only portions of actigraphy data, while our lightweight decoder is tasked with reconstructing an entire week's actigraphy from the encoder representation tokens and mask tokens. An advantage of our Actigraphy Transformer over conventional convolutional-based models is its ability to divide our sequential data into fixed-size, non-overlapping patches for pretraining. Using this structure, we can randomly select patches to remove or mask once given a mask ratio. As reported in the original MAE paper, we

also observed that random masking with a high mask ratio increases the complexity of the reconstruction task, as the autoencoder would not be able to rely merely on adjacent patches of actigraphy for reconstruction (See Table 3).

**Figure 3. PAT Pretraining and Fine-tuning**

![](_page_19_Figure_2.jpeg)

*Figure 3. PAT Pretraining and Fine-tuning. (a) During the pretraining phase, actigraphy data is transformed into patch embeddings with positional embeddings and then randomly masked without replacement (e.g. 90% of patches are removed). The remaining patches are fed into a transformer encoder. The encoder embedding tokens are then realigned with masked tokens, and positional embeddings are added again before being fed into a lightweight decoder. The decoder output is then fed to a simple output layer to reconstruct the input actigraphy data. The mean squared error between the input and the reconstructed output is calculated for loss. (b) After pretraining, the patch embedding layer and the pretrained transformer encoder are extracted to create PAT. Any output layer can be added to PAT for a variety of actigraphy understanding tasks (e.g. classification).*

Following the approach in the original MAE paper, masked patches are entirely removed when fed into the encoder. The pretraining speed is faster since we only feed a small subset of the patches to the encoder. Since our encoder will not encounter mask tokens during the fine-tuning step, removing mask tokens during pretraining has been shown to increase performance<sup>42</sup> . Mask tokens are, however, introduced to the decoder. The input to our decoder, then, is each patch's embedding from our encoder and mask tokens representing the remaining patches.

As the transformer models are positionally invariant, fixed positional embeddings are added to all the tokens before being inputted into both the encoder and decoder, giving the transformer models information on the position of each token or patch.

**Reconstruction Target.** Our input into the autoencoder is a sequence of standardized actigraphy data for 10,080 minutes. We attempt to reconstruct this standardized actigraphy data on the minute level with the decoder, such that the output length is also 10,080 minutes. We use a mean squared error at the minute level between the standardized input and the reconstructed output for our loss function.

**Fine-tuning**. We remove and extract the weights of our patch embedding layer and encoder for the fine-tuning step. We name this encoder and its layer for creating patch embeddings Pretrained Actigraphy Transformer (PAT). Any layers can be added to the end of PAT for differing tasks. For our evaluation purposes, we add a small feed forward layer and a sigmoid activation function at the end of PAT for binary classification. During fine-tuning, we evaluate end-to-end fine-tuning (FT), where we freeze no weights, and linear probing (LP), where we freeze PAT and only train the added layers. Since FT and LP yielded similar results, we only report FT in this manuscript, with LP results shown in the supplementals.

#### **Model Explainability**

We use our models' intrinsic attention matrices for simple model explainability. We extract the attention weights from the last transformer block since that block generally yields the most abstracted data<sup>33</sup> . From there, we sum the attention weights across the key positions for each query. Then, we average and normalize the summed attention weights across each head. Using this method, we can see which patches (representing short sequences of activity) our model pays the most attention to. See Figure 1.

### **4.3 Data and Evaluation Scheme**

**Data Source and Collection** This study utilizes participant data from the National Health and Nutrition Examination Survey (NHANES)<sup>43</sup> , which has received approval from the NCHS Research Ethics Review Board. NHANES is a longitudinal, nationally representative study designed to assess the health and nutritional status of the American population. Actigraphy data was collected from differing individuals during the study years 2003-2004, 2005-2006, 2011-2012, and 2013-2014. From 2003 to 2006, data was captured using a hip-mounted ActiGraph AM-7164 piezoelectric accelerometer, which records uniaxial acceleration as summed activity intensity per minute over a week. In later cycles from 2011 to 2014, NHANES shifted to a wrist-worn triaxial accelerometer (Actigraph GT3X+). We use the summed triaxial data per minute to align with earlier uniaxial data.

**Pretraining Data.** The pretraining task requires only the actigraphy data. The 2003-2004 dataset contains actigraphy data for 7,176 participants, the 2005-2006 dataset contains actigraphy data for 7,455 participants, and the 2011-2012 actigraphy dataset contains actigraphy data for 6,907 participants. In total, data from 21,538 participants was used for pretraining. Each participant in

this study recorded one week or 10,080 minutes of accelerometer data, and each dataset is standardized with respect to itself.

A model pretrained on every year of available actigraphy data, including 2013-2014, for a total of 29,307 participants, is also available on our GitHub, though it has not been evaluated.

**Supervised Training Dataset.** We used the 2013-2014 NHANES dataset for the supervised training tasks. This data, collected via wrist-worn accelerometers, differs from the hip-mounted devices used in the earlier pretraining phase. The focus of the 2013 dataset is to assess the model's predictive performance on various health outcomes, including benzodiazepine usage, depression, sleep disorders, and sleep abnormalities, based on actigraphy data.

A subset of 7,769 participants who provided actigraphy and medication data was identified. Using this data, we create the benzodiazepine and SSRI datasets. In the benzodiazepine dataset, we label participants taking benzodiazepines as 1 and all others as 0. In the SSRI dataset, we label participants taking SSRIs as 1 and all others as 0. A subset of 4,800 participants had accompanying PHQ-9 depression screener54,55 results alongside actigraphy data. We create a dataset that labels participants with a PHQ-9 score of 10 or above as 1 (has depression) and all others as 0. A subset of 5,429 participants had actigraphy and sleep disorder data, as per the SLQ-sleep survey. Using this data, we create two datasets: sleep disorders and sleep abnormalities. For the sleep disorder dataset, if the participant reports having had a sleep disorder at any time in their life, they were labeled as 1 and all others as 0. For the sleep abnormalities dataset, participants were labeled 1 if they had any of the following: (1) had a sleep disorder at

any time in their life, (2) regularly slept more than or equal to 12 hours, or (3) regularly slept less than or equal to 5 hours.

**Supervised Learning Evaluation Scheme.** Each dataset is first split into a train set and then a held-out test set with 2,000 participants. For any dataset, assume N is the total number of participants left in the training set. To assess a model's ability to train on various dataset sizes, we sample participants with replacement using a stratified splitting method and create the following test sizes: "500", "1,000", "2,500", and N. We then take 20% of each new dataset to create corresponding validation sets. We save all these new datasets to train our model.

**Evaluation Metric.** We use AUC, or the area under the receiver operating characteristic curve, to evaluate our model's performance in binary classification. To assess our model's performance, even when the presented supervised training data is limited, we rank our models after averaging their AUC scores on the test set after being trained on each dataset size "500", "1,000", "2,500", and N. A higher average AUC score is better.

**PAT Experiments.** We perform experiments on the best pretraining parameters using only the benzodiazepine dataset. With 7,769 participants, 2,000 were first removed to create a held-out test set. The remaining 5,769 participants were used to find the optimal parameters for pretraining, including the optimal mask ratio, loss function, and whether or not using smoothed data bolsters performance.

1,000 participants were set aside for this experiment, distinct from the held-out test set. This left us with 4,769 participants with whom to train the model. We searched for the best pretraining parameters using the evaluation metric mentioned in the section above (i.e., the best average AUC score on the 1,000-participants used for this experiment).

**Smoothing and Standardization.** In models where we trained with smoothed data, we used a Savitzky-Golay filter with a window length of 51-time steps and a polynomial order of 3, which has been shown to retain temporal patterns while reducing noise23,56 .

In our supervised datasets, all train, test, and validation sets were standardized separately using Sklearn's StandardScaler, such that each activity intensity is reported as a standard deviation from the mean of a given minute in the sequence. In our pretraining data, we standardized the 2003-2004, 2005-2006, and 2011-2012 datasets separately, as differing devices were used throughout the years. Standardization allows consistency among actigraphy datasets used, where absolute accelerometer measurements may vary.

### **5. Data Availability**

The datasets used during the current study are available from the National Health and Nutrition Examination Survey (NHANES), maintained by the Centers for Disease Control and Prevention (CDC). These datasets are publicly available and can be accessed at <https://wwwn.cdc.gov/nchs/nhanes/Default.aspx>.

# **6. Code Availability**

The underlying code for this study is publicly available and can be accessed via this link ([https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer/\)](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer/). The repository also includes links to tutorial/demo notebooks for fine-tuning PAT, model explainability, and masked autoencoder pretraining. Additionally, the pretrained models themselves are available for download and use.

# **7. REFERENCES**

- 1. Dunn J, Kidzinski L, Runge R, et al. Wearable sensors enable personalized predictions of clinical laboratory measurements. *Nat Med*. 2021;27(6):1105-1112. doi:10.1038/s41591-021-01339-0
- 2. Boe AJ, McGee Koch LL, O'Brien MK, et al. Automating sleep stage classification using wireless, wearable sensors. *Npj Digit Med*. 2019;2(1):131. doi:10.1038/s41746-019-0210-1
- 3. Yoo JY, Oh S, Shalish W, et al. Wireless broadband acousto-mechanical sensing system for continuous physiological monitoring. *Nat Med*. 2023;29(12):3137-3148. doi:10.1038/s41591-023-02637-5
- 4. Kline CE. Actigraphy (Wrist, for Measuring Rest/Activity Patterns and Sleep). In: Gellman MD, ed. *Encyclopedia of Behavioral Medicine*. Springer International Publishing; 2020:20-24. doi:10.1007/978-3-030-39903-0\_782
- 5. Fuster-Garcia E, Bresó A, Miranda JM, García-Gómez JM. Actigraphy Pattern Analysis for Outpatient Monitoring. In: Fernández-Llatas C, García-Gómez JM, eds. *Data Mining in Clinical Medicine*. Vol 1246. Methods in Molecular Biology. Springer New York; 2015:3-17. doi:10.1007/978-1-4939-1985-7\_1
- 6. Kupfer DJ. Psychomotor Activity in Affective States. *Arch Gen Psychiatry*. 1974;30(6):765. doi:10.1001/archpsyc.1974.01760120029005
- 7. Kryger MH, Roth T, Dement WC, eds. *Principles and Practice of Sleep Medicine*. Sixth edition. Elsevier; 2017.
- 8. Tryon WW, Williams R. Fully proportional actigraphy: A new instrument. *Behav Res Methods Instrum Comput*. 1996;28(3):392-403. doi:10.3758/BF03200519
- 9. Tazawa Y, Wada M, Mitsukura Y, et al. Actigraphy for evaluation of mood disorders: A systematic review and meta-analysis. *J Affect Disord*. 2019;253:257-269. doi:10.1016/j.jad.2019.04.087
- 10. Tahmasian M, Khazaie H, Golshani S, Avis KT. Clinical Application of Actigraphy in Psychotic Disorders: A Systematic Review. *Curr Psychiatry Rep*. 2013;15(6):359. doi:10.1007/s11920-013-0359-2
- 11. Brown AC, Smolensky MH, D'Alonzo GE, Redman DP. Actigraphy: A Means of Assessing Circadian Patterns in Human Activity. *Chronobiol Int*. 1990;7(2):125-133. doi:10.3109/07420529009056964
- 12. Schneider J, Fárková E, Bakštein E. Human chronotype: Comparison of questionnaires and wrist-worn actigraphy. *Chronobiol Int*. 2022;39(2):205-220. doi:10.1080/07420528.2021.1992418
- 13. Wyatt J, Guo CC. From Research to Application of Wearable-Derived Digital Health Measures—A Perspective From ActiGraph. *J Meas Phys Behav*. 2024;7(1):jmpb.2023-0045. doi:10.1123/jmpb.2023-0045
- 14. Korszun A, Young EA, Engleberg NC, Brucksch CB, Greden JF, Crofford LA. Use of actigraphy for monitoring sleep and activity levels in patients with fibromyalgia and depression. *J Psychosom Res*. 2002;52(6):439-443. doi:10.1016/S0022-3999(01)00237-9
- 15. Strain T, Wijndaele K, Dempsey PC, et al. Wearable-device-measured physical activity and future health risk. *Nat Med*. 2020;26(9):1385-1391. doi:10.1038/s41591-020-1012-3
- 16. Zheng NS, Annis J, Master H, et al. Sleep patterns and risk of chronic disease as measured by long-term monitoring with commercial wearable devices in the All of Us Research

Program. *Nat Med*. 2024;30(9):2648-2656. doi:10.1038/s41591-024-03155-8

- 17. Schalkamp AK, Peall KJ, Harrison NA, Sandor C. Wearable movement-tracking data identify Parkinson's disease years before clinical diagnosis. *Nat Med*. 2023;29(8):2048-2056. doi:10.1038/s41591-023-02440-2
- 18. Perez AJ, Zeadally S. Recent Advances in Wearable Sensing Technologies. *Sensors*. 2021;21(20):6828. doi:10.3390/s21206828
- 19. Masoumian Hosseini M, Masoumian Hosseini ST, Qayumi K, Hosseinzadeh S, Sajadi Tabar SS. Smartwatches in healthcare medicine: assistance and monitoring; a scoping review. *BMC Med Inform Decis Mak*. 2023;23(1):248. doi:10.1186/s12911-023-02350-w
- 20. Ancoli-Israel S, Martin JL, Blackwell T, et al. The SBSM Guide to Actigraphy Monitoring: Clinical and Research Applications. *Behav Sleep Med*. 2015;13(sup1):S4-S38. doi:10.1080/15402002.2015.1046356
- 21. Liguori C, Mombelli S, Fernandes M, et al. The evolving role of quantitative actigraphy in clinical sleep medicine. *Sleep Med Rev*. 2023;68:101762. doi:10.1016/j.smrv.2023.101762
- 22. Heinz MV, Price GD, Ruan F, et al. Association of Selective Serotonin Reuptake Inhibitor Use With Abnormal Physical Movement Patterns as Detected Using a Piezoelectric Accelerometer and Deep Learning in a Nationally Representative Sample of Noninstitutionalized Persons in the US. *JAMA Netw Open*. 2022;5(4):e225403. doi:10.1001/jamanetworkopen.2022.5403
- 23. Ruan F, Adjei S, Amanna A, Price G, Heinz MV, Jacobson NC. Characterizing Benzodiazepine Use in a Large National Study via Wearables and Deep Learning. Published online September 16, 2024. doi:10.31234/osf.io/5ckme
- 24. Dorris H, Oh J, Jacobson N. Wearable Movement Data as a Potential Digital Biomarker for Chronic Pain: An Investigation Using Deep Learning. *Phys Act Health*. 2024;8(1):83-92. doi:10.5334/paah.329
- 25. Rahman SA, Adjeroh DA. Deep Learning using Convolutional LSTM estimates Biological Age from Physical Activity. *Sci Rep*. 2019;9(1):11425. doi:10.1038/s41598-019-46850-0
- 26. Patterson MR, Nunes AAS, Gerstel D, et al. 40 years of actigraphy in sleep medicine and current state of the art algorithms. *Npj Digit Med*. 2023;6(1):51. doi:10.1038/s41746-023-00802-1
- 27. Lim B, Zohren S. Time-series forecasting with deep learning: a survey. *Philos Trans R Soc Math Phys Eng Sci*. 2021;379(2194):20200209. doi:10.1098/rsta.2020.0209
- 28. Liu X, Wang W. Deep Time Series Forecasting Models: A Comprehensive Survey. *Mathematics*. 2024;12(10):1504. doi:10.3390/math12101504
- 29. Wang Y, Wu H, Dong J, Liu Y, Long M, Wang J. Deep Time Series Models: A Comprehensive Survey and Benchmark. Published online 2024. doi:10.48550/ARXIV.2407.13278
- 30. Zhao J, Huang F, Lv J, et al. Do RNN and LSTM have Long Memory? In: III HD, Singh A, eds. *Proceedings of the 37th International Conference on Machine Learning*. Vol 119. Proceedings of Machine Learning Research. PMLR; 2020:11365-11375. https://proceedings.mlr.press/v119/zhao20c.html
- 31. Lindemann B, Müller T, Vietz H, Jazdi N, Weyrich M. A survey on long short-term memory networks for time series prediction. *Procedia CIRP*. 2021;99:650-655. doi:10.1016/j.procir.2021.03.088
- 32. Le P, Zuidema W. Quantifying the Vanishing Gradient and Long Distance Dependency Problem in Recursive Neural Networks and Recursive LSTMs. In: *Proceedings of the 1st*

*Workshop on Representation Learning for NLP*. Association for Computational Linguistics; 2016:87-93. doi:10.18653/v1/W16-1610

33. Vaswani A, Shazeer N, Parmar N, et al. Attention is All you Need. In: *Advances in Neural Information Processing Systems*. Vol 30. Curran Associates, Inc.; 2017. Accessed May 23, 2024.

https://proceedings.neurips.cc/paper\_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a8 45aa-Abstract.html

- 34. Wen Q, Zhou T, Zhang C, et al. Transformers in Time Series: A Survey. In: *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence*. International Joint Conferences on Artificial Intelligence Organization; 2023:6778-6786. doi:10.24963/ijcai.2023/759
- 35. Bi J, Zhu Z, Meng Q. Transformer in Computer Vision. In: *2021 IEEE International Conference on Computer Science, Electronic Information Engineering and Intelligent Control Technology (CEI)*. IEEE; 2021:178-188. doi:10.1109/CEI52496.2021.9574462
- 36. Fournier Q, Caron GM, Aloise D. A Practical Survey on Faster and Lighter Transformers. *ACM Comput Surv*. 2023;55(14s):1-40. doi:10.1145/3586074
- 37. Tay Y, Dehghani M, Bahri D, Metzler D. Efficient Transformers: A Survey. *ACM Comput Surv*. 2023;55(6):1-28. doi:10.1145/3530811
- 38. Lemola S, Ledermann T, Friedman EM. Variability of Sleep Duration Is Related to Subjective Sleep Quality and Subjective Well-Being: An Actigraphy Study. *PLOS ONE*. 2013;8(8):e71292. doi:10.1371/journal.pone.0071292
- 39. Price GD, Heinz MV, Zhao D, Nemesure M, Ruan F, Jacobson NC. An unsupervised machine learning approach using passive movement data to understand depression and schizophrenia. *J Affect Disord*. 2022;316:132-139. doi:10.1016/j.jad.2022.08.013
- 40. Nie Y, Nguyen NH, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. Published online March 5, 2023. Accessed October 19, 2024. http://arxiv.org/abs/2211.14730
- 41. Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Published online May 24, 2019. doi:10.48550/arXiv.1810.04805
- 42. He K, Chen X, Xie S, Li Y, Dollár P, Girshick R. Masked Autoencoders Are Scalable Vision Learners. Published online 2021. doi:10.48550/ARXIV.2111.06377
- 43. NHANES National Health and Nutrition Examination Survey Homepage. May 13, 2024. Accessed May 23, 2024. https://www.cdc.gov/nchs/nhanes/index.htm
- 44. Defining the undefinable: the black box problem in healthcare artificial intelligence | Journal of Medical Ethics. Accessed October 24, 2024. https://jme.bmj.com/content/48/10/764.abstract
- 45. Shining Light Into the Black Box of Machine Learning | JNCI: Journal of the National Cancer Institute | Oxford Academic. Accessed October 24, 2024. https://academic.oup.com/jnci/article/111/9/877/5272613
- 46. Grap MJ, Hamilton VA, McNallen A, et al. Actigraphy: Analyzing patient movement. *Heart Lung*. 2011;40(3):e52-e59. doi:10.1016/j.hrtlng.2009.12.013
- 47. Feasibility of Continuous Actigraphy in Patients in a Medical Intensive Care Unit | American Journal of Critical Care | American Association of Critical-Care Nurses. Accessed October 24, 2024.

https://aacnjournals.org/ajcconline/article-abstract/26/4/329/4274/Feasibility-of-Continuous-

Actigraphy-in-Patients

- 48. Walia HK, Mehra R. Practical aspects of actigraphy and approaches in clinical and research domains. In: *Handbook of Clinical Neurology*. Vol 160. Elsevier; 2019:371-379. doi:10.1016/B978-0-444-64032-1.00024-2
- 49. Zhang Z, Lv Z, Gan C, Zhu Q. Human action recognition using convolutional LSTM and fully-connected LSTM with different attentions. *Neurocomputing*. 2020;410:304-316. doi:10.1016/j.neucom.2020.06.032
- 50. Haghayegh S, Khoshnevis S, Smolensky MH, Diller KR. Application of deep learning to improve sleep scoring of wrist actigraphy. *Sleep Med*. 2020;74:235-241. doi:10.1016/j.sleep.2020.05.008
- 51. Wang Z, Wu L. Theoretical Analysis of the Inductive Biases in Deep Convolutional Networks. In: Oh A, Naumann T, Globerson A, Saenko K, Hardt M, Levine S, eds. *Advances in Neural Information Processing Systems*. Vol 36. Curran Associates, Inc.; 2023:74289-74338. https://proceedings.neurips.cc/paper\_files/paper/2023/file/eb1bad7a84ef68a64f1afd6577725 d45-Paper-Conference.pdf
- 52. Sharir G, Noy A, Zelnik-Manor L. An Image is Worth 16x16 Words, What is a Video Worth? Published online May 27, 2021. Accessed October 19, 2024. http://arxiv.org/abs/2103.13915
- 53. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. Published online June 3, 2021. doi:10.48550/arXiv.2010.11929
- 54. Kroenke K, Spitzer RL, Williams JBW. The PHQ-9. *J Gen Intern Med*. 2001;16(9):606-613. doi:10.1046/j.1525-1497.2001.016009606.x
- 55. Urtasun M, Daray FM, Teti GL, et al. Validation and calibration of the patient health questionnaire (PHQ-9) in Argentina. *BMC Psychiatry*. 2019;19(1):291. doi:10.1186/s12888-019-2262-9
- 56. Guiñón JL, Ortega E, García-Antón J, Pérez-Herranz V. Moving average and Savitzki-Golay smoothing filters using Mathcad. *Pap ICEE*. 2007;2007:1-4.

### **Supplementary Material**

| MODEL                 | Avg AUC* | n=500 | n=1000 | n=2500 | n=5769 | Params |
|-----------------------|----------|-------|--------|--------|--------|--------|
| LSTM                  | 0.493    | 0.501 | 0.487  | 0.474  | 0.512  | 15 K   |
| LSTM (smoothing)      | 0.499    | 0.506 | 0.508  | 0.482  | 0.499  | 15 K   |
| Wavelet Transform     | 0.620    | 0.674 | 0.625  | 0.598  | 0.583  | 10 K   |
| CNN-1D                | 0.632    | 0.621 | 0.630  | 0.640  | 0.637  | 10 K   |
| CNN-1D (smoothing)    | 0.639    | 0.633 | 0.634  | 0.644  | 0.646  | 10 K   |
| Conv LSTM (smoothing) | 0.667    | 0.666 | 0.680  | 0.653  | 0.671  | 1.75 M |
| Conv LSTM             | 0.668    | 0.663 | 0.681  | 0.650  | 0.677  | 1.75 M |
| CNN-3D                | 0.693    | 0.683 | 0.693  | 0.693  | 0.703  | 790 K  |
| CNN-3D (smoothing)    | 0.697    | 0.677 | 0.695  | 0.696  | 0.719  | 790 K  |
| PAT-S (FT)            | 0.701    | 0.706 | 0.718  | 0.677  | 0.703  | 285 K  |
| PAT-S (LP)            | 0.706    | 0.721 | 0.721  | 0.677  | 0.705  | 285 K  |
| PAT Conv-S (FT)       | 0.726    | 0.737 | 0.711  | 0.722  | 0.735  | 285 K  |
| PAT Conv-S (LP)       | 0.727    | 0.734 | 0.713  | 0.722  | 0.738  | 285 K  |
| PAT-M (FT)            | 0.744    | 0.743 | 0.745  | 0.742  | 0.745  | 1.00 M |
| PAT-M (LP)            | 0.745    | 0.734 | 0.746  | 0.742  | 0.756  | 1.00 M |
| PAT Conv-M (LP)       | 0.759    | 0.753 | 0.753  | 0.759  | 0.773  | 1.00 M |
| PAT Conv-M (FT)       | 0.761    | 0.753 | 0.756  | 0.760  | 0.773  | 1.00 M |
| PAT Conv-L (FT)       | 0.762    | 0.763 | 0.756  | 0.754  | 0.773  | 1.99 M |
| PAT Conv-L (LP)       | 0.762    | 0.765 | 0.755  | 0.754  | 0.773  | 1.99 M |
| PAT-L (FT)            | 0.767    | 0.771 | 0.765  | 0.760  | 0.771  | 1.99 M |
| PAT-L (LP)            | 0.768    | 0.775 | 0.764  | 0.760  | 0.771  | 1.99 M |

**Supplemental Table 1. Evaluating models predicting benzodiazepine usage from actigraphy (Including linear probing data).**

*Supplemental Table 1 Evaluating models predicting benzodiazepine usage from actigraphy. The dif erence between this table and Table 1 in the manuscript is that we include linear probing data, denoted as LP. FT stands for end-to-end finetuning. In this dataset, the input is actigraphy, and the label indicates whether that participant is taking benzodiazepines. Each model is trained on dataset sizes "500", "1,000", "2,500", and "5,769", (seen in the columns) and evaluated using AUC on a held-out test set of 2,000 participants. The "Avg AUC" represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that it was trained on smoothed data. An underline indicates the best baseline model. A bolded PAT model suggests that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. PATs significantly outperform the baseline models in every dataset size in this task.*

| MODEL              | Avg Score* | n=500 | n=1000 | n=2500 | n=5769 | Params |
|--------------------|------------|-------|--------|--------|--------|--------|
| LSTM (Smooth)      | 0.523      | 0.520 | 0.505  | 0.541  | 0.527  | 15 K   |
| LSTM               | 0.527      | 0.533 | 0.534  | 0.518  | 0.524  | 15 K   |
| Wavelet Transform  | 0.572      | 0.569 | 0.559  | 0.552  | 0.606  | 10 K   |
| Conv LSTM          | 0.606      | 0.444 | 0.585  | 0.691  | 0.703  | 1.75 M |
| CNN-1D (Smooth)    | 0.611      | 0.487 | 0.643  | 0.651  | 0.664  | 10 K   |
| CNN-1D             | 0.616      | 0.548 | 0.600  | 0.660  | 0.655  | 10 K   |
| PAT-S (FT)         | 0.641      | 0.586 | 0.626  | 0.674  | 0.679  | 285 K  |
| PAT-S (LP)         | 0.643      | 0.598 | 0.617  | 0.676  | 0.679  | 285 K  |
| Conv LSTM (Smooth) | 0.655      | 0.583 | 0.639  | 0.700  | 0.698  | 1.75 M |
| PAT Conv-S (FT)    | 0.656      | 0.536 | 0.711  | 0.692  | 0.684  | 285 K  |
| PAT-L (LP)         | 0.662      | 0.495 | 0.721  | 0.713  | 0.720  | 1.99 M |
| PAT Conv-S (LP)    | 0.666      | 0.571 | 0.714  | 0.693  | 0.687  | 285 K  |
| PAT Conv-M (FT)    | 0.668      | 0.552 | 0.705  | 0.705  | 0.712  | 1.00 M |
| CNN-3D             | 0.677      | 0.668 | 0.678  | 0.668  | 0.695  | 790 K  |
| CNN-3D (Smooth)    | 0.680      | 0.671 | 0.675  | 0.682  | 0.692  | 790 K  |
| PAT Conv-M (LP)    | 0.680      | 0.597 | 0.707  | 0.703  | 0.714  | 1.00 M |
| PAT-M (FT)         | 0.690      | 0.661 | 0.704  | 0.684  | 0.710  | 1.00 M |
| PAT Conv-L (LP)    | 0.694      | 0.674 | 0.718  | 0.709  | 0.674  | 1.99 M |
| PAT Conv-L (FT)    | 0.695      | 0.677 | 0.717  | 0.710  | 0.675  | 1.99 M |
| PAT-L (FT)         | 0.700      | 0.651 | 0.720  | 0.710  | 0.721  | 1.99 M |
| PAT-M (LP)         | 0.702      | 0.698 | 0.702  | 0.699  | 0.710  | 1.00 M |

#### **Supplemental Table 2. Evaluating models predicting SSRI usage from actigraphy.**

*Supplemental Table 2 Evaluating models predicting SSRI usage from actigraphy. In this dataset, the input is actigraphy, and the label indicates whether that participant is taking SSRIs. Each model is trained on dataset sizes "500", "1,000", "2,500", and "5,769", (seen in the columns) and evaluated using AUC on a held-out test set of 2,000 participants. The "Avg AUC" represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that it was trained on smoothed data. LP stands for linear probing, and FT stands for end-to-end finetuning. An underline indicates the best baseline model. A bolded PAT model suggests that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. PATs outperform the baseline models in every dataset size in this task.*

| MODEL              | Avg Score* | n=500 | n=1000 | n=2500 | n=3429 | Params |
|--------------------|------------|-------|--------|--------|--------|--------|
| LSTM               | 0.494      | 0.480 | 0.490  | 0.509  | 0.499  | 15 K   |
| LSTM (Smooth)      | 0.506      | 0.486 | 0.511  | 0.489  | 0.540  | 15 K   |
| Wavelet Transform  | 0.529      | 0.525 | 0.510  | 0.544  | 0.539  | 10 K   |
| CNN-1D (Smooth)    | 0.558      | 0.556 | 0.540  | 0.570  | 0.566  | 10 K   |
| CNN-1D             | 0.563      | 0.571 | 0.545  | 0.568  | 0.569  | 10 K   |
| PAT-S (FT)         | 0.587      | 0.584 | 0.546  | 0.605  | 0.612  | 285 K  |
| PAT-S (LP)         | 0.596      | 0.586 | 0.579  | 0.607  | 0.613  | 285 K  |
| CNN-3D (Smooth)    | 0.605      | 0.600 | 0.597  | 0.601  | 0.621  | 790 K  |
| Conv LSTM          | 0.606      | 0.591 | 0.608  | 0.602  | 0.623  | 1.75 M |
| CNN-3D             | 0.608      | 0.611 | 0.612  | 0.587  | 0.624  | 790 K  |
| Conv LSTM (Smooth) | 0.609      | 0.591 | 0.604  | 0.609  | 0.633  | 1.75 M |
| PAT Conv-S (LP)    | 0.613      | 0.600 | 0.619  | 0.615  | 0.620  | 285 K  |
| PAT Conv-M (FT)    | 0.616      | 0.588 | 0.622  | 0.617  | 0.637  | 1.00 M |
| PAT Conv-S (FT)    | 0.616      | 0.600 | 0.624  | 0.617  | 0.622  | 285 K  |
| PAT Conv-M (LP)    | 0.616      | 0.587 | 0.621  | 0.618  | 0.637  | 1.00 M |
| PAT Conv-L (LP)    | 0.627      | 0.614 | 0.632  | 0.626  | 0.636  | 1.99 M |
| PAT Conv-L (FT)    | 0.631      | 0.624 | 0.633  | 0.630  | 0.637  | 1.99 M |
| PAT-L (FT)         | 0.632      | 0.633 | 0.644  | 0.613  | 0.638  | 1.00 M |
| PAT-L (LP)         | 0.634      | 0.631 | 0.644  | 0.621  | 0.639  | 1.00 M |
| PAT-M (FT)         | 0.641      | 0.625 | 0.647  | 0.639  | 0.652  | 1.99 M |
| PAT-M (LP)         | 0.641      | 0.624 | 0.647  | 0.640  | 0.652  | 1.99 M |

**Supplemental Table 3. Evaluating models predicting if a participant has or once had a sleep disorder from actigraphy.**

*Supplemental Table 3 Evaluating models predicting if participant has or once had a sleep disorder from actigraphy. In this dataset, the input is actigraphy, and the label indicates if a participant has or once had a sleep disorder. Each model is trained on dataset sizes "500", "1,000", "2,500", and "3,429", (seen in the columns) and evaluated using AUC on a held-out test set of 2,000 participants. The "Avg AUC" represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that it was trained on smoothed data. LP stands for linear probing, and FT stands for end-to-end finetuning. An underline indicates the best baseline model. A bolded PAT model suggests that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. PATs outperform the baseline models in every dataset size in this task.*

| MODEL              | Avg Score* | n=500 | n=1000 | n=2500 | n=3429 | Params |
|--------------------|------------|-------|--------|--------|--------|--------|
| LSTM               | 0.513      | 0.500 | 0.498  | 0.532  | 0.524  | 15 K   |
| LSTM (Smooth)      | 0.515      | 0.522 | 0.524  | 0.493  | 0.522  | 15 K   |
| CNN-1D (Smooth)    | 0.519      | 0.478 | 0.513  | 0.534  | 0.550  | 10 K   |
| Wavelet Transform  | 0.525      | 0.503 | 0.525  | 0.547  | 0.524  | 10 K   |
| CNN-1D             | 0.534      | 0.501 | 0.548  | 0.549  | 0.538  | 10 K   |
| PAT-S (FT)         | 0.555      | 0.527 | 0.516  | 0.610  | 0.568  | 285 K  |
| PAT-S (LP)         | 0.565      | 0.531 | 0.558  | 0.607  | 0.565  | 285 K  |
| PAT Conv-S (LP)    | 0.571      | 0.512 | 0.596  | 0.610  | 0.564  | 285 K  |
| PAT Conv-S (FT)    | 0.573      | 0.506 | 0.604  | 0.615  | 0.568  | 285 K  |
| Conv LSTM (Smooth) | 0.579      | 0.518 | 0.609  | 0.592  | 0.598  | 1.75 M |
| Conv LSTM          | 0.585      | 0.558 | 0.607  | 0.586  | 0.589  | 1.75 M |
| CNN-3D             | 0.606      | 0.596 | 0.632  | 0.547  | 0.650  | 790 K  |
| CNN-3D (Smooth)    | 0.615      | 0.588 | 0.618  | 0.628  | 0.625  | 790 K  |
| PAT Conv-M (FT)    | 0.627      | 0.591 | 0.624  | 0.649  | 0.644  | 1.00 M |
| PAT Conv-M (LP)    | 0.632      | 0.599 | 0.632  | 0.649  | 0.647  | 1.00 M |
| PAT-M (LP)         | 0.641      | 0.583 | 0.653  | 0.661  | 0.665  | 1.00 M |
| PAT-M (FT)         | 0.641      | 0.585 | 0.653  | 0.661  | 0.666  | 1.00 M |
| PAT Conv-L (LP)    | 0.659      | 0.614 | 0.661  | 0.676  | 0.685  | 1.99 M |
| PAT Conv-L (FT)    | 0.659      | 0.616 | 0.661  | 0.675  | 0.685  | 1.99 M |
| PAT-L (LP)         | 0.665      | 0.627 | 0.667  | 0.678  | 0.686  | 1.99 M |
| PAT-L (FT)         | 0.665      | 0.626 | 0.667  | 0.679  | 0.686  | 1.99 M |

**Supplemental Table 4. Evaluating models in predicting sleep abnormality from actigraphy.**

*Supplemental Table 4 Evaluating models in predicting sleep abnormality from actigraphy. In this dataset, the input is actigraphy, and the label indicates whether that participant has sleep abnormalities. Each model is trained on dataset sizes "500", "1,000", "2,500", and "3,429", (seen in the columns) and evaluated using AUC on a held-out test set of 2,000 participants. The "Avg AUC" represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that it was trained on smoothed data. LP stands for linear probing, and FT stands for end-to-end finetuning. An underline indicates the best baseline model. A bolded PAT model suggests that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. PATs outperform the baseline models in every dataset size in this task.*

| MODEL              | Avg Score* | n=500 | n=1000 | n=2500 | n=2800 | Params |
|--------------------|------------|-------|--------|--------|--------|--------|
| LSTM               | 0.489      | 0.472 | 0.489  | 0.497  | 0.497  | 15 K   |
| LSTM (Smooth)      | 0.506      | 0.496 | 0.494  | 0.519  | 0.515  | 15 K   |
| CNN-1D (Smooth)    | 0.517      | 0.461 | 0.540  | 0.537  | 0.528  | 10 K   |
| CNN-1D             | 0.522      | 0.500 | 0.533  | 0.530  | 0.525  | 10 K   |
| Wavelet Transform  | 0.523      | 0.550 | 0.531  | 0.512  | 0.500  | 10 K   |
| Conv LSTM (Smooth) | 0.547      | 0.476 | 0.561  | 0.573  | 0.580  | 1.75 M |
| Conv LSTM          | 0.550      | 0.507 | 0.534  | 0.579  | 0.581  | 1.75 M |
| PAT-M (LP)         | 0.557      | 0.488 | 0.597  | 0.564  | 0.577  | 1.00 M |
| PAT-M (FT)         | 0.559      | 0.489 | 0.591  | 0.566  | 0.589  | 1.00 M |
| PAT-S (LP)         | 0.560      | 0.547 | 0.552  | 0.565  | 0.575  | 285 K  |
| PAT-S (FT)         | 0.560      | 0.550 | 0.556  | 0.560  | 0.574  | 285 K  |
| PAT-L (LP)         | 0.582      | 0.495 | 0.595  | 0.618  | 0.620  | 1.99 M |
| CNN-3D (Smooth)    | 0.583      | 0.576 | 0.576  | 0.593  | 0.589  | 790 K  |
| CNN-3D             | 0.586      | 0.587 | 0.580  | 0.598  | 0.580  | 790 K  |
| PAT Conv-S (FT)    | 0.587      | 0.568 | 0.576  | 0.603  | 0.600  | 285 K  |
| PAT Conv-S (LP)    | 0.587      | 0.567 | 0.575  | 0.604  | 0.603  | 285 K  |
| PAT-L (FT)         | 0.589      | 0.541 | 0.577  | 0.618  | 0.620  | 1.99 M |
| PAT Conv-M (LP)    | 0.589      | 0.556 | 0.584  | 0.611  | 0.605  | 1.00 M |
| PAT Conv-M (FT)    | 0.594      | 0.576 | 0.585  | 0.609  | 0.606  | 1.00 M |
| PAT Conv-L (FT)    | 0.610      | 0.594 | 0.606  | 0.617  | 0.624  | 1.99 M |
| PAT Conv-L (LP)    | 0.611      | 0.594 | 0.606  | 0.618  | 0.625  | 1.99 M |

#### **Supplemental Table 5. Evaluating models predicting depression from actigraphy.**

*Supplemental Table 5 Evaluating models predicting depression from actigraphy. In this dataset, the input is actigraphy, and the label indicates whether that participant has depression (PHQ-9 scores > 9). Each model is trained on dataset sizes "500", "1,000", "2,500", and "2,800", (seen in the columns) and evaluated using AUC on a held-out test set of 2,000 participants. The "Avg AUC" represents the averaged AUC scores across each training dataset size. If the model name has "smoothing" after it, it denotes that it was trained on smoothed data. LP stands for linear probing, and FT stands for end-to-end finetuning. An underline indicates the best baseline model. A bolded PAT model suggests that it performed better than the best baseline, and a bolded and underlined PAT indicates the model with the best performance. PATs outperform the baseline models in every dataset size in this task.*

**Supplemental Table 6. Finding optimizing mask ratio, data preprocessing, and loss function.**

| (a)<br>MODEL<br>(Size,<br>Masking<br>Ratio) | Avg<br>Score* | n=500 | n=1000 | n=2500 | n=4769 |
|---------------------------------------------|---------------|-------|--------|--------|--------|
| Medium<br>0.50<br>(LP)                      | 0.703         | 0.661 | 0.672  | 0.693  | 0.785  |
| Medium<br>0.50<br>(FT)                      | 0.707         | 0.727 | 0.672  | 0.642  | 0.788  |
| Medium<br>0.25<br>(LP)                      | 0.724         | 0.689 | 0.698  | 0.765  | 0.744  |
| Medium<br>0.25<br>(FT)                      | 0.737         | 0.711 | 0.734  | 0.745  | 0.759  |
| Medium<br>0.75<br>(FT)                      | 0.743         | 0.662 | 0.742  | 0.798  | 0.772  |
| Medium<br>0.<br>75<br>(LP)                  | 0.747         | 0.686 | 0.753  | 0.781  | 0.767  |
| Medium<br>0.90<br>(LP)                      | 0.768         | 0.720 | 0.766  | 0.794  | 0.792  |
| Medium<br>0.90<br>(FT)                      | 0.773         | 0.753 | 0.764  | 0.786  | 0.788  |
| (b)<br>MODEL<br>(Size,<br>Smoothing)        | Avg<br>Score* | n=500 | n=1000 | n=2500 | n=4769 |
| Medium<br>Smooth<br>(LP)                    | 0.734         | 0.686 | 0.686  | 0.767  | 0.797  |
| Medium<br>Smooth<br>(FT)                    | 0.741         | 0.732 | 0.682  | 0.756  | 0.792  |
| Medium<br>Raw<br>(LP)                       | 0.768         | 0.720 | 0.766  | 0.794  | 0.792  |
| Medium<br>Raw<br>(FT)                       | 0.773         | 0.753 | 0.764  | 0.786  | 0.788  |
|                                             |               |       |        |        |        |
| (c)<br>MODEL<br>(Size,<br>Loss)             | Avg<br>Score* | n=500 | n=1000 | n=2500 | n=4769 |
| Medium,<br>MSE<br>MASK<br>(LP)              | 0.534         | 0.461 | 0.430  | 0.542  | 0.704  |
| Medium,<br>MSE<br>MASK<br>(FT)              | 0.541         | 0.437 | 0.515  | 0.560  | 0.652  |
| Medium,<br>MSE<br>ALL<br>(LP)               | 0.768         | 0.720 | 0.766  | 0.794  | 0.792  |
| Medium,<br>MSE<br>ALL<br>(FT)               | 0.773         | 0.753 | 0.764  | 0.786  | 0.788  |

*Supplemental Table 6 Finding optimizing mask ratio, data preprocessing, and loss function. All models are pretrained and end-to-end fine-tuned (FT) on the benzodiazepine training data. The dif erence between this table and Table 7 in the manuscript is that we also show linear probing (LP) results. For the experiments, 1,000 participants were removed from the training data to create an evaluation set, and these 1,000 participants are separate from the held-out test set seen in Table 1. The "Avg Score" metric is the average AUC score on the evaluation set after the medium model was trained on dataset sizes "500", "1,000", "2,500", and "4,769". (a) We test a PAT-M pretrained using MSE loss on every data point. We find that a higher mask ratio during pretraining leads to better results. (b) We test PAT-M pretrained on 90% masking and MSE loss on all data and find that smoothing does not improve performance (c) We test a PAT-M with 90% masking and find that MSE on only the masked patches decreases performance***.**

#### **Supplemental Figure 1: Attention Weight Patterns and Daily Activity Trends for a Non-Benzodiazepine Participant**

![](_page_36_Figure_1.jpeg)

![](_page_36_Figure_2.jpeg)