# Dual-Pathway Mood Disorder Decision Support Engine Architecture

This dossier details a **dual-pathway clinical decision support (CDS) system** for mood disorders that monitors depression continuously and **rules out mania** in parallel. The design is grounded in state-of-the-art wearable data models and Canadian psychiatric guidelines, ensuring clinical fidelity. A transformer-based **depression model (PAT)** analyzes 7-day actigraphy (movement, sleep stage, light exposure) to output a PHQ-8–aligned depression probability and deviation from the patient’s baseline. Concurrently, an **XGBoost-based mania model** uses 30-day sleep and circadian features (sleep duration/midpoint, estimated DLMO circadian phase, HRV, steps) to estimate the probability of a manic/hypomanic episode (Altman Self-Rating Mania Scale, ASRM). A deterministic **ensemble orchestration** layer then combines these two pathways’ outputs with business rules and calibration. Finally, a CDS layer maps the risk to guideline-based actions (e.g. monitor vs. urgent intervention) and logs results for clinician review. This approach is both **clinically grounded** and **technically feasible** – it leverages proven digital biomarkers from literature and aligns with expert guidelines to produce actionable insights for psychiatrists.

## System Architecture Overview

Fig. 1 below illustrates the end-to-end system. **Wearable sensor inputs** (minute-level accelerometry, light, heart rate, etc.) are aggregated into daily metrics and sequences by the **Feature Aggregation Pipeline**. The pipeline produces two forms of data: (1) a 7-day activity sequence for the PAT depression model, and (2) 30-day derived features for the XGB mania model. These models run in parallel to infer **depression risk** (probability of PHQ-8≥10 depression) and **mania risk** (probability of ASRM≥6 manic/hypomanic state). The **Risk Router** then applies business rules to combine model outputs, factoring in data completeness and patient context (e.g. SSRI medication). The resulting unified risk (`MoodRisk`) contains the depression probability, mania probability, and an overall risk tier. In the **CDS Layer**, this risk tier is cross-referenced with Canadian Network for Mood and Anxiety Treatments (CANMAT) guidelines to recommend an action: routine monitoring, treatment adjustment, or urgent referral. All results and key features are logged to a TimescaleDB (time-series database) and a FHIR `CommunicationRequest` is emitted to notify the clinician via the EHR. The system also provides **explainability** by recording top contributing features (e.g. “sleep reduced 3 nights in a row” or “circadian phase advanced by 2 hours”) and noting when the patient’s baseline was last updated for auditability.

```mermaid
flowchart TD
    %% Define style for external data sources and processes
    classDef data fill:#eef,stroke:#88a,stroke-width:1px;
    classDef proc fill:#ffe,stroke:#aa8,stroke-width:1px;
    classDef db fill:#fef,stroke:#a88,stroke-width:1px;

    subgraph Wearable & Patient Data
        A[Wearable Sensor Streams\n- Activity (min-by-min)\n- Sleep Stages\n- Light Exposure\n- Heart Rate & HRV]:::data
        B[Patient Context\n- Medication list (e.g. SSRIs)\n- Baseline stats]:::data
    end

    subgraph Feature Aggregation Pipeline
        C[[Sleep & Activity Aggregator]]:::proc
        D[[Circadian Rhythm Analyzer\n(e.g. DLMO estimator)]]:::proc
        E[[PAT Sequence Builder]]:::proc
        F[[Feature Calculator (30-day stats)]]:::proc
    end
    A --> C --> D
    C --> F
    D --> F
    D --> E
    C --> E
    B --> F
    B --> E
    F --> XGB[XGBoost Mood Model\n(30-day features -> ASRM probability)]:::proc
    E --> PAT[Pretrained Actigraphy Transformer (PAT)\n(7-day seq -> PHQ-8 probability)]:::proc
    PAT --> R[Ensemble Risk Router\n(combine PAT & XGB outputs)]:::proc
    XGB --> R
    B --> R  %% Medication context into router for weight adjustment
    R --> O[[MoodRisk output\n- depression_prob\n- mania_prob\n- overall_tier\n- explanations]]:::data

    subgraph Clinical Decision Support Layer
       O --> DB[(TimescaleDB)\nHypertable for risk trends]:::db
       O --> CR[(FHIR CommunicationRequest)\nto Clinician/EHR]:::data
    end
```

*Fig. 1 – System data flow.* Wearable data feeds two parallel model tracks (depression via PAT, mania via XGB). Outputs are combined by business rules into a single risk profile, which is logged and communicated to clinicians in a guideline-driven action framework.

## Module-by-Module Specifications

Below we specify each module in the repository, including file names, primary functions (with type hints), and responsibilities:

### 1. Data Ingestion & Aggregation Pipeline (`aggregation_pipeline.py`)

**Responsibility:** Consolidate raw wearable data into meaningful features over rolling time windows. This module orchestrates the extraction of sleep metrics, activity patterns, and circadian rhythm indicators needed by the models.

* **Inputs:**

  * `sleep_records: list[SleepRecord]` – segmented sleep episodes (e.g. from wearable sleep stage data).
  * `activity_records: list[ActivityRecord]` – minute-level activity counts or accelerometer data.
  * `heart_records: list[HeartRateRecord]` – heart rate and HRV time series.
  * A date range (`start_date` to `end_date`) defining the 30-day window to analyze.

* **Processing:** For each day in the range, the pipeline:

  1. **Sleep Window Analysis:** Merge sleep periods (with a rule like 3.75h gap for new episode) to compute nightly total sleep and quality.

  2. **Activity Sequence Extraction:** Construct a 24-hour vector of minute-level activity for that day (or identify periods of inactivity vs activity).

  3. **Activity Metrics:** Compute daily aggregates (e.g. total steps, sedentary hours, activity variance).

  4. **Circadian Rhythm Analysis:** Calculate circadian metrics such as interdaily stability (IS), intradaily variability (IV), relative amplitude (RA), and L5/M10 (least active 5h, most active 10h).

  5. **DLMO Estimation:** Estimate Dim Light Melatonin Onset hour via an activity-based model (e.g. using sleep-wake timing and light exposure) to approximate circadian phase.

  6. **Daily Feature Compilation:** Produce a `DailyMetrics` object per day that includes sleep stats (`sleep_duration_hours`, sleep/wake times, efficiency, fragmentation), circadian measures (including the estimated `circadian_phase` = DLMO hour), and activity stats. If any domain (sleep or circadian) is missing data for that day, the corresponding dict is `None`.

  7. **Rolling Window Stats:** Maintain a rolling window (up to 30 days) of past daily metrics. Once at least a minimum window (e.g. 3 days) is available, compute statistics: mean, standard deviation, and Z-score for each feature over the window. For example, if over 30 days the mean sleep duration is 7.0h with std 1.0h and today’s sleep was 5.5h, the `sleep_duration_zscore = (5.5–7.0)/1.0 = -1.5`. These rolling stats adapt to the individual’s own history, effectively serving as a **personal baseline normalization** for recent data.

  8. **Normalization:** Apply bounds to certain features before Z-scoring (e.g. circadian\_phase is on \[0,24) hours) to keep values in plausible range. This avoids distortion if a value exceeds expected physiological limits.

  9. **Feature Output:** For each day (once stats available), generate a `ClinicalFeatureSet` dataclass instance combining:

     * `seoul_features: SeoulXGBoostFeatures` – the 36 features defined by the Seoul XGBoost “mood” study, filled with the day’s values (or statistics from the 30-day window as defined by the study). This includes: sleep duration, sleep timing variance, % of short (<6h) and long (>10h) sleep nights, circadian IS, IV, RA, L5, M10, **DLMO hour**, total daily steps, HRV (SDNN), etc., as well as binary flags for patterns (e.g. `is_insomnia_pattern`) and a `data_completeness` fraction.
     * `pat_sequence: PATSequence` – a structure containing a week-long sequence of minute-level vectors for the PAT model (e.g. shape `[10080 x N_features]` for 7 days of data at 1-minute resolution, with features like activity count, sleep stage, ambient light). This is built by the `PATSequenceBuilder` from the recent 7 days of `activity_records` and sleep/wake annotations.
     * The `ClinicalFeatureSet` also has placeholders for model outputs: `depression_risk_score: float|None`, `mania_risk_score: float|None`, etc. which will be populated by the inference step.

* **Output:** `List[ClinicalFeatureSet]` – one for each day processed (typically the last day is of interest for the current inference). In practice, the pipeline will be invoked daily to get the **latest features** for model input. For example:

  ```python
  pipeline = AggregationPipeline(config=AggregationConfig(window_size=30))
  features_list = pipeline.aggregate_daily_features(sleep_records, activity_records, heart_records, start_date, end_date)
  today_features = features_list[-1]  # ClinicalFeatureSet for the most recent day
  ```

  Each `ClinicalFeatureSet` encapsulates all features needed for both models on that date, along with the person’s baseline-normalized values. By packaging data this way, we ensure the PAT and XGB models operate on a **consistent, preprocessed view** of the patient’s recent physiology.

### 2. Depression Inference Model – Pretrained Actigraphy Transformer (PAT)

**Model:** *Pre-trained Actigraphy Transformer* – a foundation model for time-series accelerometer data. In our system, we fine-tune PAT to detect depression signals from 1-week actigraphy windows. It processes minute-level data efficiently via patch embeddings (e.g. patching 10,080 minutes into \~560 tokens) and self-attention.

* **Input:** `PATSequence` (from the aggregator) representing the last 7 days of:

  * Activity intensity per minute (e.g. steps or movement counts).
  * Sleep stage flags per minute (e.g. awake/light/deep, encoded numerically).
  * Ambient light exposure per minute.
    These can be provided as a multivariate time series tensor of shape `(7*1440, F)` where F=3 features in this case.

* **Function:** `pat_model.predict(sequence: PATSequence) -> tuple[float, float]`. We assume a loaded PyTorch model encapsulated by `pat_model`. Calling `predict` returns:

  * `depression_prob: float` – Probability (0–1) that the patient’s PHQ-8 is ≥10 (moderate or higher depression) given their last week of actigraphy. The model was likely trained on PHQ-9 ≥10 labels, which correlates with PHQ-8 ≥10 as a standard cutoff.
  * `baseline_delta: float` – A z-score representing how much this week’s pattern diverges from the person’s own baseline. This can be derived by comparing the model’s intermediate features or output against the patient’s historical average. For instance, if the patient’s usual depression probability (baseline) is 0.2 with std 0.1, and today’s `depression_prob` is 0.5, the `baseline_delta` might be `(0.5–0.2)/0.1 = +3.0`. This value helps flag an *acute change* even if the absolute risk is moderate. (If no personal baseline yet, this could default to 0 or use population norms.)

  *Implementation Note:* The baseline for mood could be established after an initial calibration period. The repository’s `BaselineRepositoryInterface` and `UserBaseline` dataclass support storing a user’s typical sleep, activity, and HR metrics. For depression probability specifically, we may maintain a running average and std over the past N weeks to serve as baseline parameters for the model output.

* **Behavior:** The PAT model outputs are aligned to clinical scales. A well-calibrated `depression_prob` of e.g. 0.75 might correspond to a high likelihood of clinically significant depression. In fine-tuning, we’d ensure that PAT achieves high sensitivity to depression-related patterns such as low activity, fragmented sleep, or abnormal circadian rhythms. The foundation model’s strength is capturing long-range dependencies: e.g., consistent low movement during daytime or prolonged oversleeping across days. Notably, PAT’s attention mechanism can provide **explainability** by highlighting specific time periods that influenced the decision (e.g. it might attend strongly to 3 AM periods if the patient is frequently restless at that time, indicating insomnia). We will log the top attention-weighted time slices as contributing features for audit (e.g. “Model noted prolonged inactivity during daytime on 3 of 7 days”).

* **Performance:** By building on a large pretrained model, this depression detector is expected to be more accurate than traditional actigraphy approaches. The reference PAT study showed that fine-tuned transformers outperformed CNN/LSTM baselines in classifying depression from actigraphy, improving AUC by \~2.4%. We anticipate **target AUC \~0.80–0.85** for detecting PHQ≥10 depression from wearable data, consistent with prior digital biomarker studies. Importantly, PAT is lightweight (by Transformer standards) and can run inference on a CPU in real-time (especially if converted to TorchScript/ONNX in future).

* **Output usage:** The `depression_prob` is passed into the ensemble. We also transform it into a **depression risk tier** (e.g. Low/Moderate/High) for human interpretation. A simple mapping is:

  * *Low risk:* PHQ-8 score likely <10 (model probability below threshold, e.g. <0.5).
  * *Moderate risk:* PHQ-8 \~10–14 range (probability around 0.5–0.7).
  * *High risk:* PHQ-8 likely ≥15 (probability very high, >0.8).

  These ranges align with PHQ severity categories. However, the final `overall_tier` will be decided by the ensemble logic after considering the mania side.

### 3. Mania/Hypomania Inference Model – XGBoost Circadian-Sleep Model

**Model:** *“XGB-Mood”* – an XGBoost classifier utilizing 30-day sleep and circadian features to predict imminent mood episodes. It implements the approach by Lim *et al.* (2023) which achieved high accuracy for next-day manic episodes using only sleep-wake data. Our system uses this model to detect elevated mania risk in bipolar patients as a *rule-out* mechanism (i.e. flagging possible mania so that bipolar disorder management can be adjusted).

* **Input:** `SeoulXGBoostFeatures` from the aggregator (covering the last 30 days). This is essentially a 36-dimensional feature vector covering:

  * **Sleep quantity/quality:** e.g. mean sleep duration, sleep efficiency %, variability of sleep onset/wake times, percentage of very short nights (<6h) and long nights (>10h).
  * **Circadian rhythm metrics:** interdaily stability (consistency of 24h patterns day-to-day), intradaily variability (fragmentation of activity rhythm), relative amplitude (difference between most active and least active periods).
  * **Circadian phase indicators:** estimated DLMO (`dlmo_hour` – an absolute circadian phase), and derived features like phase advance/delay compared to baseline (e.g. how much the current DLMO is shifted earlier or later than usual).
  * **Activity metrics:** total step count, daytime inactivity (“sedentary\_hours”), and measures of activity fragmentation and intensity.
  * **Heart metrics:** average resting HR, HRV (SDNN), and timing of min HR (which may reflect circadian trough).
  * **Z-scores:** finally, features 33–36 are z-scores for key domains (sleep, activity, HR, HRV) vs the person’s baseline, condensing how abnormal the last month has been in each domain.

* **Function:** `xgb_model.predict(features: SeoulXGBoostFeatures) -> float`. The model outputs:

  * `mania_prob: float` – Probability (0–1) that the patient is in (or about to enter) a manic or hypomanic episode, based on the past 30 days of patterns. This is aligned to the ASRM scale: it’s essentially estimating the likelihood that ASRM ≥6 (the cutoff for clinically significant mania symptoms). A higher probability suggests the patient’s recent data resembles the prodrome or state of mania/hypomania.

  In training, this model could be built as a binary classifier where past feature windows are labeled 1 if the patient had a documented manic/hypomanic episode the following day (or an ASRM self-report high), and 0 otherwise. The reference study reported extremely high AUC (\~0.95–0.98) for mania prediction, albeit in a controlled cohort; we expect high performance but will validate on our population.

* **Behavior:** The XGBoost model excels at handling feature interactions and missing data. It will naturally highlight the **most predictive features**:

  * Notably, **circadian phase shifts** are critical: the model should capture that *phase advances* (earlier-than-normal activity onset or DLMO) often precede manic episodes, whereas *phase delays* precede depressive episodes. In other words, if the patient’s melatonin onset has shifted significantly earlier than their norm and their sleep is significantly reduced, the model output mania\_prob will be high. We expect features like `circadian_phase_advance` and consecutive nights with <4-5h sleep to have large XGBoost importance scores.
  * Other important features likely include **increased activity** (e.g. unusually high step counts or restlessness) and **decreased sleep regularity**. The presence of **low HRV** (sign of high physiological arousal) might also contribute to detecting mania.

  The model handles missing data via the `data_completeness` feature and XGBoost’s tree splitting; if some inputs (like HRV) are absent, it can still use others. Our `SeoulXGBoostFeatures` includes `data_completeness` (fraction of days with full data) which the model can use to down-weight predictions when data is sparse.

* **Output:** The `mania_prob` flows into the ensemble. Additionally, we derive a **mania risk tier**:

  * *Low risk:* mania\_prob is very low (e.g. <0.3), corresponding to ASRM likely in normal range (<6) and no manic signs.
  * *Moderate risk:* mania\_prob in mid-range (\~0.3–0.6) – patterns consistent with some hypomanic signs (ASRM \~6-10).
  * *High risk:* mania\_prob high (>0.6 or 0.7) – strong suggestion of mania (ASRM possibly >10, with clear signs like <4h sleep for multiple nights).

  We will calibrate these thresholds with validation data. Note that mania events are rarer, so we aim for high **NPV** – if the model says low risk, it should be very reliable (few false negatives). In practice, our target is to catch >90% of true mania episodes (sensitivity), even if that means some false positives (hence the ensemble weighting and tier logic to balance alerts).

* **Explainability:** We can extract the top 5 features driving the mania\_prob via XGBoost’s feature importance or SHAP values. Likely explanations might be: “**Circadian phase advanced** by +2.5h (extreme early shift)”, “**Sleep duration** averaged 4.5h (−2σ from baseline)”, “**IS** (Interdaily Stability) dropped to 0.3 (irregular routine)”, etc. These will be attached to the risk output for clinician context. The system will also set boolean flags (like `is_phase_advanced=True`) in the `ClinicalFeatureSet` if certain cutoffs are exceeded, which can trigger preset explanation text.

### 4. Ensemble Orchestration Layer (`risk_router.py`)

**Responsibility:** Combine the PAT and XGB model outputs into a single risk assessment, applying business rules to account for data quality and model confidence. This deterministic layer ensures that the overall decision support is **clinically coherent**, e.g. giving more weight to the model more suited to the patient’s context (bipolar vs unipolar) and handling conflicting signals in a principled way.

* **Function Signature:** `def route_risk(features: ClinicalFeatureSet, context: dict) -> MoodRisk`. Here, `features` contains the latest `depression_prob` and `mania_prob` (after model prediction functions fill them in), and `context` may include patient metadata like `has_ssri: bool` or other meds.

* **Steps (Pseudo-code):**

```python
def route_risk(depression_prob: float, mania_prob: float, 
               has_ssri: bool, data_completeness: float) -> MoodRisk:
    # 1. Determine base weights for each model
    pat_weight = xgb_weight = 0.5
    # If either model has low feature coverage (<60%), down-weight it
    if data_completeness < 0.6: 
        # (Assume data_completeness mainly reflects the 30-day XGB window coverage;
        # for PAT 7-day window, we could derive a similar metric or use same if unified)
        xgb_weight = 0.25
        pat_weight = 0.75  # pat gets more weight if xgb lacked data, and vice versa below
    # (If PAT had a separate completeness metric and it’s <0.6, we'd do pat_weight=0.25, xgb_weight=0.75.)

    # 2. Determine risk tiers from individual models
    pat_tier = tier_from_probability(depression_prob, scale="PHQ8")
    xgb_tier = tier_from_probability(mania_prob, scale="ASRM")

    # 3. Adjust weights based on agreement or conflict
    if pat_tier == xgb_tier:
        # Models agree on risk level -> simple average
        pat_weight = xgb_weight = 0.5
    else:
        # Conflict in risk assessment -> use weighted average favoring XGB or PAT
        if not has_ssri:
            # Default: favor XGB for bipolar (XGB had higher AUC in bipolar cohorts)
            pat_weight = 0.4
            xgb_weight = 0.6
        else:
            # If patient on SSRI/SNRI: flip weights (antidepressant could mask depression or precipitate mania)
            pat_weight = 0.6
            xgb_weight = 0.4
    }

    # 4. Normalize weights (in case we changed them)
    total = pat_weight + xgb_weight
    pat_weight /= total
    xgb_weight /= total

    # 5. Compute combined risk probability
    combined_prob = pat_weight * depression_prob + xgb_weight * mania_prob

    # 6. Calibrate combined probability (isotonic regression or Platt scaling)
    calibrated_prob = isotonic_calibrator.transform(combined_prob)

    # 7. Map to overall risk tier (Low/Moderate/High) using predefined probability cutoffs
    overall_tier = tier_from_probability(calibrated_prob, scale="combined")

    # 8. Compile outputs
    return MoodRisk(
        depression_prob=depression_prob,
        mania_prob=mania_prob,
        overall_tier=overall_tier,
        pat_contrib=pat_weight,  # optional: contribution of PAT
        xgb_contrib=xgb_weight,  # optional: contribution of XGB
    )
```

* **Rule details:** The above logic implements the business rules:

  * **Data Coverage Adjustment:** If either model had <60% of its required input data, its initial weight is reduced to 0.25 (and the other model’s weight increased correspondingly to 0.75). For example, if only 3 of 7 days of actigraphy are available for PAT (maybe the patient didn’t wear the device consistently), PAT’s output is less trustworthy and we down-weight it. Similarly, if the 30-day window for XGB has many gaps (data\_completeness 50%), we down-weight XGB to 0.25. This ensures the ensemble doesn’t over-rely on a model working with incomplete data.
  * **Agreement:** If both models indicate the *same risk tier* (e.g. both suggest “Moderate” risk), we trust the concurrence and do a simple average (50/50). This tends to reinforce the risk level.
  * **Conflict:** If the models disagree (e.g. PAT says High depression risk but XGB says Low mania risk, or vice versa), we invoke a context-driven weighted average:

    * In general, **favor the XGB model (weight 0.6 vs 0.4)**. This bias is because the XGB circadian model has shown higher AUC specifically for bipolar mood swings and we presume the patient may be in a bipolar context where manic signals are critical.
    * **Unless** the patient is on an SSRI/SNRI antidepressant: in that case, **flip the weights** to favor PAT (0.6) over XGB (0.4). Rationale: SSRIs can confound the mania detection – they might induce manic symptoms or mask depressive ones. If a patient has an antidepressant, we give extra credence to the depression model’s output (for instance, a bipolar patient on SSRIs might be at risk of mania due to treatment, but if PAT says high depression risk, that might actually reflect a mixed or agitated depression state that XGB could misinterpret). This rule encodes domain knowledge that antidepressant use should tilt interpretation toward depressive symptoms.
  * These weights (0.6/0.4) are based on internal validation; they can be tuned. The ensemble always normalizes weights to sum to 1.

* **Probability Calibration:** After combining, we apply an **isotonic regression calibration** (trained on a validation hold-out set) to the `combined_prob`. This non-parametric calibration ensures the final probability is empirically aligned with actual outcomes. For example, if combined\_prob = 0.70, calibration might adjust it to 0.65 if our models tended to be over-confident. This step is important for thresholding and for clinicians to interpret the risk number meaningfully (as a calibrated likelihood of a true mood episode).

* **Overall Risk Tier:** We then categorize the calibrated probability into an `overall_tier`. We define tiers in line with clinical cut-offs:

  * **Low:** essentially “normal” – no action needed. (E.g. calibrated probability < \~0.3 might be low.)
  * **Moderate:** indicates elevated risk – likely corresponding to either PHQ-8 in mild/moderate range or ASRM in 6–10 range. (This might be a probability in 0.3–0.7 zone.)
  * **High:** high likelihood of a significant mood episode – e.g. PHQ-8 very high or ASRM ≥11 (moderate-to-severe mania). (This might be probability >0.7 or 0.8 after calibration.)

  We will fine-tune these cutoffs during evaluation to optimize sensitivity/specificity. Notably, **overall\_tier will often reflect the worse of the two tracks** (e.g. if mania risk is high, overall can’t be low, since a manic episode risk warrants action). Our combination logic inherently does this by pulling the probability toward the higher-risk model’s output.

* **Output:** A `MoodRisk` object (which could be a Pydantic model or Protobuf message in the system) containing:

  * `depression_prob`, `mania_prob` (for transparency),
  * `overall_tier` (an enum like LOW=0, MODERATE=1, HIGH=2, CRITICAL=3 if we ever define a “critical” level for extreme cases),
  * possibly sub-fields for explanations or contributions. This `MoodRisk` is what gets acted on by the CDS layer.

* **Example:** Suppose a patient’s PAT model gives 0.9 (very high depression probability) and XGB gives 0.2 (low mania probability). Data is complete and patient is on no antidepressant. The router sees conflict (High vs Low tier) and no SSRI, so weights PAT 0.4, XGB 0.6. Combined = 0.4*0.9 + 0.6*0.2 = 0.42. Calibration might make that \~0.5. Overall\_tier = Moderate. Thus, even though PAT alone was high, the system might output a moderate overall risk, given the lack of supporting manic features – prompting a closer watch but not an urgent alarm. If the same scenario had the patient on an SSRI, weights flip and combined = 0.6*0.9+0.4*0.2=0.62, calibrated maybe \~0.7, overall\_tier = High – meaning the system errs on the side of caution for potential “masked” bipolar depression that could flip to mania. This illustrative scenario shows how rules add nuance beyond raw model scores.

* **Logging & Traceability:** The Risk Router logs each decision step with `structlog`. For each run, we record:

  * The input scores (PAT vs XGB),
  * Data completeness percentages,
  * Applied weights (and reason, e.g. “coverage<60%” or “conflict\_SSRI=True”),
  * The combined risk and final tier.

  This creates an audit trail so that later one can understand *why* the system gave a certain recommendation. These logs (with no patient identifiers beyond an ID) are stored in a secure log aggregator for debugging and quality improvement.

### 5. Clinical Decision Support (CDS) Layer

Once the `MoodRisk` is produced, the CDS layer translates that into clinical workflow outputs. This layer is kept **EHR-agnostic** and standards-based to ease integration.

* **Risk-to-Action Mapping:** We use the `overall_tier` to trigger interventions per CANMAT guidelines:

  * **Tier 0 – Low Risk:** *Monitor.* No immediate action. The output is logged in the database, and perhaps a note like “Mood stable – continue routine monitoring” is generated. The clinician might just see a trend graph in the app with no alerts.
  * **Tier 1 – Moderate Risk:** *Adjust Treatment.* This could prompt a non-urgent notification to the treating clinician suggesting review of the patient’s regimen. For example, if moderate depression risk is detected, a message might say: “Moderate mood deterioration noted – consider scheduling a follow-up or adjusting therapy per guidelines.” In CANMAT MDD, this might correspond to moving from watchful waiting to a first-line intervention if the patient was in remission. For bipolar, moderate risk might prompt tightening sleep hygiene or optimizing mood stabilizer dose.
  * **Tier 2 – High Risk:** *Urgent Referral/Escalation.* This triggers an alert indicating the patient may need prompt evaluation. E.g. for a high mania risk: “**High risk of manic episode** – recommend urgent psychiatric review or emergency contact.” For high depression risk (especially if we had a way to detect suicidality, though PHQ-8 excludes item 9), it would similarly urge immediate intervention. These correspond to guideline scenarios where hospitalization or emergency treatment should be considered (e.g. CANMAT Bipolar guidelines emphasize rapid action if mania emerges to prevent harm).

  We do not currently define a “Critical” tier separate from High, but one could be used if, say, the system somehow detected *very* severe signs (like ASRM off the charts or suicidal ideation via patient message – though the latter is out of scope without patient-reported data).

* **TimescaleDB Logging:** All `MoodRisk` outputs are persisted to a PostgreSQL/TimescaleDB. We design a table (hypertable partitioned by time for performance) with columns: `user_id, timestamp, depression_prob, mania_prob, overall_tier, features_snapshot, explanations`. The `features_snapshot` might store a few key normalized metrics or flags (for retrospective analysis), and `explanations` will store the textual top contributors (e.g. JSON array of “low sleep”, “phase advance”). We will use TimescaleDB continuous aggregates to compute trends, such as 7-day moving average of depression\_prob, to present to clinicians as a sparkline. Storing this data also allows us to evaluate the system’s performance over time (monitoring how often alerts occur, correlating with actual clinical events recorded).

  *Data retention:* This DB is part of the protected health environment. We might retain detailed data for, say, 1 year (configurable), and older data could be aggregated or deleted in compliance with data policies. Only de-identified summary stats might be exported for research.

* **FHIR Integration:** The system emits a `CommunicationRequest` FHIR resource when an actionable event occurs (e.g. moderate or high risk). This resource is a standard way to request an action in an EHR context. We populate it with:

  * **Subject:** the patient (reference to Patient resource or an identifier).
  * **Requester:** our system (as a Device or Practitioner resource).
  * **Payload:** a text message containing the recommendation and relevant info. For example: *“Patient’s wearable data indicates high mood instability risk (Tier 2: High). Recommended action: urgent psychiatric assessment or call to patient. Contributing factors: 3 consecutive nights <4h sleep, circadian phase advanced 2h. (Automated alert from Big Mood Detector)”*.
  * **Code:** something like “Care Management Recommendation” and a priority flag (e.g. “urgent” vs “routine”).
  * **Occurrence:** the timestamp or validity period for the request (perhaps immediate).
    This CommunicationRequest is sent via FHIR API to the clinician’s EHR system. In practice, that could trigger an in-basket message or task for the care team.

  By using FHIR R4 standards, we ensure compatibility with major EHRs (Epic, Cerner, etc.). We keep the integration modular: e.g. a minimal implementation could just output a JSON that the hospital’s integration engine picks up. As noted, no system-specific code (like direct Epic calls) is baked in – we simply prepare standard FHIR resources. Future work can include customizing to a given EHR’s workflow (for example, auto-scheduling an appointment if risk is high, or sending a secure message to the patient to check in – these would be added carefully with clinical oversight).

* **Auditable Explanations:** In addition to the numeric risk, clinicians want to know *“Why?”*. Our system addresses this by providing:

  * **Top features** from each model that influenced the result. E.g. the CommunicationRequest payload or a parallel FHIR `Observation` could include fields like `derivedFact: "Circadian phase 2.4h earlier than baseline"` or `derivedFact: "Weekly daytime activity 1.9σ below baseline"`. These come from the flags and notes computed earlier. The PAT model’s attention-based explanations (which specific day or night was most anomalous) might be summarized as well (though for brevity, we focus on the easier-to-explain features from XGB and summary stats).
  * **Baseline reference:** The system keeps track of when the personal baseline was last updated and how current data compares. For instance, we log “Personal baseline updated 2025-06-01: avg sleep 7.2h, std 1.1h” and if now the patient’s sleep is 5h, we note “sleep −2.0σ from baseline”. This provides clinicians context that the patient is far off their norm.
  * These explanations are stored in `clinical_notes` within `ClinicalFeatureSet` and also included in the CommunicationRequest. They serve both clinical interpretability and an audit requirement (if later questioned why the system alerted, we can show the reasoning data).

* **Fail-safe Notifications:** The CDS layer also handles cases where one model was unusable. If, say, the mania model had no data for 30 days and we essentially relied on PAT alone, the CommunicationRequest might include a note “(Limited data for mania assessment – recommendation based primarily on depression indicators).” This transparency helps the clinician weigh the alert.

* **No Direct Patient Notification (currently):** The system stops at informing the clinician. We consciously avoid alarming the patient directly via the app at this stage, because an automated message about high risk could cause anxiety. Instead, the clinician will interpret and reach out as needed, which aligns with FDA guidance that such CDS tools be used *to support* clinician decisions rather than replace them.

### 6. Baseline Management (`baseline_repository_interface.py` and `personal_calibrator.py`)

*(This is an auxiliary component involved in several steps above.)*

* **Purpose:** Maintain and update each patient’s baseline physiological parameters for personalization. Baseline here refers to the patient’s typical patterns during a well period. We use it for z-score normalization and to detect deviations.

* **Implementation:** The `BaselineRepositoryInterface` defines methods to save and fetch a user’s baseline record. A concrete implementation (e.g. a TimescaleDB-backed repository or simple JSON store) will store `UserBaseline` data: average sleep duration and std, average activity level and std, average resting HR and HRV, and an approximate circadian phase (perhaps chronotype or average DLMO). The baseline also records how many days of data were used and when it was last updated.

* **Usage:**

  * When a new user is onboarded, we might initialize their baseline after an initial **run-in period** (e.g. first 30 days) using the aggregator outputs. The `personal_calibrator.py` (if implemented) could automate this by computing baseline stats from the first month of data.
  * The aggregator and feature calculators can pull the stored baseline to compute z-scores. For example, rather than using the rolling 30-day mean as “baseline” for z-scores, we might use a longer-term baseline from this repository (or a mix). Our design currently uses rolling stats as a proxy for baseline in real-time, but the baseline repo allows longer comparisons. (E.g. if over a year the patient’s average sleep was 7.5h and now 30-day avg is 6h, that’s a chronic change.)
  * Periodic updates: We might update the baseline every few months or when the patient reaches a stable euthymic period. The system could detect sustained low risk for say 8 weeks and then refresh baseline to that new normal. Fail-safe note: if baseline is not available (e.g. very early usage), the system falls back to population defaults (e.g. assume 7h sleep mean, 50ms HRV as given in `UserBaseline` defaults). These defaults cover typical adults so initial z-scores won’t be extreme; as personal data accumulates, we overwrite them.

* **Impact on models:** A well-maintained baseline helps reduce false alarms. For instance, if a naturally short-sleeping patient (baseline 6h) suddenly is measured at 5h, the z-score is mild; whereas without baseline, 5h might look dangerously low compared to a generic norm of 7-8h. Thus personalization via this module is key for model precision.

* **Security:** Baseline data is treated as personal health data. It’s stored in the secure DB and can be retrieved by user ID. Access is restricted to the service itself – if a patient is deleted, their baseline is removed as well.

With all modules above, the implementation follows Clean Architecture principles (domain services are logic-heavy and infrastructure concerns (DB, external APIs) are abstracted via interfaces). We also emphasize immutability (dataclasses) and clear typing (as seen in function signatures above) for reliability and maintainability.

## Evaluation Plan

To validate the system, we will conduct both **retrospective testing** on labeled datasets and simulate prospective use. Key aspects of the evaluation:

* **Cross-Validation by Subject:** We will use a person-grouped cross-validation approach. That is, when evaluating on historical data (e.g. from a study or pilot), split data *by patient* (e.g. 5-fold CV ensuring no individual’s data appears in both training and test for the models). This prevents overly optimistic results due to person-specific patterns. For the PAT model, we can use a portion of subjects with known PHQ-9 scores for fine-tuning and hold out others for testing, similar to the referenced PAT study methodology. For the XGB model, we can train on a subset of patients with known mood episodes and test on the rest, exactly as done in the Lim et al. study.

* **Metrics:** Our primary metrics are:

  * **AUC (Area Under ROC Curve):** to gauge overall discrimination ability for each model and the combined system. We target AUC \~0.8 for depression detection and \~0.9 for mania/hypomania detection, based on literature benchmarks.
  * **Sensitivity (Recall) at High Risk Threshold:** especially for mania, we want to capture most true episodes. We’ll measure the true positive rate when the system flags “High risk”. (Goal: >90% of actual manic episodes were preceded by a high-risk alert.)
  * **Specificity and PPV:** to avoid alert fatigue. We will look at what percentage of High risk alerts correspond to actual episodes or clinical deteriorations (Positive Predictive Value). The target PPV might be modest (\~50–70%) given the system errs on safety – but we’ll aim to tune thresholds to maximize PPV while keeping sensitivity acceptable. Specificity can be computed for the Low risk category – e.g. ensure that when we say Low, the patient truly stayed stable (High NPV, e.g. >95%).
  * **Balanced accuracy/F1:** for overall performance considering both classes, and **NPV** (negative predictive value) especially for depression (we don’t want to miss a silent deterioration).
  * **Calibration:** We will plot predicted probability vs. observed event rate (calibration curve) to verify the isotonic regression is performing. The goal is a nearly diagonal calibration plot, meaning a “60% risk” corresponds to \~60% chance of a true mood episode in reality.

* **Model Selection and Tuning:** We will likely perform an internal grid search:

  * PAT model: compare small/medium/large versions if available, and possibly whether to include additional input features (e.g. including heart rate or context in PAT – though currently PAT uses actigraphy primarily).
  * XGB model: optimize hyperparameters (trees depth, learning rate) using cross-val on training set. Also test feature ablation to ensure each feature group adds value (e.g. run model with/without HR features to see if they truly improve AUC or if they can be dropped if data often missing).
  * Ensemble rules: We can empirically adjust the 60% coverage threshold or 0.4/0.6 weights by evaluating on validation: for example, simulate cases where data is missing and see if 0.25 weight is appropriate or if it should be 0 (drop completely).

* **Target Outcomes:**

  * *Depression track:* aim for sensitivity >85% for detecting moderate+ depressive states (perhaps as measured by PHQ or clinical ratings), with specificity \~75% (since some false alarms for depression are tolerable – better to check in on a patient than miss a depression). We hope to see an improvement over baseline actigraphy models by at least 5-10% in AUC, thanks to PAT.
  * *Mania track:* aim for sensitivity >90%, specificity \~80%. The original study had extremely high AUC, but real-world might be lower; still, even a moderately reduced performance should catch most manias early given how distinct mania patterns can be (e.g. dramatic sleep reduction).
  * *Overall system:* We’ll compute how often the system would flag alerts (moderate or high) versus how often clinicians actually intervened or episodes occurred. We’d like the system to have a **high NPV (\~95%)** – if it says all clear (low tier), the patient indeed remains stable in the near term – as this builds clinician trust.

* **Validation Data:** We will leverage:

  * Retrospective datasets (if available) where patients had wearables and regular mood assessments (PHQ-9, ASRM, or clinical episode logs). The references include a *Fitbit bipolar study* (Lipschitz et al.) with PHQ-8 and ASRM data – that could serve as a testbed to simulate our system.
  * Internal pilot data from our clinic: We may do a silent trial where we run the detector in the background on patients who agree, and see if it predicts changes recorded in their charts (without acting on it initially). This can provide real-world PPV/NPV before fully deploying.
  * Edge cases: fabricate scenarios (synthetic data) to test system logic – e.g. feed 0% data to ensure it falls back to baseline, or extreme values to ensure no crash.

* **Cross-Validation and Hold-out:** We’ll do k-fold CV for model development, and reserve a final hold-out set (or use a rolling forecasting approach for time-series) to evaluate the fully combined pipeline’s performance. This final eval will tell us if our ensemble and calibration work well on unseen patients.

* **Human Factors Evaluation:** In addition to pure accuracy, we plan to evaluate how clinicians respond to the alerts. For example, in a pilot deployment, track:

  * How often did clinicians agree with the alert and take action?
  * Were there false alarms that were considered noise?
  * Did any episodes occur without alert (misses) and why?
    This feedback will guide tuning thresholds for practical use (perhaps raising the threshold for “High” to reduce false positives if clinicians feel moderate alerts are sufficient for some cases).

* **Performance Monitoring:** Once deployed, we will continuously monitor the system’s predictions vs actual outcomes (ground truth from follow-up PHQ-9 or clinical notes about mood switches). This will be part of a **model monitoring plan** to catch any drift (for example, if patient population changes or device data quality shifts).

In summary, the evaluation plan is rigorous, combining quantitative metrics (AUC, PPV, NPV, etc.) with real-world validation. Our goal is to demonstrate that the engine can *“accurately predict mood destabilizations”* as claimed in literature, and that it does so in a clinically actionable way (with a tolerable false alarm rate and clear benefit of early detection).

## Fail-Safe Design and Graceful Degradation

Healthcare data can be incomplete or models can fail – our system includes several fail-safes to handle these scenarios without giving dangerous or no output:

* **Missing Data Handling:** The ensemble weighting already addresses partial data. If the wearable data stream for one pathway is largely missing, that pathway’s influence is minimized (effectively, the other model “carries” the prediction). For instance, if a patient only wears their smartwatch 2 days a week, the depression model’s coverage is <30%; the router will down-weight PAT to 0.25 and rely mainly on the 30-day features from those sparse points plus whatever mania model can glean (which might also be limited). In extreme cases, if **both models lack minimum data**, the system recognizes it cannot make a confident assessment. Rather than producing a possibly wrong guess, it would output an `overall_tier` of “Indeterminate” (or default to Low with a caveat). In the UI or message, we’d indicate “Insufficient data to assess mood risk this week.” This is far safer than a false sense of security or false alarm. Simultaneously, an alert can be sent to technical support or the patient reminding them to wear the device or fix data connectivity.

* **Fallback to Baseline & Population Norms:** As clarified, for a new patient in the **initial 1–2 weeks**, we lack personal baseline and possibly enough data for XGB (requires 30 days). During this bootstrap period, the engine will:

  * Use **population priors** as baseline. E.g., assume average sleep 7h, normal variability, etc. This means initial Z-scores and risk may be less personalized. We expect more “Moderate” flags in early weeks if a patient is far from average, which is acceptable as a cautious start.
  * We can also incorporate any available baseline questionnaires. For example, if the patient had an intake PHQ-9 or clinical assessment, that could set an initial depression probability level (this isn’t in scope of current design, but clinicians could input a baseline risk tier manually that the system respects initially).
  * Over the first 30 days, the mania model’s output is not available until enough data accumulates. As a stop-gap, we might run a simpler heuristic: e.g. “if patient has <4h sleep for 3 nights in week1, flag an alert for possible mania” even before the full model. The `early_warning_signs` defined in the Clinical Dossier (like consecutive short sleep as a trigger) could be used. Essentially, hard-coded rules can cover the gap until the ML model is ready. These rules are derived from clinical lore: e.g. two nights of little sleep might presage mania.

* **Model Failure or API Error:** If the PAT model fails to load or crashes on a given input (rare, but possible if data format unexpected), the system will log the error and continue with the mania model alone. Similarly, if XGB model throws an error, we proceed with PAT alone. The `overall_tier` in such single-model fallback cases will be based solely on that model’s risk (we might treat the missing model’s output as neutral or low in the ensemble). The CommunicationRequest in these cases will include a note like “(Depression track data only; mania assessment unavailable)”. Meanwhile, an automated alert to the engineering team (out-of-band) will be generated to fix the model issue.

* **Graceful DB Downgrade:** If TimescaleDB is down, the system should still function for predictions (since that’s mostly in-memory models). It will queue the results and retry logging later. The CommunicationRequest can still be sent to clinicians. This ensures continuity of care even if the analytics store is temporarily offline – no data will be lost (thanks to either caching in memory or a message queue) and will sync when DB is back.

* **Redundancy & Heartbeats:** The system can be deployed with redundancy (multiple instances of the FastAPI app so that if one goes down, others pick up processing). We’ll also implement a heartbeat that checks data freshness – e.g. if no data ingested for a certain time, it can alert ops. This prevents silent failures where the pipeline stops running but no one notices (fail-silent). Clinically, if no update is generated when expected, that itself should raise a flag to check the system.

* **User Override:** We acknowledge that no automated system is perfect. Thus, clinicians will have the final say. The interface should allow a clinician to dismiss or downgrade an alert if they have other knowledge (and conversely, to manually query the system or input concern if they feel the data isn’t capturing something). While this is more of a UI/workflow design, it’s a fail-safe to ensure the human can correct the AI’s course.

In all fail-safe scenarios, **patient safety is priority**. The system is designed to avoid **false negatives** (missing a real risk) as much as possible, even if it means occasionally outputting “unknown” or erring on the side of a false positive. By communicating uncertainties (like low data availability) clearly, we keep the clinician in the loop rather than blindly trusting incomplete outputs.

## Regulatory and Privacy Considerations

Deploying an always-on health monitoring tool involves careful compliance with health regulations (HIPAA in the US, GDPR in the EU, etc.) and ensuring patient privacy and data security:

* **HIPAA Compliance:** All data handled – raw actigraphy, heart rate, computed mood risk – is considered Protected Health Information (PHI) when tied to patient identifiers. Our system will:

  * Ensure **encryption** of data at rest and in transit. The TimescaleDB will reside on encrypted storage, and connections will use SSL. Any FHIR transmissions to the EHR will go over secure channels (HTTPS with authentication, within a hospital VPN or similar).
  * Limit PHI use: We do not use any patient names or addresses in our processing; the system operates on internally assigned user IDs. The wearable data is pseudonymized (ID only). If any data needs to be logged (for debugging), it will exclude identifiable info. For example, structlog entries will refer to `user_id` not “John Doe”.
  * **Access Control:** Only authorized services and clinicians can access the data. The mood risk outputs in TimescaleDB are accessible via the API to clinicians with the appropriate role. We’ll implement audit logs of access – e.g. if a clinician views the data, that access is logged (either in the EHR or our system) per HIPAA’s accounting of disclosures requirement.
  * **Business Associate Agreements (BAA):** If using cloud services for hosting or storage, ensure they are HIPAA-compliant and a BAA is in place. Given this is an open-source repo, implementers must host it in a compliant environment (e.g. Azure/AWS HIPAA-ready services or hospital servers).

* **GDPR Compliance:** For deployments in jurisdictions under GDPR (or similar regimes):

  * We treat the patient data as sensitive personal data (health data). We will obtain proper **consent** from users for collecting and processing their wearable data for this purpose (unless it falls under legitimate interest for treatment, which in a clinical setting it might).
  * **Data Minimization:** We only store data relevant to the mood detection purpose. Raw sensor data might be processed and discarded, keeping only aggregated features and risk scores. (If we store raw minute-level data at all, it’s short-term and purged after feature extraction). This minimizes risk if a breach were to occur.
  * **Right to Erasure:** We design the system such that a patient’s data can be deleted on request. This means removing their entries from TimescaleDB and any logs, and not retaining their baseline or model outputs. Note: since our models don’t individually retrain per patient (no online learning yet), deletion is straightforward. If in future we do personalized model tuning, we’d have to retrain without that data upon request.
  * **Anonymization for Research:** If we use aggregated data to publish findings or improve the model, we will anonymize it (e.g. no identifiers, and data pooled such that individuals can’t be re-identified easily). We also consider using differential privacy techniques if analyzing large patient datasets to ensure no one’s pattern can be isolated inadvertently.

* **FDA and Clinical Safety:** In the US, software that provides clinical decision support can be subject to FDA regulations. The FDA’s guidance on CDS (per Section 520(o)(1)(E) of FD\&C Act) exempts certain low-risk CDS that *allow the clinician to independently review the basis for the recommendation*. Our system provides transparent explanations and uses published guidelines as basis, which means the clinician *can* understand how it concluded the patient is high risk (e.g. by looking at the contributing factors). This transparency is a deliberate design to keep the tool in the assistive category rather than an opaque “black box diagnosis.” We will label the software as a clinical decision support tool that **does not replace clinician judgment**. Assuming it’s used to flag risk and not to make autonomous treatment decisions, it likely falls under a low-risk device enforcement discretion. Nonetheless, if we market it for clinical use, we will do thorough validation and possibly seek FDA clearance as a Software as Medical Device (SaMD) Class II, especially given it does analyze health signals and issue recommendations (the line can blur if it’s deemed to be driving clinical action). Our documentation will include intended use statements and proper disclaimers (e.g. “This tool is not a diagnostic; it is intended to support and not direct clinical decision-making”).

* **Clinical Governance:** We will have an oversight process – likely an internal clinical safety committee – to periodically review system outputs and any potential adverse events. For example, if the system ever failed to identify a patient who then was hospitalized, we analyze that case and adjust the algorithms or thresholds. This kind of governance is often expected by regulators and quality standards (like ISO 13485 for medical software).

* **Privacy by Design:** We built privacy into the architecture:

  * Data aggregation happens on secure servers; if any processing is done client-side (not currently, but if the wearable pre-computes features on phone), we ensure those features are transmitted securely.
  * Only the minimal necessary data leaves the patient’s device. (Potential future improvement: perform the PAT inference on device and send only the risk score to server – this would drastically reduce raw data sharing. Our current design centralizes processing, but an on-device approach could be explored for privacy.)
  * **De-identification:** If data needs to be shared with researchers or across borders, we will remove direct identifiers and use coded IDs. We also must consider that even derived data like activity patterns could potentially identify someone (for example, unique daily routines). While the risk is low, we follow guidelines to mitigate re-identification (e.g. don’t include geolocation traces, etc., which we don’t anyway).

* **Audit Logging:** Every action the system takes – data ingestion, risk calculation, alert sent, clinician viewed alert – is logged with timestamps. These audit logs are crucial for:

  * **Security audits:** to detect any unauthorized access or anomalies.
  * **Quality audits:** to review cases for continuous improvement.
  * **Legal defense:** if ever a question arises “why wasn’t an alert sent?” we can show the data state and decisions from logs.
    Our logging uses structured format (JSON) via structlog so that it’s easy to query. Sensitive info in logs (if any) can be masked. Access to logs is restricted to admin/dev personnel and they are kept separate from the general application interface.

* **Data Ownership and Use:** We will clarify in user agreements (if patient-facing) or documentation that the data collected is used for their care and, if for research, will follow IRB protocols. Under GDPR, we’ll specify the lawful basis (likely provision of health care / public interest in health).

In summary, our system is designed with a **privacy-first, compliance-first mindset**. By leveraging standard protocols (FHIR) and robust logging, we make integration easier to audit. We emphasize that the tool *augments* clinician decision-making (the clinician can always question or override it), which keeps the ultimate control in human hands – an important aspect for both ethical and regulatory acceptability. We will continuously update our privacy and security measures as new regulations or threats emerge (e.g., ensuring cybersecurity best practices so that malicious actors cannot tamper with the model or data – imagine the harm if someone falsified an “all clear” when the patient was actually in danger).

---

**Open Engineering Questions & Future Considerations:**

* **Generality vs. Personalization:** How well will one global PAT or XGB model perform across diverse patients (different ages, comorbidities)? Should we introduce personalization in the models (e.g. fine-tune PAT per individual on their data, or at least calibrate thresholds per person beyond z-scores)? This could improve accuracy but complicates deployment (tracking multiple model versions). It raises the question of model drift – as a patient’s baseline shifts, the model might need re-calibration.

* **Mania vs. Hypomania Detection:** Our current design treats “mania risk” somewhat monolithically. In bipolar disorder, hypomania (milder form) might be handled differently than full mania. Should the XGB model output be split into two probabilities (one for mania, one for hypomania)? The reference model had separate performance for manic vs hypomanic episodes. We could train a multi-class classifier or two binary classifiers. This might allow the CDS to respond with more nuance (perhaps a moderate alert for hypomania vs urgent alert for full mania). It complicates the ensemble (which output to weight?), so we kept one probability – but this is an area to explore.

* **Feature Expansion:** The current feature set is rich for sleep and circadian, but mood disorders have other digital biomarkers (phone usage, speech patterns, etc.). Could the engine incorporate additional data streams in the future? For example, a **third pathway** for **behavioral data** (text sentiment, voice tone via an app) could predict mood changes. We would need to ensure the ensemble can scale to more models and that we don’t overwhelm clinicians with complexity. But modular design allows adding new domain models relatively easily.

* **Threshold Tuning for Alerts:** What is the optimal threshold for triggering a CommunicationRequest? We have tiers, but in practice, we might decide not to actively message the clinician unless it’s high risk (to avoid too many notifications). Perhaps moderate risk just logs to a dashboard unless it persists or escalates. We need to find that sweet spot through user testing – this is more a UX/policy question but critical for adoption. Alert fatigue is real; too many moderate alerts could cause clinicians to ignore the tool.

* **Handling Medications and Clinical Events:** We used a simple rule for SSRIs. In reality, many factors (mood stabilizers, recent medication changes, therapy compliance, life events) affect mood. Should we integrate EHR data more deeply? For example, if the patient recently stopped lithium, the threshold for mania alert might be lower. Or if a patient had a psychotherapy session logged with severe symptoms, maybe the system should weight that. Integrating such information could improve accuracy but requires complex EHR data handling and introduces potential noise. An open question is how far to go in that direction or keep the system focused on objective wearable data.

* **Real-Life Translation and Evaluation:** How do we measure success in deployment? Is it reduction in hospitalizations, improved symptom scores, or just clinician satisfaction? We should plan a prospective study where half of clinicians use the tool and half don’t (randomized) to see if outcomes differ. This is more a research question, but engineering-wise we may need to implement features to support such a study (like toggling the alerts on/off for control group, etc.). The question of *“should we even implement this system”* ultimately hinges on whether it leads to better patient outcomes – something that can only be answered by real-world testing.

* **Edge Cases & Bias:** Our models are data-driven; we must be vigilant about biases. For example, will the PAT model underperform in an elderly population if it was trained mostly on adults <65? Or does the XGB model assume a 24h sleep cycle that might not apply to shift workers (circadian metrics might misinterpret shift work as “phase delay” pathological)? We might need to add logic to recognize shift workers or irregular schedule jobs (perhaps via context from EHR occupation info or by detecting 24h rhythm changes) and adjust expectations. Similarly, cultural or gender differences in activity (e.g. women might have different HRV baselines) should be considered. Our evaluation should include subgroup analysis to uncover any performance gaps. Mitigating bias might involve re-training on more diverse data or setting different alert thresholds for certain groups – an ongoing engineering challenge.

* **System Scalability:** If this is deployed to thousands of patients continuously streaming data, can our architecture handle it? We might need to ensure the pipeline (especially PAT inference) is optimized – perhaps using batch processing overnight for the bulk of computations, or leveraging streaming frameworks. TimescaleDB should handle time-series scale, but the compute scaling needs planning (Kubernetes deployment with autoscaling perhaps). This is an engineering question for the future if scale grows.

* **Integration with LLMs for Explanations:** Taking inspiration from the CDS-Bipolar-LLM paper, could we use an LLM to generate more nuanced interpretations of the risk and even suggest treatment changes? For instance, feed the model outputs and guideline text into an LLM to produce a summary: “Patient’s data suggests a high risk of mania. According to guidelines, ensure patient is taking mood stabilizer; consider temporarily pausing antidepressants.” This could be a powerful extension, but we’d need to be cautious of inaccuracies and still require clinician verification. For now, we stick to rule-based messages, but this remains an open avenue.

Each of these questions will be addressed iteratively – the current spec establishes a solid, science-backed foundation, and ongoing evaluation and feedback will inform future refinements