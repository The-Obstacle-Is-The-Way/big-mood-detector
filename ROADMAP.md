# Bipolar Mood Detector Backend Architecture & Development Plan

## Comprehensive Codebase Analysis

**Repository Composition:** The Big Mood Detector repository is a consolidation of research-grade components and reference implementations prepared for real-world use. Key directories under `reference_repos/` include:

* **`mood_ml/` – XGBoost Mood Models:** Contains pre-trained XGBoost models for predicting **depressive episodes (DE)**, **manic episodes (ME)**, and **hypomanic episodes (HME)**. These models were validated on clinical data (168 patients, \~44k patient-days) with high AUCs: \~0.80 for depression, 0.95 for hypomania, and 0.98 for mania. The pipeline uses **36 sleep and circadian features** per day as input, computed by a MATLAB script (circadian rhythm modeling). The models are considered production-ready given their strong clinical validation and self-contained `.pkl` files for direct loading. *Note:* The dependency on MATLAB for feature extraction is a current hurdle for deployment, addressed later.

* **`Pretrained-Actigraphy-Transformer/` (PAT):** A **foundation transformer model** for wearable actigraphy data, pre-trained on 29k participants (NHANES dataset). It captures minute-level movement patterns via a BERT-like transformer, enabling fine-tuning for custom tasks. In our context, PAT can analyze raw accelerometer/time-series activity to enhance mood predictions (e.g. capturing patterns beyond sleep). This component is more research-prototype; it provides `.h5` model weights and example notebooks for fine-tuning and explainability. We plan to leverage PAT in later phases for advanced analysis, but it’s not the first deployment target due to added complexity.

* **`apple-health-bot/` – Apple HealthKit Parser:** A utility to **convert Apple Health exports (XML)** into structured CSV data. It includes an `xmldataparser.py` script that extracts health metrics (sleep, activity, heart rate, etc.) from Apple’s export format for downstream use. This is crucial for ingesting the primary data source (Apple HealthKit JSON/XML) into the pipeline. The parser is production-grade, complete with Docker support, since it’s part of a project that uses LangChain/LLM to analyze health data via SQL. We will reuse the parsing logic to get our input data (e.g., daily sleep records, step counts) in a workable form.

* **`tsfresh/` – Feature Engineering Library:** An integrated copy of **Tsfresh**, a widely-used automated time-series feature extraction framework. Tsfresh can compute 100+ statistical features (peaks, spectral entropy, autocorrelation, etc.) from time-series data with minimal code. In this project, tsfresh is intended to transform raw activity/physiological time-series from wearables into features for the ML models. Tsfresh is well-tested and considered production-ready for feature engineering tasks. Its inclusion means we can generate rich features from Apple Health data (e.g. daily step count series, heart rate series) without manual coding of each feature.

* **`yasa/` – Sleep Staging Toolkit:** YASA is a library for automatic sleep stage classification from polysomnography signals (EEG/EOG/EMG). It’s included to analyze sleep architecture if high-resolution sleep data is available. However, Apple Watch/HealthKit data doesn’t include full PSG signals – at most, heart rate or maybe a proprietary sleep stage estimate. Thus, YASA is likely a supplementary module for advanced use (e.g., integrating clinical sleep lab data or using surrogates). We will not focus on YASA in early development, as it’s not critical for *HealthKit-based* mood prediction MVP.

* **Integration & API Helpers:** The repo also provides examples for **Gradio** (a UI framework for quick web apps), **tRPC** (type-safe API in Node/TS), and a **FHIR client** for healthcare data standards. These aren’t part of the core analytics pipeline but are tools for building interfaces and integrations:

  * *Gradio:* Useful for prototyping a web-based demo where clinicians or testers can upload data and see predictions instantly. We’ll use this in early testing phases (Phase 1) as a “clinical sandbox” UI.
  * *tRPC examples:* Illustrate how one might build a backend API with end-to-end type safety (likely for a Node.js environment). In our plan, the core ML backend will be Python-based, but we can still expose REST or RPC endpoints. We might not use tRPC directly in Python – instead, FastAPI is a better fit for a Python service. However, understanding tRPC patterns will help if we later integrate this service with a TypeScript/Next.js front-end (Phase 3).
  * *FHIR client:* Indicates future integration with Electronic Health Records. Not needed for MVP, but in advanced phases we should ensure our data structures (patients, observations, predictions) can map to FHIR resources for clinic/hospital use.

**Literature Alignment:** The `literature/converted_markdown/` folder contains markdown summaries of each foundational study (Seoul Nat’l, Harvard, Barcelona, Dartmouth, Berkeley, etc.). These give insight into **why** each component exists:

* The **Seoul National study (Nature Dig. Med. 2024)** underpins `mood_ml/` – it introduced the 36 circadian features and achieved the 0.80–0.98 AUC results using sleep/wake data.
* The **Dartmouth study** (PAT) justifies the transformer approach for raw actigraphy.
* Harvard’s **Fitbit study** corresponds to a BiMM model (not explicitly implemented here, but our approach covers similar ground via XGBoost and tsfresh).
* **tsfresh** is referenced as a general feature engineering approach (Blue Yonder research) included to broaden the feature set beyond the 36 handcrafted ones.

In summary, the codebase provides **all the building blocks to replicate the research pipeline**: from data ingestion (Apple HealthKit parser), through feature extraction (circadian indices + tsfresh), to predictive modeling (XGBoost, with PAT as an enhancement), plus tools for deployment and integration (Gradio UI, API patterns, FHIR). Our task is to **assemble these into a coherent, shippable backend system** focused on *Apple HealthKit JSON* input and delivering mood episode risk predictions.

## Data Flow from HealthKit Data to Mood Predictions

The end-to-end pipeline will transform a user’s Apple HealthKit data into a bipolar mood episode risk prediction. The high-level flow is:

1. **Apple Health Data Input:** The user (patient) exports or streams their Apple HealthKit data (either as a full `export.xml` file or through an API/Share mechanism in JSON). This data includes daily records like sleep times, step counts, heart rate, etc.. For our purposes, the critical fields are sleep start/end times, sleep duration/quality, and activity metrics, as these feed the models.

2. **Parsing & Preparation:** The raw HealthKit data is parsed using the Apple Health Bot parser. This step converts the XML/JSON into structured CSV or DataFrame format. For example, all sleep records might become a table with columns: date, sleep\_start, sleep\_end, time\_in\_bed, minutes\_asleep, minutes\_awake (similar to the `example.csv` format expected by `mood_ml`). Likewise, workouts or step counts might become a time-series table (timestamp and value). After this step, we have clean data frames ready for feature extraction.

3. **Feature Extraction:** Two parallel feature pipelines occur:

   * **Circadian & Sleep Indices (36 features):** Based on the daily sleep schedule data, we compute features capturing circadian rhythm and sleep patterns. In the reference implementation, this is done by running `Index_calculation.m` in MATLAB, which uses functions (`mnsd.p`) to produce features like sleep regularity, chronotype phase estimates (DLMO – Dim Light Melatonin Onset proxy), sleep debt, etc. This yields a `test.csv` with one row per day containing 36 features. *Workaround:* If MATLAB is unavailable in production, these features need to be reimplemented in Python. Initially, for MVP, we might use simpler proxies (e.g. total sleep, sleep/wake times variability) or run MATLAB offline as a one-time step. Addressing this dependency is critical for full automation.
   * **Time-Series Features via tsfresh (100+ features):** We feed any high-frequency data (e.g., step counts per hour, heart rate time-series) into tsfresh to automatically generate a rich feature set. Tsfresh will output numerous features (min, max, Fourier coefficients, autocorrelation, etc.) for each sensor stream. These can complement the 36 handcrafted features with additional signals (e.g., daytime activity levels, variability in movement). In the minimal pipeline, tsfresh might not be mandatory if we rely solely on the 36 features, but incorporating it can improve accuracy and generality by leveraging *all* available HealthKit data. Tsfresh integration is straightforward via its `extract_features` API in Python.

4. **Mood Episode Prediction (ML Models):** The engineered features are then fed into the pre-trained ML models:

   * **XGBoost models (primary):** We load the appropriate XGBoost model(s) – e.g., for **mania prediction** (XGBoost\_ME.pkl) – using Python’s pickle or XGBoost library, and call `.predict()` on the feature vector. The model outputs a probability of a mood episode (e.g., probability of mania vs no-mania for the upcoming day/week). We have three separate models for depression, mania, hypomania; each would take the same feature inputs but produce different risk scores. In an initial deployment we might use one (say mania, given its high accuracy) for simplicity, then later run all three to cover the spectrum.
   * **PAT Transformer (secondary, optional):** If raw accelerometer data or rich activity data is available, we can also feed it into the PAT model. The PAT, after fine-tuning or even in a zero-shot mode, could output additional insights – for instance, anomaly scores in activity that correlate with mood changes, or predictions of medication use or sleep disorder that indirectly inform mood. In practice, PAT would be an advanced add-on that runs in parallel with XGBoost. For MVP, it’s likely omitted or run in analysis mode only. In Phase 2, we plan to incorporate PAT to enhance predictions (e.g., combine PAT’s output with XGBoost via an ensemble or as additional features).
   * *(Ensemble/Combination):* If both XGBoost and PAT are used, their outputs might be combined (e.g., average of risk scores, or PAT’s features used in an extended XGBoost). But initially, each can be reported separately or we focus on XGBoost outputs.

5. **Results & Output:** Finally, the system produces **predictions and insights** for the end-user or clinician. This could include:

   * Probability of an upcoming depressive, manic, or hypomanic episode (often as daily or weekly risk scores).
   * Which features are most indicative (e.g., a large circadian phase shift might be flagged as a risk factor for mania).
   * Any other relevant analytics (for example, “sleep duration decreased 2 hours (Z-score -2.5) compared to baseline, which is associated with elevated mania risk” – turning raw predictions into explainable feedback).
   * The output format might be a JSON response if an API (with fields for each risk score), or a visual display if using a UI. We will ensure the output is **privacy-conscious** (no raw data echoed back) and clinically interpretable.

Below is a **data flow diagram** summarizing these steps from input to output:

```mermaid
flowchart LR
    A[Apple HealthKit Data\n(JSON/XML export)] -->|parse| B{{Apple Health Bot\n(XML to CSV Parser)}}
    B --> C[Sleep & Circadian Features\n(36 indices per day)]
    B --> D[Time-Series Features\n(tsfresh 100+ features)]
    C --> E[XGBoost Model\nMood Episode Prediction]
    D --> E
    D --> F[PAT Transformer\nAdvanced Analysis]
    E --> G[Predicted Mood Risk\n(episode probabilities)]
    F --> G
```

*Figure: Pipeline from raw Apple HealthKit data to mood episode predictions. Solid arrows denote Phase 1 components; dashed arrow (PAT) is an advanced Phase 2 addition.*

The Apple Health Bot (B) produces structured data used in two ways: (C) daily summary features for the circadian model, and (D) granular time-series features via tsfresh. The XGBoost model (E) is the primary predictor using features from both sources. The PAT model (F) can optionally analyze raw data for additional context. Finally, results (G) are output as risk scores and alerts.

## Technical Integration Assessment

Let’s address specific integration questions in the context of this architecture:

* **Using tsfresh with Apple Health Data:** *How can tsfresh extract features from Apple HealthKit data?* Apple HealthKit provides time-series data such as step counts (per minute/hour) and heart rate (e.g., periodic measurements). We will format these into a pandas DataFrame with a time index or an “id” for each day/subject, then call `tsfresh.extract_features` to automatically compute a broad range of features. For example, given a series of hourly step counts for a day, tsfresh can compute features like mean, variance, number of peaks, circadian rhythm strength, etc.. We can assign each day an identifier, so tsfresh yields feature rows per day. This seamlessly augments the hand-crafted sleep features. Tsfresh’s strength is its *automation and breadth* – we don’t have to manually decide all possible features, it will generate and even select relevant ones. In practice, we might limit or filter features based on relevance (to avoid overfitting), but given the moderate size of daily data, it’s manageable. Summing up, tsfresh will serve as an **automated feature engineer**, turning HealthKit’s raw activity logs into model-ready inputs in one function call.

* **Reliability of XGBoost Models (DE, ME, HME):** *Which XGBoost models are most reliable for initial deployment?* All three mood models are pre-trained and validated, but their performance differs slightly. The **mania (ME) model** achieved \~98% AUC, the **hypomania (HME)** \~95%, and the **depression (DE)** \~80%. The higher the AUC, the more robust the model was in research settings. For an initial deployment, focusing on **mania prediction** yields the biggest clinical impact with high confidence (nearly 98% AUC for distinguishing manic episodes). Hypomania is also highly accurate. Depression is a bit lower (still respectable, but mood changes for depression may be subtler). Thus, an MVP could start with the **Mania model** as the flagship (where we can confidently alert clinicians to impending manic episodes), then include the others as needed. That said, the three models operate similarly (just predicting different targets), and the cost to run all three is trivial once the features are computed. We anticipate eventually deploying all of them to give a complete picture of a patient’s mood risks. The **key point** is we will leverage these pre-trained models **as-is** – no retraining, just loading the `.pkl` and running predictions, which is fast and easy. This aligns with the directive to use existing models rather than developing new ones.

* **Role of PAT Transformers:** *How can the Pretrained Actigraphy Transformer enhance the baseline XGBoost approach?* The PAT model is a **foundation model for raw actigraphy data**, capable of capturing complex temporal patterns over days to weeks. Where XGBoost relies on summary features (like “sleep 7h, phase shift 2h”), PAT can directly ingest sequences of minute-level movement or heart rate data. This could enhance detection of mood episodes by recognizing patterns that the 36 features might miss – for example, subtle changes in daytime activity variance or fragmented sleep within the night that aren’t fully captured by aggregate features. In practice, PAT could be used in two ways:

  1. **Feature Enhancer:** Use PAT’s transformer encoding as an additional feature generator. For instance, pass a week of raw accelerometer data through PAT (without necessarily classifying), and use the latent representation or attention weights as features indicating “unusual activity pattern” which could improve the XGBoost model.
  2. **Direct Predictor:** Fine-tune PAT on mood episode classification using the available data (the repo provides fine-tuning notebooks). A fine-tuned PAT could directly predict mood episodes from raw data, potentially rivaling or exceeding XGBoost. However, fine-tuning would require a labeled dataset and training infrastructure.

  Initially, PAT integration will be **experimental/optional**. In Phase 2, we plan to incorporate PAT in parallel with XGBoost to see if it adds predictive power. PAT’s advantage is being *state-of-the-art* for wearables and might capture circadian dynamics implicitly. Its disadvantages are complexity and computational load (especially if we aim for real-time). We will treat the XGBoost pipeline as the primary engine (since it’s lightweight and already high-accuracy), and use PAT as a **secondary module for advanced analysis** or for future versions when more data is available.

* **Minimal Viable Data Pipeline:** *What is the minimal pipeline from JSON input to processed output?* The simplest end-to-end slice that delivers value is:

  * **Input:** Apple HealthKit sleep data (either as JSON from an API or as part of the export). Specifically, the **sleep schedule** – when the user slept and woke up each day, and total sleep duration.
  * **Processing:** Calculate at least basic features from this sleep data. In the absolute minimal scenario, we might skip the full MATLAB circadian index step and instead compute a few crucial features in Python: e.g., total sleep time, sleep onset variability (standard deviation of bedtime), and perhaps a crude circadian phase proxy (difference between actual sleep midpoint and user’s average sleep midpoint). These can be scaled to the user’s baseline (i.e., how unusual is today’s sleep compared to their normal – a concept critical in circadian mood triggers).
  * **Prediction:** Feed these features into one pre-trained model (say, the mania XGBoost model). Even with a handful of features, the model should produce a meaningful risk score. We expect that if, for example, the user’s circadian phase is shifting earlier rapidly (perhaps indicating a manic trend), the model’s output probability for mania will increase.
  * **Output:** A mania risk score for that day (or week). This could simply be a number between 0 and 1 (with a threshold, e.g., >0.5 triggers a “high risk” alert). The output could be delivered as JSON: `{"date": "2025-07-15", "mania_risk": 0.76}`, for instance.

  This minimal pipeline uses only **Apple HealthKit JSON → \[basic feature calc] → XGBoost model → prediction**. It avoids external dependencies (Matlab) and can run quickly on-device or on a small server. It provides immediate clinical value by giving a data-driven risk estimate of a mood episode using just passive sleep data – which meets our goal of **“shippable software”** over pure research. From this baseline, we can incrementally add complexity (e.g., more features via tsfresh, more models, etc.) to increase accuracy and scope.

## Shippable Backend Iterations (3-Phase Development Plan)

To deliver working software iteratively, we propose a **three-phase development plan**. Each phase produces a functional backend slice that builds on the previous, adding features and improving integration. The focus is on incremental delivery: after Phase 1, we already have a usable (if basic) system, which we then enhance.

### **Phase 1 – MVP: Local Sleep-Based Prediction**

**Goal:** Deliver a basic pipeline that can take Apple HealthKit sleep data and return a mood episode prediction (focusing on one model). This will prove the concept end-to-end and serve as a foundation for enhancements.

* **Data Input & Parsing:** Implement the pipeline to accept an Apple Health **export file** (XML/JSON). For development, we can manually export Health data from a test user. Use the `apple-health-bot` parser to convert this export into CSV files or DataFrames. In particular, obtain a **sleep data table** formatted like `example.csv` (with columns: date, sleep\_start, sleep\_end, time\_in\_bed, minutes\_sleep, minutes\_awake).

* **Feature Computation:** For MVP, choose a strategy:

  * *Option A:* If MATLAB is accessible in the environment, run `Index_calculation.m` on the sleep CSV to get the full 36 features per day. This gives maximum fidelity to the research.
  * *Option B:* If we cannot rely on MATLAB (likely in a deployed setting), implement a **simplified Python feature extractor**. Compute a handful of key features: e.g., total sleep minutes, sleep efficiency (% of time in bed asleep), bedtime variation, wake-time variation, and a rough circadian phase deviation (difference from user’s average sleep midpoint). These can be derived from the sleep\_start/end times.
  * The trade-off: Option A yields proven features but adds a MATLAB dependency; Option B keeps it Python-only at the cost of missing some nuance. For MVP (especially local prototype), we might use MATLAB offline to validate the pipeline, but concurrently start coding Python equivalents for those features (this can be a priority refactoring task after MVP demonstration).

* **Model Integration:** Load one XGBoost model, likely **XGBoost\_ME.pkl** (mania episode model), using Python’s pickle. Verify the model loads correctly and expects the feature columns we have. If we used `Index_calculation.m`, the model will expect exactly those 36 features named as in `test.csv`. If we used a custom subset, we may have to adjust (the model might still work if missing features are set to 0 or mean — but ideally we provide what it was trained on). We can initially test using the provided `example.csv` -> `test.csv` -> model to ensure the pipeline reproduces the expected outputs (the repo includes `expected_outcome_me.csv` as a reference output for the example data). This ensures our integration is correct.

* **Output & Verification:** The MVP should output a **prediction** for each day in the input data, indicating the probability of a manic episode. We’ll validate on the example data to see that the model output probabilities match the provided expected outcomes (or at least are in the right ballpark). Even if minor differences, we ensure qualitatively it makes sense (e.g., days with severely reduced sleep show higher risk, etc.). For clinician interpretability, we might also output which features contributed most (though full SHAP analysis is complex; we can at least log the top features from the original paper – circadian phase Z-score, etc. – that the clinicians can consider).

* **Interface:** During Phase 1, the interface can be simple. Two approaches:

  * A command-line or notebook interface where a developer/clinician runs a script/notebook to generate predictions from a given export file.
  * Additionally, set up a **Gradio demo** for internal use. This would allow a user to upload an `export.xml` (or the parsed CSV) through a simple web page and see the predicted risk. Gradio is quick to configure and doesn’t require full backend deployment – it can run locally. This addresses the need for a “testing framework for clinical validation” as noted in the plan. We will include basic instructions (e.g., “Click here to upload your Apple Health export, and get a mood risk report”).

* **Delivery:** By end of Phase 1, we have a **local prototype**: for example, a Jupyter Notebook or Python script that reads data and prints predictions, plus possibly a Gradio UI for demonstration. This is not yet a multi-user scalable system, but it proves that our core components work together (HealthKit parsing → features → model → prediction). It’s also entirely local (no cloud needed), preserving privacy (the data never leaves the user’s machine in this setup).

### **Phase 2 – Enhanced Backend: Multi-Model & API Deployment**

**Goal:** Build on the MVP by adding functionality and moving toward a production-ready service. Phase 2 will introduce additional models, more automation, and a proper API so front-end apps can consume the predictions.

* **Multiple Models & Features:** Expand the pipeline to utilize **all three XGBoost models** (DE, ME, HME) so we cover depressive, manic, and hypomanic episode predictions. The input features for all three are the same 36-dimensional set (plus any tsfresh extras we append). The system will produce three risk scores instead of one, giving a fuller clinical picture. We also integrate **tsfresh** in this phase if not done in Phase 1. For example, incorporate daily step count patterns: the Apple Health parser will give us step counts per day or active energy; using tsfresh on intraday activity could add predictors for depression (perhaps low daytime activity correlates with depression). We ensure our feature assembly code now merges the circadian features with selected tsfresh features into one feature vector for each day.

* **API Development:** Develop a **backend service** (likely with **FastAPI** in Python, or an alternative framework) to expose endpoints for prediction. Key API design considerations:

  * Define an endpoint like `POST /predict` that accepts input data. Input could be the raw Apple Health export file (if small) or, more realistically, a structured JSON payload containing necessary data fields. For example, the app could pre-extract relevant fields and send: `{"sleep": [...], "steps": [...], "heartrate": [...]}` where each is an array or time-series of recent data. We’ll define the expected format clearly (e.g., last 60 days of sleep records and activity).
  * The API will call our pipeline: parse/format input → compute features → run models → return predictions. Response might look like: `{"predictions": {"depression": 0.12, "mania": 0.8, "hypomania": 0.65}, "timestamp": "...", "features_used": [...]}`.
  * Keep the API stateless for now: it processes the input and returns output without storing anything (we will add state/DB in Phase 3).
  * Implement basic error handling, input validation (e.g., if data is missing or too short), and logging of requests (for debugging).

  We lean towards **FastAPI** because it’s Pythonic and easy to document (with automatic docs via Swagger/OpenAPI). This will allow easy integration with a frontend (mobile or web) via standard HTTP calls. The **tRPC** approach (as mentioned in the repo) could be considered if the front-end is built in TypeScript – one could write a thin Node service that calls our Python service or use something like **gRPC** between front-end and back-end. However, to keep Phase 2 manageable, a simple REST API with JSON over HTTPS is likely sufficient. We’ll ensure CORS and auth can be configured for the eventual client app.

* **Cloud/Container Deployment:** While developing Phase 2, we containerize the application. Write a **Dockerfile** that sets up the environment (Python, required packages, model files). The Apple Health Bot already has a Dockerfile; we might combine steps or use it as reference. The container will allow deployment to cloud providers or on-prem servers easily. We might deploy an instance on a cloud VM or Heroku-type service in this phase for testing. This moves us from a “local-only” solution to one that can be accessed by remote clients (with appropriate network security). We will also integrate **Gradio as a testing UI on the deployed service** (maybe run it in a separate mode) so that even when running headless, we have a way to quickly test via a browser.

* **Enhancements Introduced:** Two advanced integrations are planned in Phase 2 (time permitting):

  * **PAT Integration:** Begin incorporating PAT in the pipeline for those data types where it applies. Concretely, if the user’s data includes accelerometer readings or if the Apple Watch provides granular activity data (e.g., Apple’s “Mobility” metrics or raw accelerometer from ResearchKit), we can run a forward pass of the PAT model. Since PAT is large, we might do this as a background or optional job. The result could be a secondary risk score or some features. This is marked as a stretch goal for Phase 2, and we will evaluate if it measurably improves predictions before making it default. (In any case, we’ll ensure the system is designed to accommodate plugging in PAT easily later.)
  * **Additional Analytics:** Possibly use **YASA** if any form of sleep stage or resting heart rate variability data is available from Apple Health. For instance, some users track HRV or use devices that estimate sleep stages. If present, YASA could analyze those signals for added insight (e.g., detecting REM sleep percentage changes). This again would be supplementary – not required for core function, but could be presented in reports for richer context.

* **Privacy & Security (Phase 2):** Now that data might be sent to a server, we must enforce security. We will implement **HTTPS** for any network communication. We also consider at least a rudimentary authentication for the API if multi-user (even an API key or token) to ensure only authorized clients (e.g., our mobile app) can hit the endpoint. Data at rest (if any) should be minimal in this phase, but any temporary files or logs will be handled carefully (no sensitive info in logs). We might decide **not** to store user data on the server at all in Phase 2, returning results immediately and discarding input, to minimize privacy concerns.

By the end of Phase 2, we expect to have a **functional web-service** that a front-end (like an iOS app or React web app) can hit to get mood predictions. It will cover multiple mood types and use more of the data (improving accuracy). This phase transitions us from a prototype into a **deliverable backend application**.

### **Phase 3 – Advanced: Real-Time & Clinical-Grade System**

**Goal:** Evolve the backend into a scalable, real-time system with all necessary features for clinical deployment, including data persistence, continuous monitoring, and integration with health systems.

* **Real-Time Data Processing:** Rather than relying solely on manual exports, Phase 3 will handle continuous data feed. For Apple HealthKit, this could mean the iOS app sends daily updates (or even streams, via HealthKit background delivery). We design the backend to accept incremental data and update predictions on the fly. Possible implementation:

  * The mobile app could call an endpoint (or use webhooks) every day with the last 24h of data. The backend appends this to the user’s record and re-computes the features and predictions for the new day.
  * We ensure the pipeline is efficient to possibly handle multiple users concurrently. XGBoost predictions are fast (milliseconds per user), and feature extraction for one day is also quick. If PAT is used on long sequences, that might be the slowest part, so we could schedule PAT analysis less frequently or on a separate worker.
  * Consider using **WebSocket or Server-Sent Events** for pushing alerts to clients in real-time if a risk threshold is crossed, enabling immediate notification (this could be done via the front-end polling in Phase 2, but Phase 3 can refine it to a push model).

* **Database Integration:** Introduce a **database** to store user data and predictions. This is important for:

  * **Baseline Computation:** The circadian features rely on a baseline (mean, SD of a person’s metrics). Maintaining a history of each user’s data allows computing personalized baselines and trends over time. A DB (SQL or NoSQL) will allow queries like “average sleep duration in last 30 days” for baseline, or “how has feature X changed in past week”.
  * **User Management:** If multiple users (patients) are using the system, we need to segregate their data. A database with user IDs and data entries ensures each user’s info is isolated and persists across sessions. We would implement strict access control (each user or their clinician can only retrieve their data).
  * **Audit and Analysis:** Storing predictions and inputs allows retrospective analysis, debugging, and model improvement. Clinicians may want to review past trends (e.g., the system predicted high risk on certain days – what was the actual outcome?).

  We’ll likely use a **PostgreSQL database** (potentially via a service like Supabase, as hinted by the repo). Supabase could simplify setting up real-time subscriptions (e.g., the front-end can listen to changes in the DB for their user’s risk score). We’ll design a schema containing tables for Users, DailyFeatures, Predictions, etc. All sensitive health data stored will be encrypted at rest and we’ll follow HIPAA guidelines for protection.

* **Scalability & Deployment:** At this stage, consider deploying on a robust cloud infrastructure. Containerized app can be deployed on AWS/GCP/Azure or a HIPAA-compliant cloud (if dealing with US health data, consider a service with signed BAA). We might use Kubernetes or a simpler container service to allow scaling to more users. Also, implement monitoring (logs, health checks) for the service as it becomes production-critical.

* **Advanced Feature Enhancements:**

  * **Personalization:** Incorporate logic for individual baseline adjustment and thresholding. For example, after collecting 30 days of data for a user, the system can automatically calibrate what “normal” looks like for them (e.g., average sleep 7h, std 30min). Then a drop to 5h might be a >2σ event, triggering an alert even if the absolute model probability isn’t high. This personalization was emphasized in research (Z-score normalization per patient). Implementing this could be as simple as computing Z-scores for key features before feeding to models, or dynamically adjusting output probabilities with user-specific thresholds.
  * **Multi-Modal Data Fusion:** Extend the pipeline to utilize other streams: e.g., **heart rate data** for detecting changes in resting HR (could signal depression or anxiety), **activity levels** for detecting inactivity (depression) or hyperactivity (mania), **schedule data** (if user logs calendar or phone usage patterns, etc.). The models can be expanded or new models introduced (like the Harvard study’s random forest on Fitbit data which included steps and HR). We might integrate an existing model (the repo includes a **ngboost** library which could be explored for probabilistic predictions, though not priority).
  * **Clinical Integration (FHIR):** Use the `fhir-client` reference to map outputs to FHIR resources (e.g., Device data, Observation for mood risk). This would enable sending results to electronic health records or allowing clinicians to pull the data in their systems. We could create a FHIR **Observation** for “Mood Episode Risk Score” with codes for mania risk, etc., and host a simple FHIR API or push to an EHR through an integration engine. This step ensures our application can fit into clinical workflows and not just exist in isolation.
  * **UI/Frontend:** By Phase 3, a more polished frontend should accompany the backend. While not the focus here, we assume an **iOS app or React web dashboard** will be built. We will support that by providing clear API documentation, possibly client libraries (if tRPC, a generated client, if REST, an OpenAPI spec). The front-end would handle user authentication, data syncing from HealthKit (for iOS, HealthKit frameworks can provide continuous data), and display of the predictions (e.g., a graph of risk over time, alert banners on high risk days, etc.). Additionally, a *clinician dashboard* might be built (perhaps as a web app) to allow doctors to monitor multiple patients. Our backend must therefore handle multi-tenancy and secure data separation accordingly.

* **Testing & Validation:** In Phase 3, we conduct more rigorous **clinical validation tests**. Using existing datasets (if available) or starting a pilot where real patient data is fed through the system and outcomes are compared to what happened clinically. We’ll unit test each component but also do end-to-end integration tests: simulate a new user onboarding, feeding data daily, and ensure the predictions update correctly. Given the critical nature, we may run a small-scale trial in parallel with clinical care to see if the alerts correspond to actual mood changes. This feedback would loop into adjusting the system (maybe recalibrating model thresholds or adding features).

By the end of Phase 3, the backend will be **production-ready for real clinical use**: able to handle live data, at scale, with the necessary compliance and integration features. We will have a complete solution from a phone’s HealthKit to a clinician’s screen, providing proactive mood episode warnings.

## Architecture & Infrastructure Recommendations

**Local vs Cloud Processing:** We recommend a hybrid approach:

* **Start Local (Edge)** for initial versions – Phase 1 demonstrated that everything can run on a single machine. This is great for privacy (data stays with the user or clinician) and avoids cloud costs. In fact, one long-term vision could be to deploy the core model on-device (iPhone) so that even predictions happen on the user’s phone, sending out only alerts. Apple’s CoreML could theoretically host an XGBoost model converted to a lightweight model, though the feature engineering might be a challenge on-device. However, for practical reasons and maintainability, a **cloud-hosted backend** is easier to update and monitor. So,
* **Cloud Backend** from Phase 2 onward – a centralized service that receives data from the app. This allows us to use heavier dependencies (like Python libraries, possibly MATLAB if we had to, or easier integration of PAT with GPU if needed) without burdening the mobile app. We will enforce strong security: all traffic encrypted, minimal data retention, and possibly anonymization (the backend might only see a user ID, not personal identifiers). If regulations or user preferences demand, we could also offer an **on-premise or offline mode** (for instance, a clinic could run the container locally to process data in-house with no external network communication).

**API Framework – Gradio vs tRPC vs FastAPI:** For **rapid prototyping** we leverage **Gradio** to get a UI without building a full front-end. This is mainly for demonstration and internal validation. For the actual product interface, we expose a **RESTful API using FastAPI**. FastAPI is chosen due to:

* Native Python usage (fits our codebase where all ML is in Python).
* Excellent performance and documentation generation.
* Flexibility to later add WebSocket endpoints or OAuth2 security easily.

Using FastAPI doesn’t preclude leveraging tRPC concepts. If our front-end team uses tRPC, we could stand up a lightweight Node service that acts as a tRPC client to our FastAPI, or simply have the front-end call our OpenAPI endpoints. Given our stack, a direct REST API is simplest, but we remain open to tRPC for specific integration (especially if we adopt Supabase – Supabase’s client can directly subscribe to data changes, reducing the need for a custom RPC for realtime updates).

**Storage & State:** We plan to use **PostgreSQL (via Supabase)** as indicated for real-time sync and ease of integration. Supabase gives us out-of-the-box authentication and row-level security which can be configured for multi-user privacy. It also provides realtime subscription (e.g., the app can listen for new prediction entries for that user). This significantly simplifies implementing features like live updates. Alternative could be a simple SQLite or local file in Phase 1 (for dev testing), but by Phase 3, a robust cloud DB is needed. We should also consider **caching** mechanisms (like Redis) if we get high throughput or need to store session data (though likely not needed for our workload yet).

**Security & Privacy Considerations:**

* **HIPAA Compliance:** If we handle identifiable health data, host on HIPAA-compliant services. Use encrypted databases, secure backups, and logging of access.
* **PII Minimization:** The system doesn’t require names or addresses – only data like sleep times and maybe age/gender for context. We can identify users by anonymized IDs. Any linking to personal identity can reside on the client side; the backend can operate with tokens.
* **Data Ownership:** Possibly allow users to opt to keep data local (maybe in a future version, the app computes features and only sends features to server, not raw data). Or allow easy deletion of data from the server (compliance with privacy laws).
* **Audit Logging:** Record when predictions are made and perhaps what data influenced them (for liability, one might need to show why a certain alert was triggered – e.g., “on July 15, data showed a 3-hour phase advance in sleep time, model flagged high mania risk”).
* **Fail-safes:** Because this is clinical, ensure that the absence of data or system downtime fails gracefully (no false assurances of “all is well” when data simply didn’t arrive). Possibly notify if data is missing for X days (could indicate the user isn’t wearing device or an integration issue).

In summary, the architecture will transition from a **local, user-run prototype** to a **cloud-based microservice** with appropriate APIs and database support. We emphasize starting simple (to get something working and test with real data) and gradually layering on the robust infrastructure pieces.

## Implementation Roadmap & Next Steps

With the plan in place, here are actionable next steps to start building the system, focusing on near-term tasks and referencing specific components to use:

**1. Environment Setup:** *Ensure all necessary packages and files are available.* Install Python dependencies (`pip install -r requirements.txt` from the repository root should handle most). This includes pandas, tsfresh, xgboost, etc. If using MATLAB for features, ensure MATLAB R2022b is installed and accessible from the command line for running the script. Download the pre-trained model files if not already present (the repo’s `download_models.py` can fetch PAT weights; the XGBoost .pkl are already in `mood_ml/`).

**2. Apple Health Data Parsing:** *Use a sample Apple Health export to test the parser.* Obtain an `export.xml` from an iPhone (the Apple Health app allows exporting data). Run the parser module to convert it:

* Execute `python reference_repos/apple-health-bot/dataParser/xmldataparser.py export.xml` as given in documentation. This should output CSV files (check the `apple-health-bot` docs for where output goes; likely in the current directory or a `./data` folder).
* Specifically retrieve the **sleep analysis CSV** (it may be named something like `SleepAnalysis.csv` or similar). Confirm it contains entries with start and end times. If the parser has known issues (their roadmap mentioned fixing sleep data handling), be prepared to apply quick fixes or parse the XML manually for sleep if needed.
* Retrieve other useful data: e.g., `StepCount.csv`, `HeartRate.csv` if available. These will be used for tsfresh features.

**3. Prepare Input Data Format:** *Convert parsed data into the format expected by the model.* This may involve writing a small script or notebook section to take the `SleepAnalysis.csv` and create an `example.csv` equivalent. For instance:

```python
import pandas as pd
sleep_df = pd.read_csv("SleepAnalysis.csv")
# Transform into columns: date, sleep_start, sleep_end, time_in_bed, minutes_sleep, minutes_awake
# (The exact logic depends on how Apple data is structured – possibly intervals of “InBed” vs “Asleep”)
```

We might need to aggregate sleep segments per night into one record (some nights have multiple entries if user was awake briefly). Sum up total minutes asleep and awake, compute time in bed, etc. The end result should align with `reference_repos/mood_ml/example.csv` structure.

**4. Feature Extraction Implementation:** *Generate features needed by XGBoost model.*

* **Option A (Using MATLAB):** Once `example.csv` is ready, run `matlab -batch "run('reference_repos/mood_ml/Index_calculation.m')"` (or via MATLAB GUI) to produce `test.csv`. Load `test.csv` in Python to confirm it has 36 feature columns plus a date.
* **Option B (Pure Python):** Write a Python function to calculate a subset of features. For example, implement calculation of sleep efficiency (`minutes_sleep/time_in_bed*100`), sleep onset delay (difference between bedtime and sleep start if provided), and use Python libraries for circadian metrics. One possible approach: use `tsfresh` on the sleep timeline (a time series of awake/asleep states over the night) to extract features like number of awakenings, etc. But given time, simpler is fine.
* Initially, do Option A to ensure the pipeline works, then plan to replace with Option B before deployment. Document any differences observed.

**5. Model Prediction Test:** *Load the pre-trained model and run a prediction on the features.* For example:

```python
import pickle
import pandas as pd
features = pd.read_csv("reference_repos/mood_ml/test.csv")
model = pickle.load(open("reference_repos/mood_ml/XGBoost_ME.pkl", 'rb'))
y_pred = model.predict_proba(features.drop(columns=['date']))  # if using XGBClassifier
print(y_pred)
```

This code (similar to the snippet in README) should output probabilities for each class. Verify that for the example data, the predictions match the `expected_outcome_me.csv` provided. If they match, we’ve confirmed the pipeline integrity. If not, debug if features misaligned (maybe column order or missing values).

**6. Integrate Pipeline in a Script or Service:** *Combine the above steps into a single Python script or a function.* This could be `predict_mood_from_healthkit(export_file)` that:

1. Calls the parser (maybe via subprocess or by importing the xmldataparser if it has a callable function).
2. Reads and transforms the data to features.
3. Loads models and produces predictions.
4. Returns the predictions (and maybe intermediate features for transparency).

Ensure this function can be easily invoked by an API route or CLI.

**7. Build FastAPI API (Phase 2 prep):** *Create the web service structure.* Set up FastAPI and define endpoints:

* `/healthcheck` (GET) to verify service is running.
* `/predict` (POST) to accept input. For now, the input might just be the raw file (we can accept a file upload using `UploadFile` in FastAPI) for simplicity. Or require JSON of already parsed data. Decide and implement parsing accordingly.
* In the endpoint logic, call the function from step 6 and return results as JSON.
* Test the API locally (e.g., using `uvicorn` to run and `curl` or a REST client to hit `/predict`). This will flush out any serialization issues (for instance, ensure numpy types are converted to Python native types for JSON).

**8. Gradio Interface for Testing:** *(optional)* Create a Gradio demo that wraps the function. This is as simple as:

```python
import gradio as gr
def demo_predict(file):
    # save uploaded file and call our pipeline
    result = predict_mood_from_healthkit(file.name)
    return result
gr.Interface(fn=demo_predict, inputs="file", outputs="json").launch()
```

This provides a quick way for stakeholders to try the system by uploading their Health data file. It’s not a priority deliverable, but useful for feedback and validation.

**9. Remove MATLAB Dependency (Planned Task):** *Start working on eliminating the MATLAB step.* This might not be completed in MVP, but it’s critical for shippability. Begin by examining what `Index_calculation.m` and `mnsd.p` do:

* Possibly consult the references given (Katori et al. 2022 for sleep phenotypes, Walch’s DLMO predictor code) to understand the math.
* Reimplement key components in Python. For example, DLMO prediction can be done by fitting a sine wave to activity or using known formulas from chronobiology; sleep regularity can be calculated via overlap of sleep between days, etc.
* We might integrate an existing circadian rhythm library if available, or use `numpy` and `pandas` to compute these features.
* Test the Python features vs the MATLAB ones on the example data (they should be close).
* This is an ongoing effort; we will schedule it such that by Phase 2 or 3, the entire pipeline is Python-only.

**10. Containerization:** Write a **Dockerfile** that encompasses everything for Phase 2:

* Base image: `python:3.9-slim` (for example).
* Copy the repository or at least the necessary parts (our app code, the `reference_repos/mood_ml` files for models, etc.).
* Install dependencies (maybe use the provided requirements.txt).
* If MATLAB is still needed at this point, that complicates container (likely we won’t include MATLAB in container; hence importance of step 9).
* Expose the FastAPI app. Test building and running the container locally. This ensures deployability.

**11. Database Schema & Integration Plan:** Although actual DB integration is Phase 3, begin designing the schema now:

* Users table (id, optional auth info or link to app’s auth).
* SleepData table (user\_id, date, sleep\_start, sleep\_end, etc.).
* Features table or extend SleepData with computed features (or compute on the fly as needed).
* Predictions table (user\_id, date, depression\_prob, mania\_prob, hypomania\_prob, generated\_at).
  This schema will help in Phase 3. For now, maybe just use a local SQLite to prototype storing and retrieving a few entries.

**12. Testing with Realistic Data:** Use any available dataset to test robustness. If the repository’s literature or references have sample data beyond `example.csv` (maybe the Harvard Fitbit study data isn’t provided, but we could simulate some patterns), feed it in. Particularly test edge cases: incomplete data (what if a day’s sleep is missing?), outliers (what if a user pulled an all-nighter?), different timezones, etc. Adjust the pipeline to handle these (e.g., if missing data, maybe carry forward last known values or flag that day as “insufficient data”).

**13. Documentation & API Spec:** Document how to use the system. Even in Phase 1, produce a README or wiki explaining:

* How to run the prediction script.
* Input data format (for developers or users).
* Explanation of outputs.
* In Phase 2, also produce API docs (FastAPI can generate OpenAPI JSON; we can create some human-friendly docs too).
  This will be useful for onboarding front-end developers and for future maintainers.

Each of these steps moves us closer to a shippable product. We will iterate, test, and refine at each step, keeping the focus on delivering a functioning feature at a time. The **immediate priority** is to get Phase 1’s core pipeline running and validated (steps 1–5), as this forms the backbone for everything else. Once we trust the predictions on known data, expanding outward to APIs and realtime is much easier.

By following this roadmap, we ensure that we’re not just writing about a solution but actively building and verifying it at each stage – aligning with the mission to prioritize **working software** that maintains the clinical-grade accuracy of the research and respects the privacy needs of health data. Each phase delivers a slice of value, culminating in a robust backend ready to integrate with a frontend and ultimately improve patient outcomes by forecasting mood swings before they happen.
