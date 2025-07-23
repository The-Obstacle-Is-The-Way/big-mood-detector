# Big Mood Detector - System Understanding Plan
Generated: 2025-07-23

## Goal
Trace the entire data pipeline from Apple Health ingestion to mood predictions, understanding every transformation and identifying where the feature name mismatch occurs.

## Research & Understanding Phase

### 1. Literature Review
- [ ] Review Seoul paper for exact feature definitions and names
- [ ] Review PAT transformer paper for input requirements
- [ ] Document the expected inputs/outputs for each model

### 2. Data Flow Tracing
Starting from user's primary use case: Apple Health XML → Predictions

#### Stage 1: Data Ingestion
- [ ] Trace XML parsing (streaming_parser.py)
- [ ] Trace JSON parsing (for alternate path)
- [ ] Document what raw records look like

#### Stage 2: Domain Entity Creation
- [ ] How raw XML/JSON becomes SleepRecord, ActivityRecord, HeartRateRecord
- [ ] What fields are extracted and transformed

#### Stage 3: Aggregation Pipeline
- [ ] How daily summaries are created
- [ ] What happens in SleepAggregator, ActivityAggregator, HeartRateAggregator
- [ ] Document the output format of each aggregator

#### Stage 4: Feature Extraction
- [ ] Trace ClinicalFeatureExtractor.extract_seoul_features()
- [ ] Document all 36 features we generate
- [ ] Find where our feature names are defined

#### Stage 5: Model Input Preparation
- [ ] Find where features are prepared for XGBoost
- [ ] Find where features are prepared for PAT
- [ ] Identify where the name mapping should happen

#### Stage 6: Model Inference
- [ ] Trace XGBoostModels.predict()
- [ ] Understand what feature names the model expects
- [ ] Find where the mismatch occurs

### 3. Current Implementation Analysis
- [ ] Map all the refactorings that have happened
- [ ] Identify what the Feature Engineering Orchestrator adds
- [ ] Understand the ensemble logic

### 4. Testing Infrastructure
- [ ] Review existing tests for feature extraction
- [ ] Review existing tests for model inference
- [ ] Identify gaps in test coverage

## Deliverables
1. Complete data flow diagram
2. Feature name mapping (our names vs Seoul names)
3. Identification of where the fix needs to happen
4. Test plan for validating the fix

## Questions to Answer
1. What are the exact 36 feature names the XGBoost model expects?
2. Where in our code do we generate our feature names?
3. Where should the mapping happen?
4. Why did this work before but not now?
5. Is the Feature Engineering Orchestrator involved in this issue?

## Next Steps After Understanding
1. Create failing tests that expose the issue
2. Implement the feature name mapping
3. Verify all models work with real data
4. Update documentation

This is a methodical approach - no rushing, just careful understanding.