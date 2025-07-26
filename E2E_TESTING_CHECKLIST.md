# 🧪 End-to-End Testing Checklist - Big Mood Detector

## Purpose
This checklist ensures all components work together correctly with real data before shipping. While unit tests verify individual components, E2E tests verify the complete user journey from input to output.

## Testing Environment
- Date: July 26, 2025
- Version: 0.4.0
- Python: 3.12 (WSL2)
- Models: XGBoost (production) + PAT-Conv-L (0.5929 AUC)

## 1. Data Input Testing ✅❌

### XML Processing (Apple Health Export)
- [ ] Small file (<10MB) - Should complete in <5s
- [ ] Medium file (50-100MB) - Should complete in <30s  
- [ ] Large file (500MB+) - Should complete in <2 min
- [ ] Very large file (1GB+) - Should use streaming, <100MB memory
- [ ] Corrupted XML - Should fail gracefully with clear error
- [ ] Empty XML - Should report "no health data found"

### JSON Processing (Health Auto Export)
- [ ] Single metric file (Heart Rate.json)
- [ ] Multiple metric files in directory
- [ ] Mixed JSON + XML in same directory
- [ ] Missing required metrics - Should warn but continue
- [ ] Malformed JSON - Should fail gracefully

### Date Range Filtering
- [ ] --days-back 30 - Should process only last 30 days
- [ ] --date-range 2025-01-01:2025-03-31 - Should respect range
- [ ] Future dates - Should ignore or warn
- [ ] No data in range - Should report clearly

## 2. Feature Extraction Pipeline ✅❌

### Sleep Analysis
- [ ] Normal sleep patterns (7-9 hours) - Correct duration
- [ ] Fragmented sleep - Should merge windows <3.75h apart
- [ ] No sleep data - Should use defaults, warn user
- [ ] 24+ hour "sleep" - Should cap or flag as anomaly

### Activity Data
- [ ] Steps + distance correlation
- [ ] Exercise minutes aggregation
- [ ] Missing activity days - Should interpolate or warn
- [ ] Extreme values (50k+ steps) - Should flag

### Heart Metrics
- [ ] Resting heart rate trends
- [ ] HRV calculation
- [ ] Missing HR data - Should handle gracefully
- [ ] Abnormal values (<30 or >200) - Should filter

### Circadian Rhythm
- [ ] Sleep midpoint calculation
- [ ] DLMO estimation (if enabled)
- [ ] Jet lag detection
- [ ] Shift work patterns

## 3. ML Model Pipeline ✅❌

### XGBoost Predictions
- [ ] All 36 Seoul features present
- [ ] Risk scores between 0-1
- [ ] Confidence scores reasonable
- [ ] Personal baselines applied (if user_id provided)

### PAT Depression Assessment
- [ ] 7-day activity sequence (10,080 points)
- [ ] NHANES normalization applied
- [ ] Depression probability 0-1
- [ ] Model loads successfully
- [ ] CUDA/CPU fallback works

### Temporal Ensemble (🚧 NOT YET INTEGRATED)
- [ ] PAT assesses current state
- [ ] XGBoost predicts tomorrow
- [ ] No averaging of time windows
- [ ] Both results displayed clearly

## 4. Output Generation ✅❌

### CLI Output
```bash
# Test command
python src/big_mood_detector/main.py predict data/sample_export.xml --report
```

Expected output:
- [ ] Summary shows all risk scores
- [ ] Confidence displayed
- [ ] Days analyzed count correct
- [ ] Warnings displayed if any

### API Endpoints
```bash
# Start server
python src/big_mood_detector/main.py serve

# Test health endpoint
curl http://localhost:8000/health

# Test depression prediction
curl -X POST http://localhost:8000/predictions/depression \
  -H "Content-Type: application/json" \
  -d '{"activity_sequence": [0.0, 1.0, 2.0, ...]}'  # 10,080 values
```

- [ ] /health returns {"status": "healthy"}
- [ ] /predictions/depression returns probability
- [ ] Invalid input returns 422 with clear error
- [ ] Model not loaded returns 503

### Report Generation
- [ ] Clinical report saves to file
- [ ] PDF generation (if matplotlib available)
- [ ] CSV export contains all predictions
- [ ] JSON export preserves full structure

## 5. Error Handling & Edge Cases ✅❌

### Missing Data Scenarios
- [ ] No sleep data → Uses population defaults
- [ ] No activity data → Flags low confidence  
- [ ] Partial data (3 days) → Refuses prediction
- [ ] Gap in data → Interpolates or warns

### System Resources
- [ ] Memory usage stays <2GB for 1 year of data
- [ ] CPU usage reasonable (no infinite loops)
- [ ] Disk space checks before writing
- [ ] Handles full disk gracefully

### User Errors
- [ ] Wrong file format → Clear error message
- [ ] Invalid date range → Helpful suggestion
- [ ] Missing permissions → Actionable error
- [ ] Network issues (for API) → Timeout handling

## 6. Integration Testing ✅❌

### Personal Calibration Flow
```bash
# First run - builds baseline
python src/big_mood_detector/main.py predict data/export.xml --user-id "test_user"

# Second run - uses baseline
python src/big_mood_detector/main.py predict data/export_new.xml --user-id "test_user"
```

- [ ] Baseline created on first run
- [ ] Baseline loaded on second run
- [ ] Predictions adjust to personal norms
- [ ] Different users isolated

### Ensemble Mode (When Ready)
```bash
python src/big_mood_detector/main.py predict data/export.xml --ensemble
```

- [ ] Both models load successfully
- [ ] PAT extracts embeddings
- [ ] Temporal separation maintained
- [ ] Performance acceptable (<5s overhead)

## 7. Performance Benchmarks ✅❌

### Processing Speed
- [ ] XML parsing: >30MB/s
- [ ] Feature extraction: <1s per year
- [ ] Model inference: <100ms
- [ ] API response: <200ms average

### Resource Usage
- [ ] Memory: <100MB for streaming
- [ ] CPU: Efficient multi-core usage
- [ ] Disk I/O: Minimal temp files
- [ ] Network: API responds quickly

## 8. Clinical Validation ✅❌

### Risk Thresholds
- [ ] Depression >0.7 = HIGH risk flag
- [ ] Hypomania >0.7 = HIGH risk flag  
- [ ] Mania >0.7 = URGENT flag
- [ ] Low confidence = Clear disclaimer

### Temporal Patterns
- [ ] Rapid cycling detected
- [ ] Seasonal patterns noted
- [ ] Med adherence impacts visible
- [ ] Lifestyle factors reflected

## Test Data Sources

1. **Synthetic Data** - `/tests/fixtures/`
2. **Sample Export** - `data/sample_export.xml` (if available)
3. **NHANES Cache** - `data/cache/nhanes_pat_data_subsetNone.npz`
4. **Your Own Data** - Best real-world test!

## Running the Full Test Suite

```bash
# 1. Clean environment
make clean
make install

# 2. Run all unit tests
make test

# 3. Type checking
make type-check

# 4. Linting
make lint

# 5. Run E2E with sample data
./scripts/run_e2e_tests.sh  # Create this script

# 6. Performance test
time python src/big_mood_detector/main.py process data/large_export.xml

# 7. API integration test
python src/big_mood_detector/main.py serve &
./scripts/test_api_endpoints.sh  # Create this script
```

## Sign-off Checklist

Before declaring "ready to ship":

- [ ] All critical paths tested with real data
- [ ] Performance meets targets
- [ ] Error messages are helpful
- [ ] Documentation updated
- [ ] No hard-coded paths
- [ ] Secrets not exposed
- [ ] Docker image builds
- [ ] README accurate
- [ ] --help commands useful
- [ ] Version number updated

## Known Issues (Document Here)

1. **Temporal Ensemble Not Integrated** - Using deprecated EnsembleOrchestrator
2. **PAT Only Does Depression** - No medication or other predictions yet
3. **Large Files** - May timeout in some environments (use direct Python)

---

*Last tested: [DATE]*
*Tested by: [NAME]*
*Test data location: [PATH]*