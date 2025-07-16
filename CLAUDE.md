# CLAUDE.md

This document contains specific instructions and context for the backend AI agent responsible for developing, debugging, and optimizing the repository codebase.

## 🎯 Core Objectives

Your primary goal is to:

1. **Write clean, maintainable, and test-driven code**.
2. **Adhere strictly to the provided Clean Architecture**.
3. **Optimize performance for large datasets** (multi-GB scale).
4. **Ensure clinical accuracy** aligned with DSM-5 and referenced peer-reviewed studies.

## 🚀 Paste-and-Run Development Commands

Use these commands verbatim to streamline your workflow:

### 🛠 Initial Setup & Development

```bash
# One-time setup
python3 -m venv .venv
source .venv/bin/activate  # Always activate first!
make setup                 # or: pip install -e ".[dev,ml,monitoring]"

# Daily workflow
source .venv/bin/activate  # Start of each session
git checkout -b feature/<feature-name>
make dev                   # Start development server
```

### ✅ Testing & Validation

```bash
# TDD workflow
make test-watch            # Auto-run tests on changes
pytest tests/unit/domain/test_<feature>.py -v  # Specific test

# Test categories
make test-fast             # Unit tests only
make test-ml               # ML model validation
pytest -m clinical         # Clinical tests only
pytest -m "not large"      # Skip large file tests
```

### 🧹 Code Quality

```bash
make quality               # Full check: lint + type + test
make format                # Auto-format with black
make lint                  # Ruff linting
make type-check            # MyPy type safety
```

### 💾 Database Operations

```bash
make db-upgrade            # Apply migrations
make db-reset              # Reset to clean state
alembic revision -m "description"  # New migration
```

## 📐 Architectural Guidelines

Respect the strict boundaries defined by Clean Architecture:

```
src/big_mood_detector/
├── domain/                # Core business logic (pure Python, no external deps)
│   ├── entities/          # SleepRecord, HeartRateRecord, ActivityRecord
│   ├── services/          # SleepWindowAnalyzer, ActivitySequenceExtractor
│   └── value_objects/     # Immutable: TimeRange, ClinicalThreshold
├── application/           # Orchestration and use cases
│   └── use_cases/         # ProcessHealthDataUseCase
├── infrastructure/        # Data access, parsers, ML inference
│   ├── parsers/           # XML (streaming) and JSON parsers
│   ├── repositories/      # Data persistence
│   └── ml/                # Model loading and inference
└── interfaces/            # API and CLI entry points
    ├── api/               # FastAPI routes
    └── cli/               # Typer commands
```

* **Dependency Direction**: Interfaces → Application → Domain ← Infrastructure
* **Repository Pattern**: Abstract data access in domain, implement in infrastructure
* **Factory Pattern**: Parser creation based on file type
* **Value Objects**: Immutable (frozen dataclasses) for thread safety

## 🗃 Data Pipeline & ML Guidance

### Processing Steps

1. **Input**: 
   - XML: `apple_export/export.xml` (520MB+, use streaming)
   - JSON: `health_auto_export/*.json` (smaller, direct parse)

2. **Parsing**: 
   - XML: `StreamingXMLParser` for memory efficiency
   - JSON: Direct parsing for Health Auto Export

3. **Aggregation**:
   - `SleepWindowAnalyzer`: Merge episodes within 3.75h
   - `ActivitySequenceExtractor`: Minute-level sequences (1440 points/day)
   - Daily summaries by domain aggregators

4. **Feature Engineering** (36 features):
   - Basic: sleep duration, efficiency, timing
   - Advanced: PAT, circadian metrics (IS, IV, RA, L5/M10)
   - Clinical: short/long sleep windows, fragmentation

5. **Prediction**:
   - XGBoost: Primary model (AUC 0.80-0.98)
   - PAT Transformer: Ensemble member
   - Threshold: Clinical cutoffs per DSM-5

### Key Implementation Status

✅ **Completed**:
- StreamingXMLParser (processes 520MB in 13s)
- Domain entities and basic aggregators
- SleepWindowAnalyzer (3.75h merging)
- ActivitySequenceExtractor (minute-level)
- Clinical feature extraction framework

🚧 **In Progress**:
- Circadian rhythm features (IS, IV, RA, L5/M10)
- PAT calculation refinement
- ML model integration

## 🧪 Testing Philosophy

1. **TDD Red-Green-Refactor**:
   ```python
   # 1. Write failing test
   def test_new_feature():
       assert expected == actual  # FAIL
   
   # 2. Implement minimum to pass
   # 3. Refactor for clean code
   ```

2. **Test Pyramid**:
   - Many unit tests (domain logic)
   - Some integration tests (parsers, DB)
   - Few E2E tests (full pipeline)

3. **Clinical Validation**:
   - Test against known patterns
   - Validate thresholds from papers
   - Edge cases (missing data, outliers)

## 📊 Performance Targets

- XML parsing: < 100MB RAM for any file size
- Processing rate: > 50,000 records/second
- Daily aggregation: < 1 second per year of data
- ML inference: < 100ms per prediction
- API response: < 200ms for feature extraction

## 🔍 Debugging Tips

```bash
# Memory profiling
mprof run python scripts/process_large_xml.py
mprof plot

# Line profiling
kernprof -l -v scripts/bottleneck_script.py

# SQL query analysis
export SQLALCHEMY_ECHO=true

# Structured logging
export LOG_LEVEL=DEBUG
```

## 📚 Key Papers to Reference

1. **Seoul National Study**: 3.75h sleep window merging
2. **Harvard/Fitbit Study**: 36 features for XGBoost
3. **PAT Transformer**: Activity sequence analysis
4. **DSM-5**: Clinical thresholds for bipolar disorder

Remember: **Clinical accuracy > Feature complexity > Performance**