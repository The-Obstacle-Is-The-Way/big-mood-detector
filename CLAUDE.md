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
make setup        # Initial project setup
git checkout -b feature/<feature-name>  # New feature branch
make dev          # Start development server with hot reload
```

### ✅ Testing & Validation

```bash
make test-watch        # Auto-run tests on file changes
make test-ml           # ML model validation
pytest -m clinical     # Clinical tests only
```

### 🧹 Code Quality

```bash
make quality      # Comprehensive checks (linting, typing, tests)
make format       # Auto-formatting
make type-check   # Ensure type safety
```

### 💾 Database Operations

```bash
make db-upgrade   # Apply migrations
make db-reset     # Reset database to clean state
```

## 📐 Architectural Guidelines

Respect the strict boundaries defined by Clean Architecture:

```
src/big_mood_detector/
├── domain/            # Core business logic (pure Python, no external deps)
├── application/       # Orchestration and use cases
├── infrastructure/    # Data access, parsers, ML inference
└── interfaces/        # API and CLI entry points
```

* **Dependency Direction**: Interfaces → Application → Domain ← Infrastructure
* **Use Repository & Factory Patterns** consistently.
* **Immutable Value Objects** (e.g., TimeRange, ClinicalThreshold) are critical.

## 🗃 Data Pipeline & ML Guidance

### Processing Steps

1. **Input**: XML/JSON Apple Health data (`apple_export/`, `health_auto_export/`)
2. **Parsing**: Use streaming parsers to handle large datasets
3. **Aggregation**: Clinically relevant daily summaries
4. **Feature Engineering**: Implement all 36 validated clinical features
5. **Prediction**: XGBoost primary, PAT Transformer for ensemble

### ML Model Integration

* **XGBoost**: AUC between 0.80-0.98
* **PAT Transformer**: Pre-trained model
