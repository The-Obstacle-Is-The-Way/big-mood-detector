# Developer Documentation

> [← Back to main README](../../README.md)

Technical documentation for developers working with Big Mood Detector.

## 🏗️ Quick Links

- **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** - System design & patterns
- **[API Reference](API_REFERENCE.md)** - REST endpoints & examples
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production setup

## Architecture Highlights

### Clean Architecture
```
Domain (Pure Python) → Application (Use Cases) → Infrastructure (External)
                ↑                           ↑
            CLI/API ←───────────────────────┘
```

### Key Components
- **Temporal Separation** - PAT (current state) vs XGBoost (future risk)
- **Streaming Parser** - Handles 500MB+ files with <100MB RAM
- **PyTorch PAT** - Pure PyTorch implementation matching paper (0.56 AUC)
- **Personal Baselines** - Adapts to individual patterns

## Development Setup

```bash
# Clone and setup
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml,monitoring]"

# Run tests
make test        # 976 tests
make type-check  # mypy clean
make lint        # ruff clean
make quality     # all checks
```

## Performance Benchmarks

- **XML Processing**: 33MB/s (>40k records/second)
- **Full Pipeline**: 17.4s for 365 days
- **Memory**: <100MB constant
- **API Response**: <200ms p95

## ML Model Integration

### Current Models
- **XGBoost**: Pre-trained weights from Seoul study (0.80-0.98 AUC)
- **PAT-S**: PyTorch, 0.56 AUC depression (matches paper)
- **PAT-M**: PyTorch, 0.54 AUC depression
- **PAT-L**: Training in progress

### Adding New Models
See [Model Integration Guide](model_integration_guide.md) for:
- Implementing `MoodPredictor` interface
- Weight management
- Performance requirements
- Testing guidelines

## API Integration

```python
# Quick example
import requests

# Process data
response = requests.post(
    "http://localhost:8000/api/v1/process",
    files={"file": open("export.xml", "rb")}
)

# Get predictions
predictions = requests.get(
    f"http://localhost:8000/api/v1/predict/{response.json()['user_id']}"
)
```

Full examples in [API Reference](API_REFERENCE.md).

## Additional Resources

- **[Model Weights](MODEL_WEIGHT_ARCHITECTURE.md)** - ML model management
- **[Dual Pipeline](DUAL_PIPELINE_ARCHITECTURE.md)** - JSON/XML processing
- **[Git Workflow](GIT_WORKFLOW.md)** - Contribution process
- **[Security](SECURITY.md)** - Security considerations

## For AI Agents

See [CLAUDE.md](../../CLAUDE.md) for:
- Current implementation status
- Critical bug fixes
- Performance optimizations
- Architecture decisions

---

*For user guides, see [User Documentation](../user/README.md)*  
*For clinical context, see [Clinical Documentation](../clinical/README.md)*