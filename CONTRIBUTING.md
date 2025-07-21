# Contributing to Big Mood Detector

We love your input! We want to make contributing to Big Mood Detector as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Any contributions you make will be under the Apache 2.0 Software License

In short, when you submit code changes, your submissions are understood to be under the same [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/Clarity-Digital-Twin/big-mood-detector/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/Clarity-Digital-Twin/big-mood-detector/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

* Run `make format` before committing
* Run `make lint` to check for issues
* Run `make test` to ensure tests pass
* Follow the existing code style (Clean Architecture patterns)

## 🎯 Current Priorities (v0.3.0)

### CRITICAL: Enable True Ensemble Predictions
**Context:** v0.2.0 only has XGBoost predictions. PAT provides embeddings but can't make mood predictions without classification heads.

**The Big Task: Train PAT Classification Heads**
1. Process NHANES data (already in `data/nhanes/2013-2014/`)
2. Train depression detection head on PAT embeddings
3. Wire PAT predictions into ensemble orchestrator
4. Deliver true dual-model predictions

**Quick Start for Contributors:**
```python
# Current (v0.2.0) - PAT only gives embeddings
pat_embeddings = pat.extract_features(activity)  # 96-dim vector

# Goal (v0.3.0) - PAT makes predictions  
pat_prediction = pat.predict_mood(activity)  # 0.82 depression risk
```

See [`docs/PAT_FINE_TUNING_ROADMAP.md`](docs/PAT_FINE_TUNING_ROADMAP.md) for detailed implementation guide.

### Other High-Priority Areas
1. **Clinical Validation** - Test on diverse populations
2. **Performance** - Optimize for very large Apple Health exports
3. **Wearables** - Add Fitbit, Garmin, Oura support
4. **Documentation** - Keep it accurate and honest

## Development Setup

### Prerequisites
- Python 3.12 or higher (we test on Python 3.12 in CI)
- Git
- Make (optional but recommended)

```bash
# Clone the repository
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
make setup

# Run tests
make test

# Start development server
make dev
```

## Testing

- Write tests for any new functionality
- Ensure all tests pass with `make test`
- Add integration tests for new features
- Test with real Apple Health data when possible

## Documentation

- Update the README.md if needed
- Add docstrings to all public functions
- Update API documentation for new endpoints
- Include examples in documentation

## License

By contributing, you agree that your contributions will be licensed under its Apache License 2.0.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)