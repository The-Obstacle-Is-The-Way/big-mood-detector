# Setup GitHub Actions CI/CD Workflows

## Context

The project has no automated CI/CD workflows. With 700+ tests and a production-ready backend, we need automated quality gates.

## Required Workflows

### 1. Pull Request Checks
```yaml
# .github/workflows/pr-checks.yml
name: PR Checks
on:
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev,ml,monitoring]"
      - name: Run quality checks
        run: make quality
      - name: Run fast tests
        run: make test-fast
```

### 2. Daily Full Test Suite
```yaml
# .github/workflows/daily-tests.yml
name: Daily Full Tests
on:
  schedule:
    - cron: '0 0 * * *'
  workflow_dispatch:

jobs:
  full-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run all tests including slow
        run: |
          pip install tensorflow  # For PAT tests
          make test
```

### 3. Docker Build & Push
```yaml
# .github/workflows/docker-build.yml
name: Docker Build
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          push: true
          tags: ghcr.io/clarity-digital-twin/big-mood-detector:latest
```

## Benefits
- Catch issues before merge
- Ensure consistent code quality
- Automated Docker builds
- Test matrix across Python versions
- Cache dependencies for faster runs

## Additional Considerations
- Set up test coverage reporting
- Add badge to README
- Configure branch protection rules
- Set up dependency updates (Dependabot)

@claude Please create comprehensive GitHub Actions workflows that:
1. Run on every PR with fast tests and quality checks
2. Run full test suite daily (including slow tests)
3. Build and publish Docker images on main branch
4. Cache dependencies for performance
5. Support matrix testing across Python 3.10, 3.11, 3.12