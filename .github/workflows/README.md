# CI/CD Pipeline Documentation

## Overview

Our CI/CD pipeline follows a three-stage deployment model:

```
development → staging → main (production)
```

## Branch Strategy

### Development Branch
- Active development happens here
- All feature branches merge into development
- Runs fast tests on every push
- No deployment (local development only)

### Staging Branch
- Integration testing environment
- Merges from development only
- Runs full test suite + performance benchmarks
- Deploys to staging environment for QA

### Main Branch (Production)
- Production-ready code only
- Merges from staging only (enforced by CI)
- Runs security audits + clinical validation + full regression
- Requires manual approval for production deployment
- Tagged releases trigger Docker Hub push

## Workflows

### 1. Development CI (`ci.yml`)
- **Triggers**: Push to development, PRs to any branch
- **Jobs**:
  - Lint (Ruff + MyPy)
  - Fast unit tests
  - API smoke tests
  - Security scan (informational)
  - Docker build test

### 2. Staging Deployment (`staging.yml`)
- **Triggers**: Push to staging, PRs to staging
- **Jobs**:
  - Full test suite with coverage
  - Integration tests (PostgreSQL + Redis)
  - Performance benchmarks
  - Docker build and test
  - Deploy to staging environment

### 3. Production Deployment (`production.yml`)
- **Triggers**: Push to main, version tags (v*)
- **Jobs**:
  - Security audit (blocking)
  - Clinical validation tests
  - Full regression tests
  - Docker build and push to registry
  - Create GitHub release (for tags)
  - Deploy to production (requires approval)

### 4. Merge Flow Validation (`merge-flow.yml`)
- **Triggers**: All pull requests
- **Purpose**: Enforces proper merge flow
- **Rules**:
  - ❌ Blocks direct merges to main (except from staging)
  - ⚠️  Warns about non-standard merges to staging
  - 🏷️  Auto-labels PRs based on target branch

### 5. Additional Workflows
- **Nightly Tests** (`nightly-tests.yml`): Runs extended test suite
- **Docker Tests** (`docker-tests.yml`): Validates Dockerfile changes
- **Clean Repo Check** (`check-clean-repo.yml`): Ensures no unwanted files

## Deployment Process

### Feature Development
1. Create feature branch from `development`
2. Make changes and test locally
3. Open PR to `development`
4. CI runs, PR gets reviewed
5. Merge to `development`

### Staging Release
1. Open PR from `development` to `staging`
2. Full test suite runs
3. Review and merge
4. Automatic deployment to staging
5. QA testing in staging environment

### Production Release
1. Open PR from `staging` to `main`
2. Security + clinical + regression tests run
3. Requires approval from maintainers
4. Merge triggers Docker image build
5. Manual approval required for deployment
6. Tag release for version tracking

## Environment Variables

### Required Secrets
- `CODECOV_TOKEN`: For coverage reporting
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password

### Optional Configuration
- `ENABLE_ASYNC_UPLOAD`: Enable async file uploads
- `LOG_LEVEL`: Set logging verbosity

## Best Practices

1. **Never skip the staging phase** - It catches integration issues
2. **Tag production releases** - Use semantic versioning (v1.2.3)
3. **Monitor performance benchmarks** - Catch regressions early
4. **Review security audits** - Address vulnerabilities before production
5. **Document breaking changes** - Update CHANGELOG.md

## Troubleshooting

### MyPy version mismatch
- We pin MyPy to 1.13.0 for consistency
- Run `pip install -e ".[dev]"` to match CI environment

### Test timeouts
- CI forces `--p no:timeout -o addopts=""` to disable timeouts
- For long-running tests, use appropriate markers

### Docker build failures
- Check Docker Hub rate limits
- Ensure base images are accessible
- Validate Dockerfile syntax locally

## Future Improvements

- [ ] Add Kubernetes deployment manifests
- [ ] Implement blue-green deployments
- [ ] Add automated rollback on failure
- [ ] Integrate with monitoring (Datadog/New Relic)
- [ ] Add load testing before production