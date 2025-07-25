.PHONY: help install install-dev test test-fast test-slow test-integration lint format type-check clean build run dev docs pre-commit setup

# Default target
help:
	@echo "Big Mood Detector - Clinical-Grade Backend"
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Initial project setup (install + pre-commit)"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install all dependencies including dev tools"
	@echo "  test           - Run all tests with coverage"
	@echo "  test-fast      - Run fast unit tests only"
	@echo "  test-slow      - Run slow/integration tests"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-ml        - Run ML model tests"
	@echo "  lint           - Run code linting (ruff)"
	@echo "  format         - Format code (black + ruff)"
	@echo "  type-check     - Run type checking (mypy)"
	@echo "  quality        - Run all quality checks (lint + type + test)"
	@echo "  clean          - Clean build artifacts and cache"
	@echo "  build          - Build the package"
	@echo "  run            - Run the FastAPI server (development)"
	@echo "  run-prod       - Run the FastAPI server (production)"
	@echo "  dev            - Run development server with auto-reload"
	@echo "  docs           - Build and serve documentation"
	@echo "  pre-commit     - Install pre-commit hooks"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml,monitoring]"

setup: install-dev pre-commit
	@echo "✅ Project setup complete!"

# Testing targets (TDD focused)
test:  ## Run all tests except slow_finetune and large files
	$(PYTEST) -n auto --cov=big_mood_detector --cov-report=term-missing

test-quick:  ## Quick test run with minimal output
	pytest -x --tb=short -q

test-verbose:  ## Run tests with verbose output (best for debugging)
	pytest -v --tb=short --durations=10

test-parallel:
	pytest -n auto --cov=big_mood_detector --cov-report=term-missing

test-fast:
	pytest -m "not slow and not integration and not slow_finetune" -x --tb=short

test-unit:
	pytest tests/unit -x --tb=short -q

test-slow: ## Run slow performance tests (requires --runslow flag)
	pytest --runslow -m "slow" -v

test-slow-finetune:
	pytest -m "slow_finetune" -n 0 --durations=5

test-integration:
	pytest -m "integration" -n auto

test-full:  ## Run FULL test suite including slow/large tests (serial execution)
	pytest -n 0 --durations=10

test-ml:
	pytest -m "ml" --cov=big_mood_detector.ml

test-clinical:
	pytest -m "clinical" --verbose

# Watch mode for TDD
test-watch:
	pytest-watch --clear --onpass="echo '✅ Tests passed'" --onfail="echo '❌ Tests failed'"

# Parallel TDD watch mode
test-watch-parallel:
	pytest-watch -n auto --clear --onpass="echo '✅ Parallel tests passed'" --onfail="echo '❌ Parallel tests failed'"

# ---------------- Quality targets ----------------
# Platform-aware paths
PY              ?= python
# Check for virtual environment and use appropriate paths
VENV_EXISTS := $(shell test -d .venv-wsl && echo yes || echo no)
ifeq ($(VENV_EXISTS),yes)
    # Local WSL environment
    RUFF            := .venv-wsl/bin/ruff
    MYPY            := .venv-wsl/bin/mypy
    DMYPY           := .venv-wsl/bin/dmypy
    PYTEST          := .venv-wsl/bin/pytest
else
    # CI environment (packages installed globally)
    RUFF            := ruff
    MYPY            := mypy
    DMYPY           := dmypy
    PYTEST          := pytest
endif

# Lint (autosort imports, fix whitespace; skip archived dirs)
lint:           ## Ruff + isort/black fixes
	$(RUFF) check . --fix --select I --exclude scripts/archive,docs/archive,logs,*.egg-info

lint-fix:
	$(RUFF) check . --fix --exclude scripts/archive,docs/archive,logs,*.egg-info

format:
	black .
	$(RUFF) check . --fix --select I --exclude scripts/archive,docs/archive,logs,*.egg-info

# Static typing (daemon if available, config in mypy.ini)
type-check:     ## mypy fast path
	@if [ -x "$(DMYPY)" ]; then \
		$(DMYPY) run -- --config-file mypy.ini src ; \
	else \
		$(MYPY) --config-file mypy.ini src ; \
	fi

# Combined quality check (CI/CD ready)
quality: lint type-check test-fast
	@echo "✅ All quality checks passed!"

quality-full: quality test-slow-finetune
	@echo "✅ Full quality suite passed!"

# 🔥 Development server targets (2025 enhanced)
run:
	uvicorn big_mood_detector.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn big_mood_detector.main:app --host 0.0.0.0 --port 8000 --workers 4

dev: 
	@echo "🚀 Starting with browser hot reload (HTML/CSS/JS)..."
	uvicorn big_mood_detector.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug --reload-include '*.html' --reload-include '*.css' --reload-include '*.js'

dev-hot:
	@echo "⚡ Starting with Jurigged live code patching..."
	jurigged -v uvicorn big_mood_detector.main:app --reload --host 0.0.0.0 --port 8000

dev-agents:
	@echo "🤖 Starting with AI agent dependencies loaded..."
	pip install -e ".[dev,agents]" && uvicorn big_mood_detector.main:app --reload --host 0.0.0.0 --port 8000

# Database targets
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-migration:
	@read -p "Migration name: " name; \
	alembic revision --autogenerate -m "$$name"

db-reset:
	alembic downgrade base && alembic upgrade head

# Docker targets
docker-build:
	docker build -t big-mood-detector:latest .

docker-run:
	docker run -p 8000:8000 \
		-v $(PWD)/data/input:/app/data/input:ro \
		-v $(PWD)/data/output:/app/data/output \
		-v $(PWD)/model_weights:/app/model_weights:ro \
		big-mood-detector:latest

docker-dev:
	docker-compose up --build

docker-test:
	docker run --rm big-mood-detector:latest pytest

docker-clean:
	docker-compose down -v
	docker system prune -f

# Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build

# Utility targets
clean:
	coverage erase
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage*" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	# Clean up log files in logs directory
	rm -f logs/*.log
	# Clean up any temporary files
	rm -f *.tmp *.temp

build: clean
	python -m build

pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Health checks
health-check:
	@curl -f http://localhost:8000/health || echo "❌ Health check failed"

# Load testing (for clinical performance validation)
load-test:
	@echo "Running load tests for clinical performance validation..."
	# Add your load testing command here (e.g., locust, artillery)

# ML Model validation
validate-models:
	@echo "Validating ML models against clinical benchmarks..."
	python -m big_mood_detector.validation.clinical_validation

# Security scanning
security-scan:
	bandit -r big_mood_detector/
	safety check

# Security & Audit
.PHONY: security-audit
security-audit: ## Run comprehensive security audit
	@echo "🔍 Running security audit..."
	pip-audit --desc
	@echo "\n📊 Checking for outdated packages..."
	pip list --outdated
	@echo "\n📋 Security documentation: docs/SECURITY.md"

.PHONY: security-fix
security-fix: ## Auto-fix security vulnerabilities where possible
	@echo "🔧 Attempting to fix security vulnerabilities..."
	pip-audit --desc --fix

# Full CI/CD pipeline (what runs in GitHub Actions)
ci: install-dev quality security-scan validate-models
	@echo "✅ CI pipeline completed successfully!" 

# Git Workflow
.PHONY: git-status git-sync git-dev git-staging git-main
git-status: ## Show git status and branch info
	@echo "📊 Git Status:"
	@git status --short --branch
	@echo "\n🌳 All branches:"
	@git branch -a

git-sync: ## Sync current branch with origin
	@echo "🔄 Syncing current branch with origin..."
	@git pull origin $(shell git branch --show-current)

git-dev: ## Switch to development branch
	@echo "🛠 Switching to development branch..."
	@git checkout development
	@git pull origin development

git-staging: ## Switch to staging branch  
	@echo "🧪 Switching to staging branch..."
	@git checkout staging
	@git pull origin staging

git-main: ## Switch to main branch
	@echo "🚀 Switching to main branch..."
	@git checkout main
	@git pull origin main 