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
test:
	pytest --cov=big_mood_detector --cov-report=term-missing --cov-report=html

test-parallel:
	pytest -n auto --cov=big_mood_detector --cov-report=term-missing

test-fast:
	pytest -m "not slow and not integration" --cov=big_mood_detector -n auto

test-slow:
	pytest -m "slow"

test-integration:
	pytest -m "integration" -n auto

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

# Code quality targets
lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	black .
	ruff check . --fix --select I

type-check:
	mypy src/big_mood_detector

# Combined quality check (CI/CD ready)
quality: lint type-check test
	@echo "✅ All quality checks passed!"

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
	docker build -t big-mood-detector .

docker-run:
	docker run -p 8000:8000 big-mood-detector

docker-dev:
	docker-compose up --build

# Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build

# Utility targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

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

# Full CI/CD pipeline (what runs in GitHub Actions)
ci: install-dev quality security-scan validate-models
	@echo "✅ CI pipeline completed successfully!" 