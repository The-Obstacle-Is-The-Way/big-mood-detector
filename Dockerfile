# Multi-stage build for optimized image size - 2025 best practices
# Stage 1: Builder with all dependencies
FROM python:3.12-slim-bookworm AS builder

# Security: Run as non-root user
RUN useradd -m -u 1000 appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install production dependencies only
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir gunicorn

# Stage 2: Runtime image (smaller, secure)
FROM python:3.12-slim-bookworm AS runtime

# Install only required runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Set working directory
WORKDIR /app

# Copy less frequently changing files first (better caching)
# Copy configuration files (rarely change)
COPY --chown=appuser:appuser config/ ./config/

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Copy model weights (required for predictions)
COPY --chown=appuser:appuser model_weights/ ./model_weights/

# Create symlinks for expected model names
RUN ln -s /app/model_weights/xgboost/converted/XGBoost_DE.json /app/model_weights/xgboost/converted/depression_risk.json && \
    ln -s /app/model_weights/xgboost/converted/XGBoost_HME.json /app/model_weights/xgboost/converted/hypomanic_risk.json && \
    ln -s /app/model_weights/xgboost/converted/XGBoost_ME.json /app/model_weights/xgboost/converted/manic_risk.json

# Copy entrypoint and healthcheck scripts
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
COPY --chown=appuser:appuser docker/healthcheck.py /healthcheck.py
RUN chmod +x /entrypoint.sh /healthcheck.py

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Health check with dependency verification
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /healthcheck.py || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set up volume for data (user uploads/outputs)
VOLUME /data

# Set default data directory to /data for volume mounts
# This ensures all user data goes to the mounted volume
ENV BIGMOOD_DATA_DIR=/data
ENV DATA_DIR=/data

# Use entrypoint for flexible execution
ENTRYPOINT ["/entrypoint.sh"]