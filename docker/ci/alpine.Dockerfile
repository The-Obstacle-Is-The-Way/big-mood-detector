FROM python:3.12-alpine AS build

# Install complete build toolchain including execinfo for stack traces
RUN apk add --no-cache \
    build-base gfortran cmake ninja \
    openblas-dev lapack-dev openmp-dev \
    autoconf automake libtool patchelf \
    libexecinfo-dev \
    libffi-dev openssl-dev libxml2-dev libxslt-dev \
    postgresql-dev git tzdata

WORKDIR /build

# Upgrade pip and use older setuptools to avoid PEP 517 issues
RUN pip install --upgrade pip wheel 'setuptools<70'

# CRITICAL: Use pre-built wheels from piwheels or alpine wheels when available
# This dramatically speeds up the build
RUN pip install --no-cache-dir \
    --extra-index-url https://alpine-wheels.github.io/index \
    numpy==1.26.4

# Install scipy without forcing source build if wheel is available
RUN pip install --no-cache-dir scipy==1.13.0

# For XGBoost and scikit-learn, we need to build from source
# But we can speed it up by setting build flags
ENV CFLAGS="-O2"
ENV CXXFLAGS="-O2"
RUN pip install --no-cache-dir --no-binary=xgboost xgboost==2.0.3
RUN pip install --no-cache-dir --no-binary=scikit-learn scikit-learn==1.5.0

# Now install our package
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -e .

# Runtime stage - minimal dependencies
FROM python:3.12-alpine

RUN apk add --no-cache \
    libgomp libexecinfo openblas \
    libxml2 libxslt postgresql-libs \
    tzdata

# Copy installed packages from build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

WORKDIR /app
COPY src/ ./src/

# Verify imports work
RUN python -c "import big_mood_detector, xgboost, sklearn, numpy, scipy; print('✅ Alpine imports OK')"

CMD ["python", "-m", "big_mood_detector.main", "--help"]