#!/usr/bin/env bash
# Test script to run tests inside Docker container (cross-platform testing)
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Docker test image...${NC}"
docker build --target builder -t big-mood-detector:test .

echo -e "${BLUE}Running tests in Docker container...${NC}"

# Run different test suites
test_suites=("unit" "integration")

for suite in "${test_suites[@]}"; do
    echo -e "\n${BLUE}Running $suite tests...${NC}"
    
    if docker run --rm \
        -e TZ=UTC \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$(pwd)":/workspace \
        -w /workspace \
        big-mood-detector:test \
        python -m pytest "tests/$suite" -v --color=yes; then
        echo -e "${GREEN}✓ $suite tests passed${NC}"
    else
        echo -e "${RED}✗ $suite tests failed${NC}"
        exit 1
    fi
done

echo -e "\n${GREEN}All tests passed in Docker!${NC}"