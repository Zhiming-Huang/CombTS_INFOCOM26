#!/bin/bash
# =============================================================================
# Docker Setup Test Script
# =============================================================================
# Quick test to verify Docker image builds and runs correctly
# 
# Usage: bash test_docker.sh
# =============================================================================

set -e

echo "🐳 Testing Docker setup for CombTS INFOCOM26..."
echo ""

# Check Docker
echo "1. Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon not running. Please start Docker."
    exit 1
fi

echo "✅ Docker is ready"
echo ""

# Build test image
echo "2. Building test image (minimal test)..."
docker build -t combts-test:latest . \
    --target=runtime \
    --quiet \
    || docker build -t combts-test:latest . --quiet

echo "✅ Image built successfully"
echo ""

# Test container run
echo "3. Testing container execution..."
docker run --rm combts-test:latest python -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import scipy
print('✅ All dependencies working')
print(f'NumPy: {np.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'NetworkX: {nx.__version__}')
print(f'SciPy: {scipy.__version__}')
"

echo ""
echo "4. Testing volume mount..."
mkdir -p test_output
docker run --rm \
    -v "$(pwd)/test_output:/app/output" \
    combts-test:latest \
    python -c "
import os
with open('/app/output/test.txt', 'w') as f:
    f.write('Docker volume mount test successful!')
print('✅ Volume mount working')
"

if [ -f "test_output/test.txt" ]; then
    echo "✅ Volume mount verified"
    rm -rf test_output
else
    echo "❌ Volume mount failed"
    exit 1
fi

echo ""
echo "🎉 Docker setup test completed successfully!"
echo ""
echo "Ready to run experiments:"
echo "  bash run_docker_experiments.sh quick"
echo "  python run_docker_experiments.py quick"

# Cleanup test image
docker rmi combts-test:latest &> /dev/null || true 