#!/bin/bash
# Run tests for the research template
# Usage: bash scripts/run_tests.sh

set -e

source .env

echo "Running tests..."
pytest test/ -v --tb=short

echo ""
echo "All tests passed! ✓"
