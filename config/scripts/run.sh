#!/bin/bash

# run.sh - Run the Watchdog AI Streamlit app

# Make the script executable if it's not already
chmod +x "$0"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set Python path to include the project root for imports
export PYTHONPATH="$(pwd):$(pwd)/src"
echo "PYTHONPATH: $PYTHONPATH"

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a
fi

# Check if the user wants to run tests
if [ "$1" = "test" ]; then
    echo "Running import test..."
    python test_imports.py
    exit 0
fi

# Print LLM configuration
echo "LLM Provider: ${LLM_PROVIDER:-openai}"
echo "Using Mock: ${USE_MOCK:-true}"
echo "[DEBUG] USE_MOCK is set to: $USE_MOCK"

# Run the Streamlit application
echo "Starting Streamlit app..."
streamlit run src/app.py
