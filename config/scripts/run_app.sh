#!/bin/bash

# Kill any existing Streamlit processes
echo "Killing existing Streamlit processes..."
lsof -ti:8501,8502,8503 | xargs kill -9 2>/dev/null || true

# Set up Python path and environment
export PYTHONPATH="${PYTHONPATH}:/Users/joshcopp/Desktop/v3watchdog_ai/src"

# Start Streamlit
echo "Starting Streamlit app..."
streamlit run src/watchdog_ai/ui/pages/main_app.py --server.port 8503 