#!/bin/bash

# Run script for Watchdog AI V3
# This script starts the Streamlit application and handles environment setup

# Create necessary directories
mkdir -p profiles

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit is not installed. Installing now..."
    pip install streamlit pandas numpy matplotlib altair
fi

# Run the application
echo "Starting Watchdog AI V3..."
streamlit run src/app.py