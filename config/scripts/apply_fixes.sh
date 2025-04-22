#!/bin/bash
# Making this file executable
# chmod +x apply_fixes.sh
# Script to apply fixes for the "Error generating insight" issue in Watchdog AI

echo "=== Watchdog AI Fix Script ==="
echo "This script will apply fixes to resolve the 'Error generating insight' issue."

# Check Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found on your system."
    exit 1
fi

# Ensure .env file exists and has correct permissions
echo "Setting up environment file..."
python3 setup_env.py --use-mock

# Make file executable
chmod +x setup_env.py

# Install required dependencies
echo "Installing required dependencies..."
pip install openai==0.28 python-dotenv requests streamlit pandas numpy

# Check if the insight_conversation_enhanced.py file exists
if [ -f "src/insight_conversation_enhanced.py" ]; then
    echo "Enhanced conversation manager found."
else
    echo "Error: Enhanced conversation manager not found."
    exit 1
fi

# Check if the app_enhanced.py file exists
if [ -f "src/app_enhanced.py" ]; then
    echo "Enhanced app file found."
else
    echo "Error: Enhanced app file not found."
    exit 1
fi

echo ""
echo "=== Fix Applied Successfully ==="
echo "You can now run the application with the enhanced error handling:"
echo ""
echo "    cd $(pwd)"
echo "    streamlit run src/app_enhanced.py"
echo ""
echo "To use real API calls (instead of mock responses):"
echo "    ./setup_env.py --use-api --api-key YOUR_API_KEY --provider openai"
echo ""
echo "For more help:"
echo "    ./setup_env.py --help"
echo ""
