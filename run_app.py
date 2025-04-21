#!/usr/bin/env python3
"""
Run script for Watchdog AI application.
Ensures proper Python path setup and runs the Streamlit app.
"""

import os
import sys
import subprocess

def main():
    """Set up environment and run the Streamlit app."""
    # Add the src directory to Python path
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    sys.path.insert(0, src_path)
    
    # Set environment variables
    os.environ["PYTHONPATH"] = src_path
    
    # Run the Streamlit app
    subprocess.run([
        "streamlit",
        "run",
        os.path.join(src_path, "watchdog_ai/ui/pages/main_app.py"),
        "--server.port=8503",
        "--theme.base=dark"
    ])

if __name__ == "__main__":
    main()