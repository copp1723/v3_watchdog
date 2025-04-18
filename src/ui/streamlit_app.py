"""
Main Streamlit application entry point for Watchdog AI.
"""

import streamlit as st
import os
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streamlit_app")

# Add the parent directory to the path so we can import modules properly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import UI components
from pages.modern_ui import modern_analyst_ui

def main():
    """
    Main function to run the Streamlit app.
    """
    try:
        # Run the modern UI
        modern_analyst_ui()
    except Exception as e:
        logger.error(f"Error in Streamlit app: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        if os.environ.get("DEBUG", "false").lower() == "true":
            st.exception(e)

if __name__ == "__main__":
    main()