"""
Main application entry point for Watchdog AI.
"""

import streamlit as st
import logging
from ui.pages.modern_ui import modern_analyst_ui  # Changed from src.ui.pages to ui.pages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    try:
        # Run the modern UI
        modern_analyst_ui()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        if st.button("Show Error Details"):
            st.exception(e)

if __name__ == "__main__":
    main()