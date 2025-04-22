"""
Main entry point for Watchdog AI application.

This module initializes and runs the Watchdog AI application,
including the UI and all related services.
"""

import streamlit as st
import os
import logging
import sys
from datetime import datetime
from src.watchdog_ai.ui.pages.main_app import render_app

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Watchdog AI - Data Insights",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create log directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Add file handler
log_file = os.path.join(logs_dir, f'watchdog_{datetime.now().strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Get root logger and add handler
logger = logging.getLogger("watchdog_ai")
logger.addHandler(file_handler)

if __name__ == "__main__":
    # Log startup
    logger.info("Starting Watchdog AI application")
    
    try:
        # Run the application
        logger.info("Launching Streamlit UI")
        render_app()
        
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        sys.exit(1)