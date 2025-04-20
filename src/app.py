"""
Main application entry point for Watchdog AI.
"""

import streamlit as st
import logging
import sys
import os
from pathlib import Path
from watchdog_ai.ui.pages.data_insights_tab import render_data_insights_tab as render_data_tab
from watchdog_ai.ui.pages.chat_tab import render as render_chat_tab
from watchdog_ai.config import SessionKeys
from watchdog_ai.ui.components.chat_interface import ChatInterface

# Add the src directory to the Python path
src_dir = Path(__file__).parent
sys.path.append(str(src_dir))

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Watchdog AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from watchdog_ai.ui.components.data_uploader import render_data_uploader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    try:
        chat_interface = ChatInterface()
        chat_interface.render_chat_interface()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()