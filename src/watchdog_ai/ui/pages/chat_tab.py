"""
Chat tab page for Watchdog AI.
"""

import streamlit as st
# Assuming data_uploader component exists and is importable
from watchdog_ai.ui.components.data_uploader import render_data_uploader
# Assuming chat_interface component exists and is importable
from watchdog_ai.ui.components.chat_interface import ChatInterface # Import the class
import logging
from watchdog_ai.config import SessionKeys # Import SessionKeys
import pandas as pd # Import pandas for type checking
from watchdog_ai.insights.insight_conversation import ConversationManager

logger = logging.getLogger(__name__)

# Instantiate ChatInterface - assuming it should be instantiated once per session/page load
# If it manages its own state heavily via st.session_state, this is okay.
# Otherwise, might need to manage its lifecycle differently (e.g., in session state).
chat_interface = ChatInterface()

def render():
    """Renders the Chat Analysis tab."""
    st.header("Chat Analysis")

    # Initialize conversation manager if not already done
    if SessionKeys.UPLOADED_DATA not in st.session_state:
        render_data_uploader()
        return

    data_loaded = (
        SessionKeys.UPLOADED_DATA in st.session_state and
        isinstance(st.session_state[SessionKeys.UPLOADED_DATA], pd.DataFrame) and
        not st.session_state[SessionKeys.UPLOADED_DATA].empty
    )

    if not data_loaded:
        st.warning("⚠️ Please upload data first.")
        st.markdown("You can upload data in the **Data & Insights** tab, or upload it here:")
        # Render the uploader component which should handle the upload and update session state
        render_data_uploader()
        # Optionally, stop execution here if upload is mandatory before chat
        # st.stop()
        # Or let the chat interface render its own disabled state

    # Always render the chat interface, it should handle the disabled state internally
    # based on SessionKeys.UPLOADED_DATA
    try:
        chat_interface.render_chat_interface() # Call the render method of the instance
    except Exception as e:
        logger.error(f"Error rendering chat interface: {e}", exc_info=True)
        st.error("An error occurred while loading the chat interface.")

# --- Old render function (if exists) comment out or remove ---
# def render_chat_tab():
#     st.header("Chat Analysis")
#     # Assuming ChatInterface handles data check internally now
#     chat_interface = ChatInterface()
#     chat_interface.render_chat_interface()